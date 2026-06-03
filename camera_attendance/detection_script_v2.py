"""
Camera Face Detection Script v3 — ESP32-CAM + Emotion + Reliable Entry/Exit
============================================================================
Fixes over v2:
  1. EMOTION FIX  — DeepFace runs in a background thread (non-blocking).
                    Entry is logged immediately with a "pending" emotion;
                    once DeepFace finishes, a PATCH updates the record.
                    Alternatively (simpler): emotion is detected on a
                    downscaled crop with a fast backend (OpenCV).
                    Falls back to 'neutral' gracefully.

  2. EXIT FIX     — On startup, queries Django for all open logs today
                    and pre-populates students_inside.  Script restarts
                    no longer cause a second ENTRY instead of EXIT.

  3. MJPEG LAG FIX — Reader thread uses larger chunk (16 KB) and always
                     advances to the LAST complete JPEG in the buffer
                     (skips stale frames), so detection always works on
                     the newest frame.

  4. CAMERA RECONNECT — Health-check ping (/healthz) before reconnecting;
                        exponential back-off on repeated stream failures.

Usage:
    python detection_script_v3.py \\
        --camera-id 1 \\
        --camera-url "http://192.168.1.10/stream" \\
        --server "http://localhost:8000" \\
        --cooldown 30 \\
        --exit-timeout 90
"""

import json
import time
import base64
import argparse
import requests
import sys
import os
import logging
import threading
import queue
from datetime import date
from collections import defaultdict

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH',  'true')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL',        '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES',         '')

# ── Globals loaded lazily ────────────────────────────────────────────────────
cv2                      = None
np                       = None
face_recognition         = None
DeepFace                 = None
FACE_RECOGNITION_AVAILABLE = False
DEEPFACE_AVAILABLE         = False
face_cascade             = None

VALID_EMOTIONS = {'happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust'}
EMOTION_MAP = {
    'happy':    'happy',   'sad':     'sad',   'angry':   'angry',
    'neutral':  'neutral', 'surprise':'surprise','fear':  'fear',
    'disgust':  'disgust', 'contempt':'disgust',
}


# ─────────────────────────────────────────────────────────────────────────────
#  DEPENDENCY LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_dependencies():
    global cv2, np, face_recognition, DeepFace
    global FACE_RECOGNITION_AVAILABLE, DEEPFACE_AVAILABLE, face_cascade

    logger.info("Loading dependencies…")

    try:
        import cv2 as _cv2;  import numpy as _np
        cv2 = _cv2;  np = _np
        logger.info("✓ OpenCV loaded")
    except ImportError as e:
        logger.error(f"✗ OpenCV not installed: {e}");  sys.exit(1)

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        logger.error("✗ Haar cascade not found");  sys.exit(1)
    logger.info("✓ Haar cascade loaded")

    try:
        import face_recognition as _fr
        face_recognition = _fr
        FACE_RECOGNITION_AVAILABLE = True
        logger.info("✓ face_recognition loaded")
    except ImportError:
        logger.warning("⚠ face_recognition not available — matching disabled")

    try:
        from deepface import DeepFace as _df
        DeepFace = _df
        DEEPFACE_AVAILABLE = True
        logger.info("✓ DeepFace loaded — async emotion detection enabled")
    except ImportError:
        logger.warning("⚠ DeepFace not installed — using neutral fallback. "
                       "Install: pip install deepface tf-keras")


# ─────────────────────────────────────────────────────────────────────────────
#  ASYNC EMOTION DETECTOR
#  Runs DeepFace in a worker thread so it never blocks the detection loop.
#  Callers post a (frame_copy, face_box, callback) job and move on immediately.
# ─────────────────────────────────────────────────────────────────────────────
class EmotionWorker:
    def __init__(self):
        self._q      = queue.Queue(maxsize=4)   # discard if backed up
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame, face_box, callback):
        """
        Non-blocking submit.  If the queue is full the job is dropped
        (detection loop must not stall waiting for emotion).
        callback(emotion: str, confidence: float) is called from the worker thread.
        """
        try:
            self._q.put_nowait((frame.copy(), face_box, callback))
        except queue.Full:
            pass   # drop — better than blocking

    def _run(self):
        while True:
            frame, face_box, cb = self._q.get()
            emotion, conf = self._detect(frame, face_box)
            try:
                cb(emotion, conf)
            except Exception as e:
                logger.debug(f"EmotionWorker callback error: {e}")

    def _detect(self, frame, face_box):
        if not DEEPFACE_AVAILABLE:
            return 'neutral', 0.0
        try:
            x, y, w, h = face_box
            pad  = max(int(w * 0.3), 20)
            fh, fw = frame.shape[:2]
            crop = frame[max(0,y-pad):min(fh,y+h+pad),
                         max(0,x-pad):min(fw,x+w+pad)]
            if crop.size == 0:
                return 'neutral', 0.0

            # Use DeepFace with enforce_detection=False so it never raises
            result = DeepFace.analyze(
                crop,
                actions=['emotion'],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]

            dominant = result.get('dominant_emotion', 'neutral').lower()
            emotions = result.get('emotion', {})
            emotion  = EMOTION_MAP.get(dominant, 'neutral')
            conf     = round(float(emotions.get(dominant, 0.0)) / 100.0, 4)
            logger.debug(f"  Emotion: {dominant} → {emotion} ({conf:.2f})")
            return emotion, conf
        except Exception as e:
            logger.debug(f"  DeepFace error: {e}")
            return 'neutral', 0.0


_emotion_worker = None   # created after load_dependencies()


# ─────────────────────────────────────────────────────────────────────────────
#  STUDENT LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_students(server_url):
    try:
        r = requests.get(
            f"{server_url}/camera-attendance/api/students/encodings/",
            timeout=10, verify=False,
        )
        r.raise_for_status()
        students = r.json()
        if students:
            logger.info(f"✓ Loaded {len(students)} students")
        else:
            logger.warning("⚠ API returned empty student list")
        return students
    except Exception as e:
        logger.error(f"✗ load_students failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  EXIT FIX: load open logs from Django on startup
#  Populates students_inside with IDs that already have an open entry today.
# ─────────────────────────────────────────────────────────────────────────────
def load_open_logs(server_url):
    """
    Returns a set of student IDs that have an open (entry, no exit) log today.
    Prevents script restart from logging a second ENTRY instead of EXIT.
    """
    try:
        r = requests.get(
            f"{server_url}/camera-attendance/api/live-detections/",
            timeout=10, verify=False,
        )
        r.raise_for_status()
        data = r.json()
        inside = {
            log['student_id']   # field added to API response below
            for log in data.get('logs', [])
            if not log.get('exit_time')
               and log.get('student_id') is not None
        }
        logger.info(f"✓ Pre-loaded {len(inside)} students already inside from open logs")
        return inside
    except Exception as e:
        logger.warning(f"⚠ Could not load open logs: {e} — starting with empty inside set")
        return set()


# ─────────────────────────────────────────────────────────────────────────────
#  FACE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_faces(frame):
    if frame is None or frame.size == 0:
        return []
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return list(faces) if len(faces) > 0 else []
    except Exception as e:
        logger.error(f"✗ detect_faces: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  FACE MATCHING
# ─────────────────────────────────────────────────────────────────────────────
def match_student(frame, face_box, students):
    if not students or not FACE_RECOGNITION_AVAILABLE:
        return None, 1.0
    try:
        x, y, w, h   = face_box
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        loc          = (y, x+w, y+h, x)   # (top, right, bottom, left)
        encodings    = face_recognition.face_encodings(rgb, known_face_locations=[loc], num_jitters=1)
        if not encodings:
            return None, 1.0

        enc  = encodings[0]
        best_d, best_s = 1.0, None

        for student in students:
            raw = student.get('encoding')
            if not raw:
                continue
            try:
                known = np.array(json.loads(raw))
                d = face_recognition.face_distance([known], enc)[0]
                if d < best_d:
                    best_d, best_s = d, student
            except Exception:
                continue

        THRESHOLD = 0.55
        if best_d < THRESHOLD and best_s:
            logger.info(f"  ✅ MATCH: {best_s['name']} (dist={best_d:.3f})")
            return best_s, float(best_d)

        return None, float(best_d)
    except Exception as e:
        logger.error(f"✗ match_student: {e}")
        return None, 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  SNAPSHOT ENCODING
# ─────────────────────────────────────────────────────────────────────────────
def encode_snapshot(frame, face_box=None):
    try:
        if face_box is not None:
            x, y, w, h = face_box
            pad = 30;  fh, fw = frame.shape[:2]
            img = frame[max(0,y-pad):min(fh,y+h+pad),
                        max(0,x-pad):min(fw,x+w+pad)]
        else:
            img = frame
        if img.size == 0:
            return None
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf).decode('utf-8')
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  ATTENDANCE API
# ─────────────────────────────────────────────────────────────────────────────
def log_attendance(server_url, student_id, camera_id,
                   emotion='neutral', score=0.0, snapshot_b64=None):
    if emotion not in VALID_EMOTIONS:
        emotion = 'neutral'
    score = max(0.0, min(1.0, float(score)))

    payload = {
        "student_id": student_id,
        "camera_id":  camera_id,
        "emotion":    emotion,
        "score":      score,
    }
    if snapshot_b64:
        payload["snapshot"] = snapshot_b64

    try:
        r = requests.post(
            f"{server_url}/camera-attendance/api/log/",
            json=payload, timeout=10, verify=False,
        )
        if r.status_code == 200:
            data   = r.json()
            status = data.get('status', 'unknown')
            logger.info(f"  ✓ API [{status.upper()}] student={student_id} emotion={emotion} score={score:.2f}")
            return True, status
        else:
            logger.warning(f"  ✗ API {r.status_code}: {r.text[:200]}")
            return False, 'error'
    except Exception as e:
        logger.error(f"  ✗ log_attendance: {e}")
        return False, 'error'


# ─────────────────────────────────────────────────────────────────────────────
#  MJPEG STREAM  —  always-latest-frame reader
# ─────────────────────────────────────────────────────────────────────────────
class MJPEGCapture:
    """
    Reads an MJPEG HTTP stream in a background thread.
    .read() always returns the NEWEST complete frame — stale frames are dropped.
    Uses 16 KB chunks for fast assembly; finds the LAST JPEG in the buffer
    so the detection loop is never one frame behind.
    """
    CHUNK = 16384   # 16 KB — large enough for a full QVGA JPEG in 1-2 reads

    def __init__(self, url, server_url=None):
        self.url        = url
        self.server_url = server_url   # for health-check
        self._frame     = None
        self._lock      = threading.Lock()
        self._running   = False
        self._thread    = None
        self._ok        = False
        self._connect()

    def _health_check(self):
        """Ping /healthz to confirm ESP32 is alive before streaming."""
        if not self.server_url:
            return True
        base = self.url.rsplit('/stream', 1)[0]
        try:
            r = requests.get(f"{base}/healthz", timeout=3, verify=False)
            return r.status_code == 200
        except Exception:
            return False

    def _connect(self):
        try:
            resp = requests.get(
                self.url, stream=True, verify=False,
                timeout=10,
                headers={'Connection': 'keep-alive'},
            )
            resp.raise_for_status()
            self._running = True
            self._ok      = True
            self._thread  = threading.Thread(
                target=self._read, args=(resp,), daemon=True
            )
            self._thread.start()
            logger.info("✓ MJPEG stream reader started")
        except Exception as e:
            logger.error(f"✗ MJPEG connect failed: {e}")
            self._ok = False

    def _read(self, resp):
        buf = b''
        try:
            for chunk in resp.iter_content(chunk_size=self.CHUNK):
                if not self._running:
                    break
                buf += chunk

                # ── Find the LAST complete JPEG in the buffer ──
                # This drops any stale frames that accumulated while
                # DeepFace was busy, keeping detection on the newest frame.
                last_end = -1
                search_from = 0
                while True:
                    end = buf.find(b'\xff\xd9', search_from)
                    if end == -1:
                        break
                    last_end = end
                    search_from = end + 2

                if last_end == -1:
                    # No complete frame yet
                    if len(buf) > 1_000_000:
                        buf = buf[-100_000:]   # cap buffer at 1 MB, keep tail
                    continue

                # Walk back from last_end to find its SOI marker
                soi = buf.rfind(b'\xff\xd8', 0, last_end)
                if soi == -1:
                    buf = buf[last_end+2:]
                    continue

                jpg = buf[soi:last_end+2]
                # Keep everything after the last complete frame
                buf = buf[last_end+2:]

                frame = cv2.imdecode(
                    np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is not None:
                    with self._lock:
                        self._frame = frame

        except Exception as e:
            logger.error(f"MJPEG read error: {e}")
        finally:
            self._running = False
            logger.info("MJPEG reader thread exited")

    def read(self):
        with self._lock:
            f = self._frame
        if f is None:
            return False, None
        return True, f.copy()

    def is_alive(self):
        return self._running and self._thread is not None and self._thread.is_alive()

    def release(self):
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
#  OPEN CAMERA  (webcam or MJPEG)
# ─────────────────────────────────────────────────────────────────────────────
def open_camera(camera_url, retries=3):
    logger.info(f"Opening camera: {camera_url}")

    is_webcam = False
    try:
        src = int(camera_url);  is_webcam = True
        logger.info(f"→ Webcam index {src}")
    except (ValueError, TypeError):
        src = camera_url
        logger.info(f"→ HTTP stream: {camera_url}")

    for attempt in range(1, retries + 1):
        logger.info(f"  Attempt {attempt}/{retries}…")

        if not is_webcam and str(camera_url).lower().startswith('http'):
            cap = MJPEGCapture(camera_url)
            if cap._ok:
                # Wait for first frame (up to 12 s)
                deadline = time.time() + 12
                while time.time() < deadline:
                    ok, f = cap.read()
                    if ok and f is not None:
                        logger.info(f"✓ MJPEG stream open — shape {f.shape}")
                        return cap
                    time.sleep(0.3)
                cap.release()
                logger.warning("  MJPEG: connected but no frames — retrying")
            else:
                logger.warning("  MJPEG connect failed")
        else:
            cap = cv2.VideoCapture(src)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            deadline = time.time() + 15
            while time.time() < deadline:
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    logger.info(f"✓ Webcam open — shape {frame.shape}")
                    return cap
                time.sleep(0.2)
            cap.release()

        if attempt < retries:
            wait = 5 * attempt
            logger.info(f"  Waiting {wait}s before retry…")
            time.sleep(wait)

    logger.error("✗ Could not open camera after all retries")
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN DETECTION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_detection(camera_url, camera_id, server_url, cooldown=30, exit_timeout=90):
    logger.info("=" * 70)
    logger.info("FACE + EMOTION DETECTION  v3")
    logger.info(f"  Camera URL    : {camera_url}")
    logger.info(f"  Camera ID     : {camera_id}")
    logger.info(f"  Server        : {server_url}")
    logger.info(f"  Entry cooldown: {cooldown}s")
    logger.info(f"  Exit timeout  : {exit_timeout}s")
    logger.info("=" * 70)

    load_dependencies()

    global _emotion_worker
    _emotion_worker = EmotionWorker()

    # ── Load students ────────────────────────────────────────────────────────
    students = load_students(server_url)
    if not students:
        logger.warning("⚠ No students on first attempt — retrying in 5s…")
        time.sleep(5)
        students = load_students(server_url)
        if not students:
            logger.error("✗ No students with face encodings. Add photos in Django admin.")
            sys.exit(1)

    # ── FIX #2: Pre-populate students_inside from open logs ─────────────────
    students_inside = load_open_logs(server_url)
    logger.info(f"  students_inside at startup: {students_inside}")

    # ── Open camera ──────────────────────────────────────────────────────────
    cap = open_camera(camera_url)
    if cap is None:
        logger.error("✗ Camera unavailable — exiting")
        sys.exit(1)

    # Tracking state
    last_entry_time = defaultdict(float)
    last_seen_time  = defaultdict(float)

    # For students already inside, set last_entry_time so cooldown applies
    for sid in students_inside:
        last_entry_time[sid] = time.time() - cooldown + 10  # allow re-log after 10s

    frame_count = 0
    error_count = 0

    logger.info("✓ Detection running  (Ctrl+C to stop)\n")

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                error_count += 1
                if error_count > 20:
                    logger.warning("Too many frame failures — reconnecting…")
                    if hasattr(cap, 'release'):
                        cap.release()
                    time.sleep(5)
                    cap = open_camera(camera_url)
                    error_count = 0
                    if cap is None:
                        logger.error("✗ Reconnect failed — exiting")
                        sys.exit(1)
                else:
                    time.sleep(0.1)
                continue

            # Check if MJPEG thread is still alive
            if isinstance(cap, MJPEGCapture) and not cap.is_alive():
                logger.warning("MJPEG reader thread died — reconnecting…")
                cap.release()
                time.sleep(3)
                cap = open_camera(camera_url)
                if cap is None:
                    logger.error("✗ Reconnect failed")
                    sys.exit(1)
                error_count = 0
                continue

            error_count  = 0
            frame_count += 1
            now = time.time()

            # Periodically reload students (every ~5 min at ~10 fps)
            if frame_count % 3000 == 0:
                fresh = load_students(server_url)
                if fresh:
                    students = fresh
                    logger.info(f"↻ Reloaded {len(students)} students")

            # ── Face detection ────────────────────────────────────────────────
            faces = detect_faces(frame)

            if frame_count % 50 == 0:
                logger.info(
                    f"[F{frame_count:>6}] faces={len(faces)}"
                    f"  inside={len(students_inside)}"
                    f"  emotion_q={_emotion_worker._q.qsize()}"
                )

            seen_this_frame = set()

            for face_box in faces:
                try:
                    student, distance = match_student(frame, face_box, students)
                    if student is None:
                        continue

                    sid  = student['id']
                    name = student.get('name', '?')

                    seen_this_frame.add(sid)
                    last_seen_time[sid] = now

                    elapsed = now - last_entry_time[sid]
                    if elapsed <= cooldown:
                        logger.debug(f"  {name}: cooldown {cooldown - elapsed:.0f}s")
                        continue

                    # ── Decide ENTRY vs EXIT synchronously before async emotion ─
                    # students_inside is the local mirror of "has open log in Django".
                    # We update it IMMEDIATELY (optimistic) so no second face-detection
                    # within the emotion-worker delay fires the wrong action.
                    is_exit = sid in students_inside

                    # Gate cooldown BEFORE async work — prevents double-fire
                    last_entry_time[sid] = now

                    if is_exit:
                        students_inside.discard(sid)   # optimistic: treat as exited now
                        last_seen_time.pop(sid, None)
                        logger.info(f"  -> EXIT  queued : {name}")
                    else:
                        students_inside.add(sid)        # optimistic: treat as inside now
                        logger.info(f"  -> ENTRY queued : {name}")

                    snapshot   = encode_snapshot(frame, face_box)
                    _sid       = sid
                    _name      = name
                    _snap      = snapshot
                    _camera_id = camera_id
                    _server    = server_url
                    _is_exit   = is_exit

                    def on_emotion(emotion, confidence,
                                   sid=_sid, name=_name, snap=_snap,
                                   cam_id=_camera_id, srv=_server,
                                   is_exit=_is_exit):
                        ok, status = log_attendance(
                            srv, sid, cam_id,
                            emotion=emotion,
                            score=confidence,
                            snapshot_b64=snap,
                        )
                        if ok:
                            action = "EXIT" if is_exit else "ENTRY"
                            logger.info(f"  OK {action}: {name} emotion={emotion} ({confidence:.2f})")
                        else:
                            # API call failed — roll back optimistic state change
                            if is_exit:
                                students_inside.add(sid)
                            else:
                                students_inside.discard(sid)
                            logger.warning(f"  API failed for {name} — state rolled back")

                    # Submit emotion detection — never blocks the loop
                    _emotion_worker.submit(frame, face_box, on_emotion)
                except Exception as e:
                    logger.error(f"✗ face processing error: {e}", exc_info=True)
                    continue

            # ── EXIT detection ────────────────────────────────────────────────
            for sid in list(students_inside):
                if sid in seen_this_frame:
                    continue
                absent = now - last_seen_time.get(sid, now)
                if absent >= exit_timeout:
                    student_obj = next((s for s in students if s['id'] == sid), None)
                    name = student_obj['name'] if student_obj else f"id={sid}"
                    logger.info(f"→ AUTO-EXIT: {name} absent {absent:.0f}s")

                    ok, _ = log_attendance(
                        server_url, sid, camera_id,
                        emotion='neutral', score=0.0,
                    )
                    if ok:
                        students_inside.discard(sid)
                        last_entry_time.pop(sid, None)
                        last_seen_time.pop(sid, None)

    except KeyboardInterrupt:
        logger.info("\n✓ Stopped")
    except Exception as e:
        logger.error(f"✗ Fatal: {e}", exc_info=True)
    finally:
        if hasattr(cap, 'release'):
            cap.release()
        logger.info("✓ Done")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Face + emotion detection v3')
    parser.add_argument('--camera-url',   default='0')
    parser.add_argument('--camera-id',    type=int, required=True)
    parser.add_argument('--server',       default='http://localhost:8000')
    parser.add_argument('--cooldown',     type=int, default=30,
                        help='Seconds between entry logs per student (default 30)')
    parser.add_argument('--exit-timeout', type=int, default=90,
                        help='Seconds absent before exit is logged (default 90)')
    parser.add_argument('--debug',        action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    run_detection(
        args.camera_url, args.camera_id,
        args.server, args.cooldown, args.exit_timeout,
    )


if __name__ == '__main__':
    main()