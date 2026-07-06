
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
from collections import defaultdict, deque

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
cv2                        = None
np                         = None
face_recognition           = None
DeepFace                   = None
mp                         = None
mp_face_detection          = None
FACE_RECOGNITION_AVAILABLE = False
DEEPFACE_AVAILABLE         = False

VALID_EMOTIONS = {'happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust'}
EMOTION_MAP = {
    'happy':   'happy',   'sad':     'sad',    'angry':   'angry',
    'neutral': 'neutral', 'surprise':'surprise','fear':   'fear',
    'disgust': 'disgust', 'contempt':'disgust',
}


# ─────────────────────────────────────────────────────────────────────────────
#  DEPENDENCY LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_dependencies():
    global cv2, np, face_recognition, DeepFace, mp, mp_face_detection
    global FACE_RECOGNITION_AVAILABLE, DEEPFACE_AVAILABLE

    logger.info("Loading dependencies…")

    try:
        import cv2 as _cv2
        import numpy as _np
        cv2 = _cv2
        np  = _np
        logger.info("✓ OpenCV loaded")
    except ImportError as e:
        logger.error(f"✗ OpenCV not installed: {e}")
        sys.exit(1)

    try:
        import mediapipe as _mp
        mp = _mp
        # Verify mediapipe has solutions module
        if not hasattr(mp, 'solutions'):
            raise AttributeError("MediaPipe 'solutions' module not available - may be corrupted installation")
        mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.7
        )
        logger.info("✓ MediaPipe Face Detection loaded")
    except (ImportError, AttributeError, Exception) as e:
        logger.warning(f"⚠ MediaPipe not available: {e}")
        logger.warning("  Falling back to Haar cascades (lower accuracy)")
        try:
            global face_cascade, _eye_cascade
            frontal_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(frontal_path)
            if face_cascade.empty():
                logger.error("✗ Haar frontal cascade not found")
                sys.exit(1)

            eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            _eye_cascade = cv2.CascadeClassifier(eye_path)
            if _eye_cascade.empty():
                logger.warning("⚠ Eye cascade not found — eye validation disabled")
                _eye_cascade = None
            else:
                logger.info("✓ Haar cascades loaded (frontal + eye)")
        except Exception as e2:
            logger.error(f"✗ Could not load fallback Haar cascades: {e2}")
            sys.exit(1)

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
# ─────────────────────────────────────────────────────────────────────────────
class EmotionWorker:
    def __init__(self):
        self._q      = queue.Queue(maxsize=4)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame, face_box, callback):
        try:
            self._q.put_nowait((frame.copy(), face_box, callback))
        except queue.Full:
            # Queue backed up; call back with neutral so attendance is still logged
            try:
                callback('neutral', 0.0)
            except Exception:
                pass

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
            crop = frame[max(0, y-pad):min(fh, y+h+pad),
                         max(0, x-pad):min(fw, x+w+pad)]
            if crop.size == 0:
                return 'neutral', 0.0

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
            return emotion, conf
        except Exception as e:
            logger.debug(f"DeepFace error: {e}")
            return 'neutral', 0.0


_emotion_worker = None


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
#  OPEN LOG LOADER — pre-populate students_inside on startup / periodic refresh
# ─────────────────────────────────────────────────────────────────────────────
def load_open_logs(server_url):
    """
    Returns a set of student IDs that have an open (entry, no exit) log today.
    Called on startup AND periodically so local state stays in sync with Django.
    """
    try:
        r = requests.get(
            f"{server_url}/camera-attendance/api/live-detections/",
            timeout=10, verify=False,
        )
        r.raise_for_status()
        data = r.json()
        inside = {
            log['student_id']
            for log in data.get('logs', [])
            if not log.get('exit_time') and log.get('student_id') is not None
        }
        logger.info(f"✓ Synced {len(inside)} students currently inside")
        return inside
    except Exception as e:
        logger.warning(f"⚠ Could not load open logs: {e}")
        return None   # None means "could not fetch"; caller should keep existing set


# ─────────────────────────────────────────────────────────────────────────────
#  FACE DETECTION  — Use MediaPipe for accuracy, fall back to Haar
# ─────────────────────────────────────────────────────────────────────────────
def detect_faces(frame):
    """
    Returns list of (x, y, w, h) face boxes.
    First tries MediaPipe, if not available falls back to Haar.
    """
    if frame is None or frame.size == 0:
        return []
    try:
        if mp_face_detection is not None:
            # Use MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face_detection.process(rgb_frame)
            faces = []
            if results.detections:
                h, w, _ = frame.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    # Clamp to valid coordinates
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    width = min(width, w - x1)
                    height = min(height, h - y1)
                    faces.append((x1, y1, width, height))
            return faces
        else:
            # Fall back to Haar cascades
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            raw_faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=6,
                minSize=(60, 60),
                maxSize=(800, 800),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if len(raw_faces) == 0:
                return []

            validated = []
            for (x, y, w, h) in raw_faces:
                # 1. Aspect ratio
                ar = w / h if h > 0 else 1.0
                if not (0.75 <= ar <= 1.3):
                    logger.debug(f"  ✗ Bad aspect ratio {ar:.2f}")
                    continue

                # 3. Skin-tone check: mean pixel value should be mid-range
                face_gray = gray[y:y+h, x:x+w]
                mean_val = float(np.mean(face_gray))
                if mean_val < 30 or mean_val > 230:
                    logger.debug(f"  ✗ Skin-tone check failed (mean={mean_val:.0f})")
                    continue

                validated.append((x, y, w, h))
                logger.debug(f"  ✓ Face validated at ({x},{y},{w},{h})")

            return validated
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
        x, y, w, h = face_box
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        loc  = (y, x+w, y+h, x)   # (top, right, bottom, left)
        encs = face_recognition.face_encodings(
            rgb, known_face_locations=[loc], num_jitters=2
        )
        if not encs:
            return None, 1.0

        enc            = encs[0]
        best_d         = 1.0
        second_best_d  = 1.0
        best_s         = None

        for student in students:
            raw = student.get('encoding')
            if not raw:
                continue
            try:
                known = np.array(json.loads(raw))
                d = face_recognition.face_distance([known], enc)[0]
                if d < best_d:
                    second_best_d = best_d
                    best_d, best_s = d, student
                elif d < second_best_d:
                    second_best_d = d
            except Exception:
                continue

        THRESHOLD  = 0.42   # slightly relaxed from 0.40 to reduce false rejects
        MIN_MARGIN = 0.08   # still require clear winner

        if best_d < THRESHOLD and best_s:
            margin = second_best_d - best_d
            if margin >= MIN_MARGIN or second_best_d > 0.9:
                logger.info(
                    f"  ✅ MATCH: {best_s['name']} "
                    f"(dist={best_d:.3f}, margin={margin:.3f})"
                )
                return best_s, float(best_d)
            else:
                logger.debug(
                    f"  ⚠ Ambiguous: {best_s['name']} (dist={best_d:.3f}) "
                    f"vs second (dist={second_best_d:.3f}) — skipped"
                )
                return None, float(best_d)

        logger.debug(f"  ✗ No match (best={best_d:.3f})")
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
            pad = 30
            fh, fw = frame.shape[:2]
            img = frame[max(0, y-pad):min(fh, y+h+pad),
                        max(0, x-pad):min(fw, x+w+pad)]
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
def blink_esp32(camera_url, times=2):
    """
    Tell the ESP32-CAM to blink its status LED N times.
    camera_url is the base stream URL, e.g. 'http://192.168.1.10/stream'.
    We derive the base URL by stripping the path.
    Non-blocking: runs in a daemon thread so it never delays attendance logging.
    Safe to ignore failures — blink is cosmetic only.
    """
    try:
        # Derive base URL: strip everything from the last '/' path onwards
        from urllib.parse import urlparse, urlunparse
        parsed  = urlparse(camera_url)
        base    = urlunparse((parsed.scheme, parsed.netloc, '', '', '', ''))
        blink_url = f"{base}/blink?times={times}"

        def _do_blink():
            try:
                requests.post(blink_url, timeout=3, verify=False)
                logger.debug(f"  💡 Blink sent to ESP32 ({times}x)")
            except Exception as e:
                logger.debug(f"  💡 Blink request failed (non-critical): {e}")

        t = threading.Thread(target=_do_blink, daemon=True)
        t.start()
    except Exception as e:
        logger.debug(f"  💡 blink_esp32 error (non-critical): {e}")


def log_attendance(server_url, student_id, camera_id,
                   emotion='neutral', score=0.0, snapshot_b64=None,
                   camera_url=None):
    """
    Log attendance to Django. If camera_url is provided and the log succeeds,
    triggers a 2-blink on the ESP32-CAM LED so the student gets visual feedback.
    """
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
            logger.info(
                f"  ✓ API [{status.upper()}] "
                f"student={student_id} emotion={emotion} score={score:.2f}"
            )
            # ── Blink LED on ESP32 so student gets visual confirmation ────────
            if camera_url:
                blink_esp32(camera_url, times=2)
            return True, status
        else:
            logger.warning(f"  ✗ API {r.status_code}: {r.text[:200]}")
            return False, 'error'
    except Exception as e:
        logger.error(f"  ✗ log_attendance: {e}")
        return False, 'error'


# ─────────────────────────────────────────────────────────────────────────────
#  MJPEG STREAM  — always-latest-frame reader
# ─────────────────────────────────────────────────────────────────────────────
class MJPEGCapture:
    CHUNK = 16384

    def __init__(self, url, server_url=None):
        self.url        = url
        self.server_url = server_url
        self._frame     = None
        self._lock      = threading.Lock()
        self._running   = False
        self._thread    = None
        self._ok        = False
        self._connect()

    def _connect(self):
        try:
            # Increased timeout from 10s to 30s for slow/distant networks
            # Also set read timeout to prevent stalls
            resp = requests.get(
                self.url, stream=True, verify=False,
                timeout=(30, 60),  # (connection_timeout, read_timeout)
                headers={
                    'Connection': 'keep-alive',
                    'Cache-Control': 'no-cache',
                    'Accept': 'multipart/x-mixed-replace',
                },
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

                last_end  = -1
                search_from = 0
                while True:
                    end = buf.find(b'\xff\xd9', search_from)
                    if end == -1:
                        break
                    last_end    = end
                    search_from = end + 2

                if last_end == -1:
                    if len(buf) > 1_000_000:
                        buf = buf[-100_000:]
                    continue

                soi = buf.rfind(b'\xff\xd8', 0, last_end)
                if soi == -1:
                    buf = buf[last_end+2:]
                    continue

                jpg = buf[soi:last_end+2]
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

    def read(self):
        with self._lock:
            f = self._frame
        if f is None:
            return False, None
        return True, f.copy()

    def is_alive(self):
        return (
            self._running
            and self._thread is not None
            and self._thread.is_alive()
        )

    def release(self):
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
#  MJPEG REBROADCASTER
#  Serves the latest frame from MJPEGCapture over a plain HTTP server on
#  localhost so the Django proxy can read it without opening a second
#  connection to the ESP32 (which only supports 1 concurrent stream).
#
#  The browser proxy points to:
#    http://127.0.0.1:<rebroadcast_port>/stream
#  instead of the ESP32 directly.
# ─────────────────────────────────────────────────────────────────────────────
import http.server
import socketserver

class MJPEGRebroadcaster:
    """
    Serves the latest frame from MJPEGCapture as an MJPEG stream on localhost.
    Uses ThreadingTCPServer so multiple clients (Django proxy, browser) can
    connect simultaneously without blocking each other.
    The latest encoded JPEG is stored in a class-level shared variable so all
    client threads read the same frame without re-encoding per-client.
    """
    BOUNDARY = b'gc0p4Jq0M2Yt08jU534c0p'

    def __init__(self, capture, port=8765):
        self._cap      = capture
        self._port     = port
        self._server   = None
        self._srv_thread = None
        self._enc_thread = None
        self._running  = False
        # Shared latest JPEG — written by encoder thread, read by all client threads
        self._latest_jpg      = None
        self._latest_jpg_lock = threading.Lock()
        self._frame_event     = threading.Event()  # signals new frame ready

    def _encode_loop(self):
        """Background thread: re-encodes latest camera frame at ~15 fps."""
        interval = 1.0 / 15  # 15 fps cap — enough for live view, not overloading
        while self._running:
            t0 = time.time()
            ok, frame = self._cap.read()
            if ok and frame is not None:
                ret, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    with self._latest_jpg_lock:
                        self._latest_jpg = jpg.tobytes()
                    self._frame_event.set()
                    self._frame_event.clear()
            elapsed = time.time() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self):
        boundary      = self.BOUNDARY
        get_jpg       = lambda: (self._latest_jpg_lock.__enter__() or True) and None  # placeholder
        rebroadcaster = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                if self.path not in ('/stream', '/'):
                    self.send_response(404); self.end_headers(); return

                self.send_response(200)
                self.send_header('Content-Type',
                    f'multipart/x-mixed-replace;boundary={boundary.decode()}')
                self.send_header('Cache-Control', 'no-store, no-cache')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                try:
                    while rebroadcaster._running:
                        # Wait up to 1s for a new frame
                        rebroadcaster._frame_event.wait(timeout=1.0)

                        with rebroadcaster._latest_jpg_lock:
                            jpg_bytes = rebroadcaster._latest_jpg

                        if jpg_bytes is None:
                            continue

                        part = (
                            b'--' + boundary + b'\r\n'
                            b'Content-Type: image/jpeg\r\n'
                            b'Content-Length: ' + str(len(jpg_bytes)).encode() + b'\r\n\r\n'
                            + jpg_bytes + b'\r\n'
                        )
                        try:
                            self.wfile.write(part)
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            break
                except Exception:
                    pass

        class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            allow_reuse_address = True
            daemon_threads      = True  # client threads die when main exits

        try:
            self._running = True
            self._server  = ThreadedServer(('127.0.0.1', self._port), Handler)

            # Start encoder thread
            self._enc_thread = threading.Thread(target=self._encode_loop, daemon=True)
            self._enc_thread.start()

            # Start server thread
            self._srv_thread = threading.Thread(
                target=self._server.serve_forever, daemon=True
            )
            self._srv_thread.start()
            logger.info(f"✓ MJPEG rebroadcast server on http://127.0.0.1:{self._port}/stream")
        except OSError as e:
            self._running = False
            logger.warning(f"⚠ Could not start rebroadcast server on port {self._port}: {e}")
            logger.warning("  Try a different port with --rebroadcast-port")

    def stop(self):
        self._running = False
        if self._server:
            self._server.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
#  OPEN CAMERA
# ─────────────────────────────────────────────────────────────────────────────
def open_camera(camera_url, retries=5):
    """Open camera with extended timeout for slow networks."""
    logger.info(f"Opening camera: {camera_url}")

    is_webcam = False
    try:
        src = int(camera_url)
        is_webcam = True
        logger.info(f"→ Webcam index {src}")
    except (ValueError, TypeError):
        src = camera_url
        logger.info(f"→ HTTP stream: {camera_url}")

    for attempt in range(1, retries + 1):
        logger.info(f"  Attempt {attempt}/{retries}…")

        if not is_webcam and str(camera_url).lower().startswith('http'):
            cap = MJPEGCapture(camera_url)
            if cap._ok:
                # Extended deadline for slow/distant ESP32-CAM
                deadline = time.time() + 20  # increased from 12s
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
#  MULTI-FRAME CONFIRMATION TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class ConfirmTracker:
    """
    A student is only 'confirmed' once they appear in CONFIRM_FRAMES
    consecutive frames.  This eliminates single-frame false positives
    from reflections, photos on walls, or partial obstructions.

    Usage:
        tracker = ConfirmTracker(confirm_frames=3)

        # Per frame, call update() with the set of matched student IDs.
        # Returns set of IDs that just reached confirmation threshold.
        confirmed_now = tracker.update(matched_ids_this_frame)
    """
    def __init__(self, confirm_frames=3):
        self._n      = confirm_frames
        # consecutive hit count per student
        self._hits   = defaultdict(int)
        # students already confirmed and still being seen (suppress re-confirm)
        self._active = set()

    def update(self, seen_ids: set) -> set:
        """
        seen_ids: set of student IDs matched this frame.
        Returns: set of IDs that just crossed the confirmation threshold.
        """
        newly_confirmed = set()

        # Increment hits for seen students; decrement/reset for absent ones
        for sid in list(self._hits):
            if sid not in seen_ids:
                self._hits[sid] = max(0, self._hits[sid] - 1)
                if self._hits[sid] == 0:
                    del self._hits[sid]
                    # Remove from active so they can be confirmed again
                    # (e.g. if they come back after exit)
                    self._active.discard(sid)

        for sid in seen_ids:
            if sid in self._active:
                continue   # already passed confirmation; don't fire again
            self._hits[sid] += 1
            if self._hits[sid] >= self._n:
                newly_confirmed.add(sid)
                self._active.add(sid)
                self._hits[sid] = 0   # reset counter for next pass

        return newly_confirmed

    def mark_exited(self, sid):
        """Call this when a student has exited so they can be confirmed again."""
        self._active.discard(sid)
        self._hits.pop(sid, None)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN DETECTION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_detection(camera_url, camera_id, server_url,
                  cooldown=60, exit_timeout=90, confirm_frames=3,
                  rebroadcast_port=8765):
    logger.info("=" * 70)
    logger.info("FACE + EMOTION DETECTION  v4")
    logger.info(f"  Camera URL    : {camera_url}")
    logger.info(f"  Camera ID     : {camera_id}")
    logger.info(f"  Server        : {server_url}")
    logger.info(f"  Entry cooldown: {cooldown}s")
    logger.info(f"  Exit timeout  : {exit_timeout}s")
    logger.info(f"  Confirm frames: {confirm_frames}")
    logger.info("=" * 70)

    load_dependencies()

    global _emotion_worker
    _emotion_worker = EmotionWorker()

    # ── Handle HTTP → HTTPS proxy for HTTP cameras ────────────────────────────
    # The detection script runs server-side so it reaches the ESP32 directly
    # over HTTP regardless of whether Django uses HTTPS.
    # The proxy endpoint (/cameras/<pk>/proxy-stream/) is only needed by the
    # browser — the script always connects to the camera URL directly.
    actual_camera_url = camera_url
    if camera_url.lower().startswith('http://') and server_url.lower().startswith('https://'):
        logger.info("  ℹ Camera is HTTP, server is HTTPS")
        logger.info("  → Script connects to ESP32 directly over HTTP (proxy not needed server-side)")

    # ── Load students ────────────────────────────────────────────────────────
    students = load_students(server_url)
    if not students:
        logger.warning("⚠ No students on first attempt — retrying in 5s…")
        time.sleep(5)
        students = load_students(server_url)
        if not students:
            logger.error("=" * 70)
            logger.error("✗ CRITICAL: No students with face encodings")
            logger.error("=" * 70)
            logger.error("Checklist:")
            logger.error("  1. Django running? python manage.py runserver")
            logger.error("  2. Go to: http://localhost:8000/admin/entrance_cam/student/")
            logger.error("  3. Add students with photos")
            logger.error("  4. Face encodings should auto-generate")
            logger.error("  5. Check 'Face Encoding' field has data")
            logger.error("=" * 70)
            sys.exit(1)

    # ── Pre-populate students_inside from open logs ──────────────────────────
    students_inside = load_open_logs(server_url) or set()
    logger.info(f"  Students already inside at startup: {students_inside}")

    # ── Open camera ──────────────────────────────────────────────────────────
    logger.info(f"Opening camera: {camera_url}")
    cap = open_camera(actual_camera_url)
    if cap is None:
        logger.error("=" * 70)
        logger.error("✗ CRITICAL: Camera unavailable")
        logger.error("=" * 70)
        logger.error(f"Camera URL: {camera_url}")
        logger.error("Checklist:")
        logger.error("  1. Is ESP32-CAM powered on?")
        logger.error("  2. Is IP address correct? (change in --camera-url)")
        logger.error("  3. Test in browser: http://192.168.1.10/stream")
        logger.error("  4. Are you on same network as camera?")
        logger.error("  5. Can you ping camera? ping 192.168.1.10")
        logger.error("=" * 70)
        sys.exit(1)

    # ── Per-student tracking state ───────────────────────────────────────────
    last_seen_time  = defaultdict(float)   # last time face was seen
    last_action_time = defaultdict(float)  # last time ENTRY or EXIT was fired
    confirm_tracker = ConfirmTracker(confirm_frames=confirm_frames)

    # For students already inside: set last_action_time so cooldown applies
    for sid in students_inside:
        last_action_time[sid] = time.time() - cooldown + 10

    frame_count       = 0
    error_count       = 0
    last_sync_time    = time.time()
    SYNC_INTERVAL     = 600   # re-sync students_inside from Django every 10 min

    # Snapshot captured at confirmation time (used by emotion callback)
    confirm_snapshots = {}
    confirm_faceboxes = {}
    confirm_frames_map = {}

    # ── Start local MJPEG rebroadcast server ────────────────────────────────
    # Serves the stream on localhost so the Django proxy doesn't need a
    # second connection to the ESP32.
    _rebroadcaster = None
    if isinstance(cap, MJPEGCapture):
        _rebroadcaster = MJPEGRebroadcaster(cap, port=rebroadcast_port)
        _rebroadcaster.start()

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

            # ── Periodic student reload (every ~5 min at ~10 fps) ─────────────
            if frame_count % 3000 == 0:
                fresh = load_students(server_url)
                if fresh:
                    students = fresh
                    logger.info(f"↻ Reloaded {len(students)} students")

            # ── Periodic inside-state sync from Django ────────────────────────
            if now - last_sync_time > SYNC_INTERVAL:
                synced = load_open_logs(server_url)
                if synced is not None:
                    students_inside = synced
                    logger.info(f"↻ Re-synced students_inside: {students_inside}")
                last_sync_time = now

            # ── Face detection ────────────────────────────────────────────────
            faces = detect_faces(frame)

            if frame_count % 100 == 0:
                logger.info(
                    f"[F{frame_count:>6}] faces={len(faces)}"
                    f"  inside={len(students_inside)}"
                    f"  eq={_emotion_worker._q.qsize()}"
                )

            # Match faces → student IDs
            matched_this_frame = {}   # sid -> (student_obj, face_box, distance)
            for face_box in faces:
                try:
                    student, distance = match_student(frame, face_box, students)
                    if student is None:
                        continue
                    sid = student['id']
                    # If multiple faces match same student (shouldn't happen),
                    # keep the closest one.
                    if sid not in matched_this_frame or distance < matched_this_frame[sid][2]:
                        matched_this_frame[sid] = (student, face_box, distance)
                        last_seen_time[sid]     = now
                except Exception as e:
                    logger.error(f"✗ face matching error: {e}", exc_info=True)

            # ── Multi-frame confirmation ──────────────────────────────────────
            confirmed_ids = confirm_tracker.update(set(matched_this_frame))

            for sid in confirmed_ids:
                student, face_box, _ = matched_this_frame[sid]
                name = student.get('name', '?')

                # ── FIX #1: determine is_exit HERE, fresh per student ─────────
                # This is the critical fix: is_exit MUST be evaluated from
                # students_inside at this exact moment, not from a shared loop
                # variable that can bleed across iterations.
                is_exit = (sid in students_inside)

                # Cooldown check: prevent re-logging same action within cooldown
                elapsed = now - last_action_time[sid]
                if elapsed < cooldown:
                    logger.debug(
                        f"  {name}: cooldown {cooldown - elapsed:.0f}s remaining"
                        f" (action={'EXIT' if is_exit else 'ENTRY'})"
                    )
                    continue

                # ── Optimistic state update BEFORE async emotion ───────────────
                # This ensures if the same face is seen again before the emotion
                # worker finishes, the correct action is still selected.
                if is_exit:
                    students_inside.discard(sid)
                    confirm_tracker.mark_exited(sid)
                    logger.info(f"  → EXIT  queued: {name} (id={sid})")
                else:
                    students_inside.add(sid)
                    logger.info(f"  → ENTRY queued: {name} (id={sid})")

                last_action_time[sid] = now

                # Capture snapshot now (before frame changes)
                snapshot = encode_snapshot(frame, face_box)

                # ── Closure: capture all variables explicitly ─────────────────
                # Using default-argument binding to freeze current values.
                # This prevents the classic Python closure-in-loop bug where
                # all callbacks share the last iteration's variables.
                def on_emotion(emotion, confidence,
                               _sid=sid,
                               _name=name,
                               _snap=snapshot,
                               _cam_id=camera_id,
                               _srv=server_url,
                               _is_exit=is_exit,
                               _cam_url=camera_url):

                    ok, status = log_attendance(
                        _srv, _sid, _cam_id,
                        emotion=emotion,
                        score=confidence,
                        snapshot_b64=_snap,
                        camera_url=_cam_url,   # triggers LED blink on success
                    )
                    if ok:
                        action = "EXIT" if _is_exit else "ENTRY"
                        logger.info(
                            f"  ✓ {action}: {_name} "
                            f"emotion={emotion} conf={confidence:.2f}"
                        )
                    else:
                        # Roll back optimistic state on API failure
                        logger.warning(
                            f"  ✗ API failed for {_name} — rolling back state"
                        )
                        if _is_exit:
                            # We marked them as outside; put back
                            students_inside.add(_sid)
                        else:
                            # We marked them as inside; remove
                            students_inside.discard(_sid)

                _emotion_worker.submit(frame, face_box, on_emotion)

            # ── Auto-EXIT: student absent for exit_timeout seconds ─────────────
            for sid in list(students_inside):
                absent = now - last_seen_time.get(sid, now)
                if absent >= exit_timeout:
                    student_obj = next(
                        (s for s in students if s['id'] == sid), None
                    )
                    name = student_obj['name'] if student_obj else f"id={sid}"
                    logger.info(f"→ AUTO-EXIT: {name} absent {absent:.0f}s")

                    ok, _ = log_attendance(
                        server_url, sid, camera_id,
                        emotion='neutral', score=0.0,
                        camera_url=camera_url,
                    )
                    if ok:
                        students_inside.discard(sid)
                        last_action_time.pop(sid, None)
                        last_seen_time.pop(sid, None)
                        confirm_tracker.mark_exited(sid)

    except KeyboardInterrupt:
        logger.info("\n✓ Stopped by user")
    except Exception as e:
        logger.error(f"✗ Fatal: {e}", exc_info=True)
    finally:
        if hasattr(cap, 'release'):
            cap.release()
        if '_rebroadcaster' in dir() and _rebroadcaster:
            _rebroadcaster.stop()
        logger.info("✓ Done")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='School Face Attendance v4')
    parser.add_argument('--camera-url',     default='0')
    parser.add_argument('--camera-id',      type=int, required=True)
    parser.add_argument('--server',         default='http://localhost:8000')
    parser.add_argument('--cooldown',       type=int, default=60,
                        help='Seconds between actions per student (default 60)')
    parser.add_argument('--exit-timeout',   type=int, default=90,
                        help='Seconds absent before auto-exit (default 90)')
    parser.add_argument('--confirm-frames', type=int, default=3,
                        help='Frames face must appear before action fires (default 3)')
    parser.add_argument('--rebroadcast-port', type=int, default=8765,
                        help='Local port for MJPEG rebroadcast server (default 8765)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    run_detection(
        args.camera_url,
        args.camera_id,
        args.server,
        args.cooldown,
        args.exit_timeout,
        args.confirm_frames,
        args.rebroadcast_port,
    )


if __name__ == '__main__':
    main()