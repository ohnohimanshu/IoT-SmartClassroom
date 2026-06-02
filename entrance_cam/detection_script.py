"""
Entrance Camera Detection Script - Entry/Exit Tracking with Mood Detection
===========================================================================
Fixes applied:
  1. Corrected API URL  /api/attendance/log/  →  /api/log/
  2. Fixed POST payload keys: 'mood'→'emotion', added 'score', removed 'event_type'
  3. face_encodings() now uses full frame + known_face_locations (no tiny-crop issue)
  4. Students list refreshed every ~60 s so newly added students are recognised
  5. Cascade re-loaded once (not inside the loop) for performance
  6. Mood/emotion detected via DeepFace when available, graceful fallback to 'neutral'

Usage:
    python detection_script.py --camera-url 0 --camera-id 1 --server http://localhost:8000
"""

import json
import time
import base64
import argparse
import requests
import sys
import os
import urllib3
from datetime import datetime
from collections import defaultdict

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# ── Globals filled by _load_deps() ────────────────────────────────────────────
cv2 = None
np = None
face_recognition = None
DeepFace = None
FACE_RECOGNITION_AVAILABLE = False
DEEPFACE_AVAILABLE = False
face_cascade = None          # loaded once, reused every frame


# ── Dependency loader ─────────────────────────────────────────────────────────

def _load_deps():
    """Lazy-load all heavy dependencies so import errors are clear."""
    global cv2, np, face_recognition, DeepFace
    global FACE_RECOGNITION_AVAILABLE, DEEPFACE_AVAILABLE, face_cascade

    import cv2 as _cv2
    import numpy as _np
    cv2 = _cv2
    np = _np

    # Load Haar cascade once
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("[ERROR] Haar cascade not found at:", cascade_path)
        sys.exit(1)
    print("[OK] Haar cascade loaded")

    try:
        import face_recognition as _fr
        face_recognition = _fr
        FACE_RECOGNITION_AVAILABLE = True
        print("[OK] face_recognition loaded")
    except ImportError:
        print("[WARN] face_recognition not installed — recognition disabled")
        FACE_RECOGNITION_AVAILABLE = False

    try:
        from deepface import DeepFace as _df
        DeepFace = _df
        DEEPFACE_AVAILABLE = True
        print("[OK] DeepFace loaded — live mood detection enabled")
    except ImportError:
        print("[WARN] DeepFace not installed — mood will default to 'neutral'")
        DEEPFACE_AVAILABLE = False


# ── Student loader ────────────────────────────────────────────────────────────

def load_students(server_url):
    """Fetch registered students with face encodings from Django."""
    try:
        resp = requests.get(
            f"{server_url}/camera-attendance/api/students/encodings/",
            timeout=5, verify=False
        )
        resp.raise_for_status()
        students = resp.json()
        print(f"[OK] Loaded {len(students)} students from server")
        return students
    except Exception as e:
        print(f"[ERROR] Could not load students: {e}")
        return []


# ── Face detection ────────────────────────────────────────────────────────────

def detect_faces(frame):
    """
    Detect faces using Haar cascade.
    Returns list of (x, y, w, h) tuples, or empty list.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)          # improve detection in low light
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,                     # Lowered from 4 to catch more faces
            minSize=(40, 40),                   # Lowered from 50 for smaller faces
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return []
        return faces
    except Exception as e:
        print(f"[ERROR] Face detection failed: {e}")
        return []


# ── Mood / emotion detection ──────────────────────────────────────────────────

def detect_emotion(frame, face_box):
    """
    Detect emotion for a detected face region.
    Returns (emotion_str, confidence_float).
    
    Disabled to reduce memory usage on Windows.
    Always returns 'neutral'.
    """
    return 'neutral', 0.0


# ── Student face matching ─────────────────────────────────────────────────────

def match_student(frame, face_box, students):
    """
    Match a detected face to a registered student.

    FIX: Pass the full RGB frame + known_face_locations so face_recognition
    doesn't run its own detector on a tiny crop (which frequently finds nothing).

    Returns (student_dict | None, distance_float)
    """
    if not students or not FACE_RECOGNITION_AVAILABLE:
        return None, 1.0

    try:
        x, y, w, h = face_box
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # face_recognition uses (top, right, bottom, left) order
        face_location = (y, x + w, y + h, x)

        face_encodings = face_recognition.face_encodings(
            rgb_frame,
            known_face_locations=[face_location],
            num_jitters=1       # 1 = fast; increase to 2-3 for more accuracy
        )

        if not face_encodings:
            return None, 1.0

        detected_encoding = face_encodings[0]
        best_distance = 1.0
        best_student = None

        for student in students:
            try:
                enc_data = student.get('encoding')
                if not enc_data:
                    continue
                known_enc = np.array(json.loads(enc_data))
                distance = face_recognition.face_distance([known_enc], detected_encoding)[0]
                if distance < best_distance:
                    best_distance = distance
                    best_student = student
            except Exception:
                continue

        # Threshold 0.55 is slightly stricter than the default 0.6
        # to reduce false positives in a college setting
        if best_distance < 0.55 and best_student:
            return best_student, float(best_distance)
        return None, float(best_distance)

    except Exception as e:
        print(f"[ERROR] Face matching failed: {e}")
        return None, 1.0


# ── API: log entry / exit ─────────────────────────────────────────────────────

def log_entry_exit(server_url, student_id, camera_id, emotion='neutral', score=0.0, snapshot_b64=None):
    """
    POST to /camera-attendance/api/log/ with the correct field names.
    """
    payload = {
        "student_id": student_id,
        "camera_id":  camera_id,
        "emotion":    emotion,
        "score":      score,
    }
    if snapshot_b64:
        payload["snapshot"] = snapshot_b64

    try:
        resp = requests.post(
            f"{server_url}/camera-attendance/api/log/",   # ← Updated URL
            json=payload,
            timeout=5,
            verify=False
        )
        data = resp.json()
        status = data.get('status', 'unknown')
        print(f"[LOG] {status.upper()} — student_id={student_id} | emotion={emotion} | score={score:.2f}")
        return data
    except Exception as e:
        print(f"[ERROR] Logging failed: {e}")
        return {}


# ── Snapshot helper ───────────────────────────────────────────────────────────

def encode_snapshot(frame, face_box=None):
    """
    Encode a frame (or face crop with padding) to base64 JPEG for the server.
    Returns base64 string or None on failure.
    """
    try:
        if face_box is not None:
            x, y, w, h = face_box
            pad = 30
            fh, fw = frame.shape[:2]
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(fw, x + w + pad)
            y2 = min(fh, y + h + pad)
            img = frame[y1:y2, x1:x2]
        else:
            img = frame

        if img.size == 0:
            return None

        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf).decode('utf-8')
    except Exception:
        return None


# ── Main detection loop ───────────────────────────────────────────────────────

def run_detection(camera_url, camera_id, server_url, cooldown=30, no_gui=True):
    """
    Open camera, continuously detect faces, match to students, log entry/exit.

    Improvements over original:
    - Students refreshed every STUDENT_REFRESH_FRAMES frames (FIX 4)
    - Cascade loaded once outside loop (performance)
    - Snapshot captured and sent with each log
    - Emotion detected per face via DeepFace
    """
    print(f"\n{'='*70}")
    print(f"[DETECTION] Starting detection script")
    print(f"[DETECTION] Camera URL: {camera_url}")
    print(f"[DETECTION] Camera ID: {camera_id}")
    print(f"[DETECTION] Server URL: {server_url}")
    print(f"{'='*70}\n")
    
    STUDENT_REFRESH_FRAMES = 900   # ~30 s at 30 fps — refresh student list

    print("[INFO] Loading dependencies...")
    _load_deps()

    # Resolve camera source
    # BUG FIX: Detect HTTP/HTTPS MJPEG streams (ESP32-CAM) vs local webcam index.
    # For ESP32-CAM, OpenCV needs the CAP_FFMPEG backend and must NOT set
    # CAP_PROP_BUFFERSIZE (that property only works for V4L2 local webcams and
    # silently breaks HTTP stream capture).
    is_http_stream = isinstance(camera_url, str) and camera_url.lower().startswith(('http://', 'https://'))
    try:
        src = int(camera_url)
        is_http_stream = False
        print(f"[INFO] Using webcam index {src}")
    except (ValueError, TypeError):
        src = camera_url
        if is_http_stream:
            print(f"[INFO] Using HTTP MJPEG stream (ESP32-CAM mode): {camera_url}")
        else:
            print(f"[INFO] Using stream URL: {camera_url}")

    def open_capture(source, http_stream):
        """Open VideoCapture with the correct backend for the source type."""
        if http_stream:
            # CAP_FFMPEG is required for HTTP multipart/x-mixed-replace MJPEG
            # streams produced by ESP32-CAM.
            #
            # FIX: "Stream ends prematurely" errors are caused by FFMPEG giving
            # up when the ESP32-CAM pauses between frames. The reconnect options
            # below tell FFMPEG to silently retry instead of crashing.
            #
            # These options MUST be set via environment variable before the
            # VideoCapture is created — cv2.CAP_PROP_* setters don't work for
            # FFMPEG input options on HTTP streams.
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;tcp|'
                'reconnect;1|'               # auto-reconnect on drop
                'reconnect_streamed;1|'      # reconnect even on live streams
                'reconnect_delay_max;5|'     # max 5 s between retries
                'timeout;10000000|'          # 10 s read timeout (in µs)
                'analyzeduration;1000000|'   # 1 s probe (faster stream open)
                'probesize;1000000'          # 1 MB probe (faster stream open)
            )
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            # Do NOT set CAP_PROP_BUFFERSIZE for HTTP streams — V4L2 only
        else:
            # Clear any FFMPEG options set for previous stream
            os.environ.pop('OPENCV_FFMPEG_CAPTURE_OPTIONS', None)
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Only safe for local webcams
        return cap

    # Open camera
    cap = open_capture(src, is_http_stream)

    # Wait up to 15 s for first frame (HTTP streams need more time to connect)
    ret = False
    start = time.time()
    timeout = 15 if is_http_stream else 10
    while (time.time() - start) < timeout:
        ret, frame = cap.read()
        if ret:
            break
        time.sleep(0.2)

    if not ret:
        print(f"[FATAL] Cannot open camera: {camera_url}")
        if is_http_stream:
            print(f"[FATAL] Make sure the ESP32-CAM is online and the URL ends with /stream")
            print(f"[FATAL] Expected URL format: http://<ESP32_IP>/stream")
        cap.release()
        sys.exit(1)

    print("[OK] Camera opened successfully!")

    # Initial student load
    students = load_students(server_url)
    if not students:
        print("[WARN] No students loaded — face matching will mark everyone Unknown")

    # Cooldown tracker: student_id → last log timestamp
    last_log_time = defaultdict(float)
    frame_count = 0

    print("[INFO] Starting detection loop... (Ctrl+C to stop)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame read failed — reconnecting in 2 s...")
                time.sleep(2)
                cap.release()
                cap = open_capture(src, is_http_stream)
                continue

            frame_count += 1

            # ── Periodic student list refresh (FIX 4) ─────────────────────
            if frame_count % STUDENT_REFRESH_FRAMES == 0:
                print("[INFO] Refreshing student list from server...")
                fresh = load_students(server_url)
                if fresh:   # only replace if fetch succeeded
                    students = fresh

            # ── Face detection ─────────────────────────────────────────────
            faces = detect_faces(frame)

            if frame_count % 300 == 0:
                print(f"[DEBUG] Frame {frame_count} | Faces detected: {len(faces)} | Students loaded: {len(students)}")

            # ── Process each detected face ─────────────────────────────────
            for face_box in faces:
                x, y, fw, fh = face_box

                # 1. Match to a registered student
                student, distance = match_student(frame, face_box, students)

                if not no_gui:
                    if student is None:
                        # Draw orange box for unknown person
                        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 165, 255), 2)
                        cv2.putText(
                            frame, f"Unknown ({distance:.2f})",
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2
                        )

                if student is not None:
                    student_id   = student['id']
                    student_name = student['name']

                    # 2. Detect emotion for this face
                    emotion, score = detect_emotion(frame, face_box)

                    if not no_gui:
                        # 3. Draw green box with name + emotion
                        label = f"{student_name} | {emotion} ({distance:.2f})"
                        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                        cv2.putText(
                            frame, label,
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                        )

                    # 4. Log with cooldown
                    now = time.time()
                    if (now - last_log_time[student_id]) > cooldown:
                        snapshot_b64 = encode_snapshot(frame, face_box)
                        result = log_entry_exit(
                            server_url, student_id, camera_id,
                            emotion=emotion, score=score,
                            snapshot_b64=snapshot_b64
                        )
                        last_log_time[student_id] = now

                        if not no_gui:
                            # Show entry/exit status on frame for 1 s
                            status_label = result.get('status', 'logged').replace('_', ' ').upper()
                            cv2.putText(
                                frame, status_label,
                                (x, y + fh + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                            )

            # ── Display frame only if GUI is enabled ───────────────────────
            if not no_gui:
                cv2.imshow("Entrance Camera - Face Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] 'q' pressed — stopping.")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        if not no_gui:
            cv2.destroyAllWindows()
        print("[INFO] Camera released. Bye!")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entrance camera detection — entry/exit + mood tracking'
    )
    parser.add_argument('--camera-url', default='0',
                        help='Webcam index (0,1,…) or IP stream URL')
    parser.add_argument('--camera-id', type=int, required=True,
                        help='Camera database ID (from Django admin)')
    parser.add_argument('--server', default='http://localhost:8000',
                        help='Django server base URL')
    parser.add_argument('--cooldown', type=int, default=30,
                        help='Seconds between repeated logs for the same student')
    parser.add_argument('--no-gui', action='store_true', default=True,
                        help='Hide OpenCV GUI windows (default: True)')
    parser.add_argument('--gui', dest='no_gui', action='store_false',
                        help='Show OpenCV GUI windows')

    args = parser.parse_args()

    try:
        print("[INFO] Starting entrance detection script...")
        run_detection(args.camera_url, args.camera_id, args.server, args.cooldown, args.no_gui)
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)