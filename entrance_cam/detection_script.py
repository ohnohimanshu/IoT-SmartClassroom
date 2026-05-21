"""
Entrance Camera Detection Script
---------------------------------
Run this separately on the machine connected to the camera:
    python detection_script.py --camera-id 1 --server http://localhost:8000

It will:
1. Open the camera stream (IP URL or webcam index)
2. Detect faces using OpenCV + DeepFace
3. Match against known student face encodings
4. POST entry/exit events to Django via /api/log/
"""

import cv2
import json
import time
import base64
import argparse
import requests
import numpy as np
import sys
import os
import urllib3
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── GPU / TensorFlow configuration ────────────────────────────────────────────
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

# Report GPU availability at startup
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"[GPU] OpenCV CUDA enabled — {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
    else:
        print("[GPU] No CUDA GPU for OpenCV, using CPU")
except Exception:
    pass

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    print("[WARN] DeepFace not installed. Emotion detection disabled.")
    DEEPFACE_AVAILABLE = False

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    print("[WARN] face_recognition not installed. Using OpenCV fallback.")
    FACE_RECOGNITION_AVAILABLE = False


def load_known_faces(server_url):
    """Download student face encodings from Django."""
    try:
        resp = requests.get(f"{server_url}/api/students/encodings/", timeout=5, verify=False)
        return resp.json()  # [{id, name, encoding}]
    except Exception as e:
        print(f"[ERROR] Could not load student encodings: {e}")
        return []


def detect_emotion(frame):
    """Run DeepFace on a BGR frame. Returns (emotion_label, confidence)."""
    if not DEEPFACE_AVAILABLE:
        return "unknown", 0.0

    try:
        # Convert BGR to RGB for DeepFace
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False, silent=True)

        if isinstance(result, list):
            result = result[0]

        emotion = result.get('dominant_emotion', 'unknown')
        score = result.get('emotion', {}).get(emotion, 0.0)

        return emotion, round(float(score), 2)

    except Exception as e:
        print(f"[WARN] Emotion detection failed: {e}")
        return "unknown", 0.0


def frame_to_base64(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf).decode()


def log_to_server(server_url, student_id, camera_id, emotion, score, snapshot_b64):
    try:
        resp = requests.post(
            f"{server_url}/api/log/",
            json={
                "student_id": student_id,
                "camera_id": camera_id,
                "emotion": emotion,
                "score": score,
                "snapshot": snapshot_b64,
            },
            timeout=5,
            verify=False
        )
        data = resp.json()
        print(f"[LOG] {data.get('status')} — student {student_id}, emotion: {emotion} ({score}%)")
        return data
    except Exception as e:
        print(f"[ERROR] Server log failed: {e}")
        return {}


def run_detection(camera_url, camera_id, server_url, cooldown_seconds=30):
    """
    Main detection loop.

    Entry/exit logic:
    - ENTRY: student recognised for the first time (or after cooldown from last exit).
    - EXIT:  student absent from frame for EXIT_ABSENCE_FRAMES consecutive frames.
             We keep the last known face crop so we can send a real exit-emotion snapshot.
    """
    print("[INFO] Initializing face detection...")

    # ── How many consecutive absent frames before we treat it as an exit ──────
    # At ~30 fps, 90 frames ≈ 3 seconds.  Tweak to taste.
    EXIT_ABSENCE_FRAMES = 90

    # ── DNN face detector ─────────────────────────────────────────────────────
    dnn_model = os.path.join(cv2.data.haarcascades,
                             '..', 'dnn', 'face_detector',
                             'opencv_face_detector_uint8.pb')
    dnn_config = os.path.join(cv2.data.haarcascades,
                              '..', 'dnn', 'face_detector',
                              'opencv_face_detector.pbtxt')

    face_net = None
    if os.path.exists(dnn_model) and os.path.exists(dnn_config):
        face_net = cv2.dnn.readNetFromTensorflow(dnn_model, dnn_config)
        print("[INFO] Using DNN face detector")
    else:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("[INFO] Using Haar cascade face detector (DNN model not found)")

    # ── Open camera ───────────────────────────────────────────────────────────
    print(f"[INFO] Attempting to open camera: {camera_url}")
    try:
        src = int(camera_url)
        print(f"[INFO] Camera is a webcam (index: {src})")
    except ValueError:
        src = camera_url
        print(f"[INFO] Camera is an IP stream: {camera_url}")

    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    start_time = time.time()
    frame_read = False
    while (time.time() - start_time) < 10:
        ret, frame = cap.read()
        if ret:
            frame_read = True
            break
        time.sleep(0.1)

    if not frame_read:
        print(f"[FATAL] Cannot open camera or no frames available: {camera_url}")
        cap.release()
        sys.exit(1)

    print("[OK] Camera opened successfully!")

    known_students = load_known_faces(server_url)
    print(f"[INFO] Loaded {len(known_students)} student profiles")

    if not known_students:
        print("[WARN] No encodings loaded — retrying every 10s for up to 2 minutes...")
        for _ in range(12):
            time.sleep(10)
            known_students = load_known_faces(server_url)
            if known_students:
                print(f"[INFO] Loaded {len(known_students)} student profiles on retry")
                break
        else:
            print("[ERROR] Could not load student encodings after retries. Attendance will not be marked.")

    print("[INFO] Camera stream opened. Starting detection loop…")

    # ── Tracking state ────────────────────────────────────────────────────────
    # student_id -> timestamp when entry was logged
    last_entry_logged: dict[int, float] = {}
    # student_id -> count of consecutive frames absent since last seen
    absence_counter: dict[int, int] = {}
    # student_id -> last known face crop (BGR numpy array) for exit snapshot/emotion
    last_face_crop: dict[int, np.ndarray] = {}
    # student_id -> set of states: 'inside' means entry logged, exit not yet
    currently_inside: set = set()

    frame_count = 0
    detected_faces_total = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed, retrying…")
            time.sleep(2)
            cap = cv2.VideoCapture(src)
            continue

        frame_count += 1

        # ── Face detection ────────────────────────────────────────────────────
        faces = []
        h_frame, w_frame = frame.shape[:2]

        if face_net is not None:
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                         (104.0, 177.0, 123.0), swapRB=False)
            face_net.setInput(blob)
            detections = face_net.forward()
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array(
                        [w_frame, h_frame, w_frame, h_frame])
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_frame, x2), min(h_frame, y2)
                    if x2 > x1 and y2 > y1:
                        faces.append((x1, y1, x2 - x1, y2 - y1))
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))
            if len(detected) > 0:
                faces = list(detected)

        # ── Reload student encodings periodically ─────────────────────────────
        if frame_count % 9000 == 0 or (not known_students and frame_count % 300 == 0):
            fresh = load_known_faces(server_url)
            if fresh:
                known_students = fresh
                print(f"[INFO] Refreshed {len(known_students)} student profiles")

        # ── Print stats every 30 frames ───────────────────────────────────────
        if frame_count % 30 == 0:
            print(f"[DEBUG] Frame {frame_count} | Faces detected so far: {detected_faces_total} "
                  f"| Currently inside: {len(currently_inside)}")

        # ── Batch face-recognition encoding ───────────────────────────────────
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations_css = [(y, x + w, y + h, x) for (x, y, w, h) in faces]

        frame_encodings = []
        if FACE_RECOGNITION_AVAILABLE and known_students and face_locations_css:
            try:
                frame_encodings = face_recognition.face_encodings(
                    rgb_frame, known_face_locations=face_locations_css, num_jitters=1)
            except Exception as e:
                print(f"[ERROR] Batch encoding failed: {e}")

        # ── Process each detected face ────────────────────────────────────────
        detected_in_this_frame: set = set()

        for idx, (x, y, w, h) in enumerate(faces):
            detected_faces_total += 1
            face_roi = frame[y:y + h, x:x + w]

            # ── Match to student ──────────────────────────────────────────────
            matched_student_id = None
            matched_distance = 1.0

            if FACE_RECOGNITION_AVAILABLE and known_students:
                query_enc = frame_encodings[idx] if idx < len(frame_encodings) else None
                if query_enc is not None:
                    best_distance = 1.0
                    best_student_id = None
                    for student in known_students:
                        try:
                            known_enc = np.array(json.loads(student['encoding']))
                            dist = face_recognition.face_distance([known_enc], query_enc)[0]
                            if dist < best_distance:
                                best_distance = dist
                                best_student_id = student['id']
                                matched_distance = dist
                        except Exception as e:
                            print(f"[WARN] Error matching {student['name']}: {e}")
                    if best_distance < 0.6:
                        matched_student_id = best_student_id
                else:
                    print(f"[WARN] No encoding for face at ({x},{y})")
            elif not known_students:
                continue

            if matched_student_id is None:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(frame, f"Unknown ({matched_distance:.2f})", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                continue

            # ── Student recognised ────────────────────────────────────────────
            detected_in_this_frame.add(matched_student_id)

            # Reset absence counter now that they're visible
            absence_counter[matched_student_id] = 0

            # Save a fresh face crop for later exit snapshot / emotion
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w_frame, x + w + padding)
            y2 = min(h_frame, y + h + padding)
            last_face_crop[matched_student_id] = frame[y1:y2, x1:x2].copy()

            now = time.time()

            if matched_student_id not in currently_inside:
                # ── ENTRY ─────────────────────────────────────────────────────
                # Respect cooldown: don't re-log entry too quickly after exit
                last_time = last_entry_logged.get(matched_student_id, 0)
                if (now - last_time) < cooldown_seconds:
                    # Still in cooldown — skip, but draw box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)
                    cv2.putText(frame, "COOLDOWN", (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
                    continue

                emotion, score = detect_emotion(last_face_crop[matched_student_id])
                snapshot = frame_to_base64(face_roi)

                result = log_to_server(server_url, matched_student_id, camera_id, emotion, score, snapshot)
                last_entry_logged[matched_student_id] = now
                currently_inside.add(matched_student_id)

                print(f"[ENTRY] Student {matched_student_id} — emotion: {emotion} ({score}%)")

                label = f"ENTRY | {emotion} ({score:.0f}%)"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 100), 2)
                cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 2)

            else:
                # Already inside — just draw tracking box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                cv2.putText(frame, "INSIDE", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        # ── Absence counter & EXIT detection ─────────────────────────────────
        for student_id in list(currently_inside):
            if student_id not in detected_in_this_frame:
                # Increment absence counter
                absence_counter[student_id] = absence_counter.get(student_id, 0) + 1

                if absence_counter[student_id] >= EXIT_ABSENCE_FRAMES:
                    # ── EXIT ──────────────────────────────────────────────────
                    # Use the saved face crop for a real exit emotion/snapshot
                    exit_crop = last_face_crop.get(student_id)
                    if exit_crop is not None and exit_crop.size > 0:
                        emotion, score = detect_emotion(exit_crop)
                        snapshot = frame_to_base64(exit_crop)
                    else:
                        emotion, score = "unknown", 0.0
                        snapshot = frame_to_base64(frame)

                    result = log_to_server(server_url, student_id, camera_id, emotion, score, snapshot)
                    print(f"[EXIT] Student {student_id} — emotion: {emotion} ({score}%) | "
                          f"status: {result.get('status', 'logged')}")

                    currently_inside.discard(student_id)
                    absence_counter.pop(student_id, None)
                    # Keep last_face_crop a little longer in case they come back

        # ── Optional preview (comment out on headless) ────────────────────────
        # try:
        #     cv2.imshow("Entrance Detection", frame)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # except Exception:
        #     pass

    cap.release()


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Entrance Camera Detection')
        parser.add_argument('--camera-url', default='0', help='Camera URL or webcam index')
        parser.add_argument('--camera-id', type=int, required=True, help='Camera DB ID from admin')
        parser.add_argument('--server', default='http://localhost:8000', help='Django server URL')
        parser.add_argument('--cooldown', type=int, default=30,
                            help='Seconds between re-logging the same student after an exit')
        args = parser.parse_args()

        print(f"[INFO] Starting detection for camera {args.camera_id}")
        print(f"[INFO] Camera URL: {args.camera_url}")
        print(f"[INFO] Server: {args.server}")
        print(f"[INFO] Cooldown: {args.cooldown}s")
        print(f"[INFO] face_recognition available: {FACE_RECOGNITION_AVAILABLE}")
        print(f"[INFO] DeepFace available: {DEEPFACE_AVAILABLE}")

        run_detection(args.camera_url, args.camera_id, args.server, args.cooldown)
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)