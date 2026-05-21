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
        
        # DeepFace analyze
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
    cooldown_seconds: minimum time between logging the same student (unused, using frame-based detection instead)
    """
    print("[INFO] Initializing face detection...")

    # DNN face detector — far more robust than Haar cascade for IP cameras
    # Uses OpenCV's built-in res10_300x300 SSD model (no extra download needed)
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
        # Fallback: Haar cascade with relaxed params
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("[INFO] Using Haar cascade face detector (DNN model not found)")

    # Try to open camera with timeout
    print(f"[INFO] Attempting to open camera: {camera_url}")
    try:
        src = int(camera_url)  # webcam index
        print(f"[INFO] Camera is a webcam (index: {src})")
    except ValueError:
        src = camera_url       # IP stream URL
        print(f"[INFO] Camera is an IP stream: {camera_url}")

    cap = cv2.VideoCapture(src)
    
    # Set timeout for camera opening attempt
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Try to read a frame with timeout
    start_time = time.time()
    timeout = 5  # 5 second timeout
    frame_read = False
    
    while (time.time() - start_time) < timeout:
        ret, frame = cap.read()
        if ret:
            frame_read = True
            break
        time.sleep(0.1)
    
    if not frame_read:
        print(f"[FATAL] Cannot open camera or no frames available: {camera_url}")
        print(f"[FATAL] If using IP camera, verify it is accessible and online")
        print(f"[FATAL] If using webcam, try camera index 0")
        cap.release()
        sys.exit(1)
    
    print("[OK] Camera opened successfully!")
    
    known_students = load_known_faces(server_url)
    print(f"[INFO] Loaded {len(known_students)} student profiles")

    # Retry loading encodings if server was unreachable at startup
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

    print(f"[INFO] Camera stream opened. Starting detection loop…")
    last_logged = {}  # student_id -> timestamp of last log
    currently_visible = set()  # students currently visible in frame
    frame_count = 0
    detected_faces = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed, retrying…")
            time.sleep(2)
            cap = cv2.VideoCapture(src)
            continue

        frame_count += 1

        # ── Face detection ────────────────────────────────────────────────────
        faces = []  # list of (x, y, w, h)
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
            # Relaxed params: smaller minSize, lower minNeighbors
            detected = face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))
            if len(detected) > 0:
                faces = list(detected)

        # Reload student encodings every 5 minutes, or immediately if still empty
        if frame_count % 9000 == 0 or (not known_students and frame_count % 300 == 0):
            fresh = load_known_faces(server_url)
            if fresh:
                known_students = fresh
                print(f"[INFO] Loaded {len(known_students)} student profiles")
            elif not known_students and frame_count % 300 == 0:
                print(f"[WARN] Still no student encodings — retrying...")

        # Print stats every 30 frames
        if frame_count % 30 == 0:
            print(f"[DEBUG] Processed {frame_count} frames, detected {detected_faces} faces so far")

        # ── Batch encode all faces in this frame at once ─────────────────────
        # Pass the full RGB frame + known locations to face_recognition.
        # face_recognition expects locations as (top, right, bottom, left) — CSS order.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations_css = []  # (top, right, bottom, left)
        for (x, y, w, h) in faces:
            face_locations_css.append((y, x + w, y + h, x))

        frame_encodings = []
        if FACE_RECOGNITION_AVAILABLE and known_students and face_locations_css:
            try:
                frame_encodings = face_recognition.face_encodings(
                    rgb_frame, known_face_locations=face_locations_css, num_jitters=1)
            except Exception as e:
                print(f"[ERROR] Batch encoding failed: {e}")

        # Track detected faces in this frame
        detected_in_this_frame = set()

        for idx, (x, y, w, h) in enumerate(faces):
            detected_faces += 1
            face_roi = frame[y:y+h, x:x+w]

            # Match face to student
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
                # Draw unrecognised box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                cv2.putText(frame, f"Unknown ({matched_distance:.2f})", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)
                continue

            # Track this student as visible in current frame
            detected_in_this_frame.add(matched_student_id)

            # Check if we should log entry (first time seeing this student)
            now = time.time()
            if matched_student_id not in last_logged:
                # Detect emotion on a larger region around face for better accuracy
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                face_region_padded = frame[y1:y2, x1:x2]
                
                emotion, score = detect_emotion(face_region_padded)

                # Snapshot (use original face_roi for cleaner snapshot)
                snapshot = frame_to_base64(face_roi)

                # Log entry to server
                result = log_to_server(server_url, matched_student_id, camera_id, emotion, score, snapshot)
                last_logged[matched_student_id] = now
                
                print(f"[ENTRY] Logged for student {matched_student_id} - emotion: {emotion} ({score}%)")

                # Draw recognised box
                label = f"ENTRY | {emotion} ({score}%)"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 100), 2)
                cv2.putText(frame, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,100), 2)
            else:
                # Already logged entry, just track visibility
                cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 2)
                cv2.putText(frame, "IN_FRAME", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 2)

        # Detect students who left the frame (log exit)
        students_who_left = currently_visible - detected_in_this_frame
        for student_id in students_who_left:
            # Detect emotion and take snapshot from frame (if available)
            emotion = "unknown"
            score = 0.0
            snapshot = frame_to_base64(frame[max(0, frame.shape[0]//3):min(frame.shape[0], 2*frame.shape[0]//3),
                                           max(0, frame.shape[1]//3):min(frame.shape[1], 2*frame.shape[1]//3)])
            
            # Log exit to server
            result = log_to_server(server_url, student_id, camera_id, emotion, score, snapshot)
            print(f"[EXIT] Logged for student {student_id} - {result.get('status', 'logged')}")
            
            # Remove from tracking
            if student_id in last_logged:
                del last_logged[student_id]

        # Update currently visible students for next frame
        currently_visible = detected_in_this_frame

        # Optional: show preview window (disabled for headless systems)
        # Uncomment below if running on a system with GUI support
        # try:
        #     cv2.imshow("Entrance Detection", frame)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # except:
        #     pass  # GUI not available on headless systems

    cap.release()
    # Don't call cv2.destroyAllWindows() on headless systems
    # try:
    #     cv2.destroyAllWindows()
    # except:
    #     pass


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Entrance Camera Detection')
        parser.add_argument('--camera-url', default='0', help='Camera URL or webcam index')
        parser.add_argument('--camera-id', type=int, required=True, help='Camera DB ID from admin')
        parser.add_argument('--server', default='http://localhost:8000', help='Django server URL')
        parser.add_argument('--cooldown', type=int, default=30, help='Seconds between re-logging same student')
        args = parser.parse_args()

        print(f"[INFO] Starting detection for camera {args.camera_id}")
        print(f"[INFO] Camera URL: {args.camera_url}")
        print(f"[INFO] Server: {args.server}")
        print(f"[INFO] face_recognition available: {FACE_RECOGNITION_AVAILABLE}")
        print(f"[INFO] DeepFace available: {DEEPFACE_AVAILABLE}")
        
        run_detection(args.camera_url, args.camera_id, args.server, args.cooldown)
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
