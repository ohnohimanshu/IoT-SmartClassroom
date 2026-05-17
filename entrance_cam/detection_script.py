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

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
    
    # Debug: Show which students have encodings
    if known_students:
        for s in known_students[:3]:  # Show first 3
            print(f"  - {s['name']} (ID: {s['id']}, encoding length: {len(json.loads(s['encoding']))})")
    else:
        print("[WARN] No students with face encodings found! Did you upload photos and run generate_face_encodings?")

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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        # Print stats every 30 frames
        if frame_count % 30 == 0:
            print(f"[DEBUG] Processed {frame_count} frames, detected {detected_faces} faces so far")

        # Track detected faces in this frame
        detected_in_this_frame = set()

        for (x, y, w, h) in faces:
            detected_faces += 1
            face_roi = frame[y:y+h, x:x+w]

            # Match face to student
            matched_student_id = None
            matched_distance = 1.0
            
            if FACE_RECOGNITION_AVAILABLE and known_students:
                try:
                    rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_face)
                    
                    if encodings:
                        query_enc = encodings[0]
                        
                        # Find closest match
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
                        
                        # Use lower threshold (0.6 is more permissive than 0.5)
                        if best_distance < 0.6:
                            matched_student_id = best_student_id
                    else:
                        print(f"[WARN] No face encoding extracted from frame region")
                        
                except Exception as e:
                    print(f"[ERROR] Face matching error: {e}")
            elif not known_students:
                print("[ERROR] No students with face encodings. Check admin panel.")
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
