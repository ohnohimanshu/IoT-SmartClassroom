"""
Classroom Engagement Detection Script
Run on the server or any machine with camera network access:

    python detection_script.py \
        --camera-url rtsp://192.168.1.100:554/stream \
        --camera-id 1 \
        --server http://localhost:8000 \
        --interval 30

Requirements: pip install opencv-python mediapipe ultralytics requests numpy
"""

import argparse
import os
import cv2
import numpy as np
import requests
import base64
import time
from datetime import datetime

# ── GPU: tell CUDA/TF to grow memory on demand rather than pre-allocating ──────
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

# Force OpenCV to use CUDA backend when available
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"[GPU] OpenCV CUDA enabled — {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
    else:
        print("[GPU] No CUDA-capable GPU found for OpenCV, using CPU")
except Exception:
    print("[GPU] OpenCV built without CUDA support, using CPU")


def parse_args():
    parser = argparse.ArgumentParser(description="Classroom Engagement Detection")
    parser.add_argument("--camera-url", required=True,
                        help="Camera stream URL (RTSP/HTTP) or device index (0,1,...)")
    parser.add_argument("--camera-id", required=True, type=int,
                        help="ClassroomCamera ID in Django")
    parser.add_argument("--server", default="http://localhost:8000",
                        help="Django server URL")
    parser.add_argument("--interval", type=int, default=30,
                        help="Interval between snapshots in seconds")
    return parser.parse_args()


class EngagementDetector:
    def __init__(self):
        # YOLO — will automatically use GPU if CUDA is available
        from ultralytics import YOLO
        self.yolo = YOLO("yolov8n.pt")

        # Import here so mediapipe is accessed after full module load
        import mediapipe as mp
        import os

        # mediapipe >= 0.10 uses Tasks API — download model if needed
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'face_landmarker.task')
        if not os.path.exists(model_path):
            print("[INFO] Downloading face_landmarker.task model...")
            import urllib.request
            urllib.request.urlretrieve(
                'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                model_path
            )

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=10,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._mp = mp
        self._mp_vision = mp_vision
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def get_zone_id(self, x_center, y_center, frame_width, frame_height, num_zones=10):
        cols, rows = 4, 3
        col = min(int(x_center / (frame_width / cols)), cols - 1)
        row = min(int(y_center / (frame_height / rows)), rows - 1)
        zone_id = row * cols + col + 1
        return min(zone_id, num_zones)

    def calculate_head_pose(self, landmarks, image_shape):
        h, w = image_shape[:2]

        nose_tip  = landmarks[1]
        left_eye  = landmarks[33]
        right_eye = landmarks[263]
        chin      = landmarks[152]
        forehead  = landmarks[10]

        nose_2d     = np.array([nose_tip.x * w,  nose_tip.y * h])
        left_eye_2d = np.array([left_eye.x * w,  left_eye.y * h])
        right_eye_2d= np.array([right_eye.x * w, right_eye.y * h])
        chin_2d     = np.array([chin.x * w,      chin.y * h])
        forehead_2d = np.array([forehead.x * w,  forehead.y * h])

        eye_center = (left_eye_2d + right_eye_2d) / 2
        eye_width  = np.linalg.norm(right_eye_2d - left_eye_2d)

        nose_offset_x = nose_2d[0] - eye_center[0]
        yaw = (nose_offset_x / (eye_width + 1e-6)) * 90

        face_height   = np.linalg.norm(chin_2d - forehead_2d) + 1e-6
        nose_offset_y = nose_2d[1] - eye_center[1]
        pitch = (nose_offset_y / face_height) * 180

        return yaw, pitch

    def classify_pose(self, yaw, pitch):
        if -15 <= yaw <= 15 and pitch > -10:
            return "focused"
        elif yaw < -15 or yaw > 15:
            return "looking_away"
        elif pitch < -10:
            return "head_down"
        return "not_visible"

    def calculate_mar(self, landmarks):
        upper_lip  = landmarks[13]
        lower_lip  = landmarks[14]
        left_mouth = landmarks[78]
        right_mouth= landmarks[308]

        vertical   = np.hypot(upper_lip.x - lower_lip.x,  upper_lip.y - lower_lip.y)
        horizontal = np.hypot(left_mouth.x - right_mouth.x, left_mouth.y - right_mouth.y)
        return vertical / (horizontal + 1e-6)

    def detect(self, frame):
        h, w = frame.shape[:2]
        results = []

        yolo_results = self.yolo(frame, classes=[0], verbose=False)

        for result in yolo_results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())

                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                zone_id  = self.get_zone_id(x_center, y_center, w, h)

                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    results.append({
                        "zone_id": zone_id, "pose": "not_visible",
                        "possibly_talking": False, "confidence": confidence,
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    })
                    continue

                rgb_crop    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                        data=np.ascontiguousarray(rgb_crop))
                face_result = self.face_landmarker.detect(mp_img)

                if face_result.face_landmarks:
                    lm   = face_result.face_landmarks[0]
                    yaw, pitch = self.calculate_head_pose(lm, crop.shape)
                    pose = self.classify_pose(yaw, pitch)
                    mar  = self.calculate_mar(lm)
                    results.append({
                        "zone_id": zone_id, "pose": pose,
                        "possibly_talking": mar > 0.3,
                        "confidence": confidence,
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    })
                else:
                    results.append({
                        "zone_id": zone_id, "pose": "not_visible",
                        "possibly_talking": False, "confidence": confidence,
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    })

        return results

    def draw_bboxes(self, frame, detections):
        COLOR_MAP = {
            "focused":      (0, 255, 0),
            "looking_away": (0, 255, 255),
            "head_down":    (0, 0, 255),
            "not_visible":  (128, 128, 128),
        }
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = COLOR_MAP.get(det["pose"], (128, 128, 128))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Zone {det['zone_id']}: {det['pose']}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if det["possibly_talking"]:
                cv2.circle(frame, (x1 + 20, y1 + 20), 8, (0, 165, 255), -1)
        return frame


def open_camera(camera_url: str):
    src = int(camera_url) if camera_url.isdigit() else camera_url
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap, src


def main():
    args = parse_args()
    detector = EngagementDetector()

    print(f"[*] Connecting to camera: {args.camera_url}")
    cap, src = open_camera(args.camera_url)

    if not cap.isOpened():
        print("[!] Failed to open camera")
        return

    print(f"[*] Waiting for active session on camera {args.camera_id}...")
    session_id = None
    while session_id is None:
        try:
            r = requests.get(
                f"{args.server}/classroom/api/session/active/",
                params={"camera_id": args.camera_id},
                timeout=5,
            )
            if r.status_code == 200:
                data = r.json()
                if data.get("is_active"):
                    session_id = data["session_id"]
                    print(f"[+] Active session found: {session_id}")
                else:
                    time.sleep(10)
            else:
                time.sleep(10)
        except Exception as e:
            print(f"[!] Error checking session: {e}")
            time.sleep(10)

    print(f"[*] Starting detection loop (interval: {args.interval}s)")
    last_snapshot = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Lost camera connection, retrying in 5s...")
                time.sleep(5)
                cap.release()
                cap, src = open_camera(args.camera_url)
                continue

            if time.time() - last_snapshot >= args.interval:
                last_snapshot = time.time()

                detections = detector.detect(frame)
                annotated  = detector.draw_bboxes(frame.copy(), detections)

                _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')

                # Build per-zone payload (zones 1-10)
                zone_map = {d["zone_id"]: d for d in detections}
                students_payload = []
                for zone_id in range(1, 11):
                    if zone_id in zone_map:
                        d = zone_map[zone_id]
                        students_payload.append({
                            "zone_id": zone_id, "pose": d["pose"],
                            "possibly_talking": d["possibly_talking"],
                            "confidence": d["confidence"],
                        })
                    else:
                        students_payload.append({
                            "zone_id": zone_id, "pose": "not_visible",
                            "possibly_talking": False, "confidence": 0.0,
                        })

                payload = {
                    "camera_id": args.camera_id,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "students": students_payload,
                    "frame_snapshot_b64": frame_b64,
                }

                try:
                    r = requests.post(
                        f"{args.server}/classroom/api/snapshot/",
                        json=payload, timeout=10,
                    )
                    if r.status_code == 200:
                        focused = sum(1 for d in students_payload if d["pose"] == "focused")
                        total   = sum(1 for d in students_payload if d["pose"] != "not_visible")
                        eng     = (focused / total * 100) if total > 0 else 0
                        talking = [f"Zone {d['zone_id']}" for d in students_payload if d["possibly_talking"]]
                        ts = datetime.now().strftime('%H:%M:%S')
                        print(f"[{ts}] Engagement: {eng:.0f}% | Talking: {', '.join(talking) or 'None'}")
                    else:
                        print(f"[!] Server error: {r.status_code} — {r.text}")
                except Exception as e:
                    print(f"[!] Failed to post snapshot: {e}")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[*] Stopping...")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
