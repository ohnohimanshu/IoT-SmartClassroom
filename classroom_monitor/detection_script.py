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
        import os
        # Use absolute path so the model is found whether called from views.py or CLI
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'yolo11s.pt')
        self.yolo = YOLO(model_path)

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

    # COCO food & utensil class IDs
    _FOOD_CLASSES = {
        41: 'cup', 42: 'bowl', 43: 'fork', 44: 'knife', 45: 'spoon',
        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
        50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
        54: 'donut', 55: 'cake',
    }

    def detect(self, frame):
        h, w = frame.shape[:2]
        results = []

        # ── Single pass: persons + phones + food/utensils ─────────────────────
        detect_classes = [0, 67] + list(self._FOOD_CLASSES.keys())
        all_results = self.yolo(frame, classes=detect_classes, verbose=False, conf=0.15)

        persons = []
        phone_boxes = []
        food_boxes = []

        for result in all_results:
            for box in result.boxes:
                cls = int(box.cls[0].cpu().numpy())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                bbox = (int(x1), int(y1), int(x2), int(y2))
                if cls == 0:
                    persons.append({'bbox': bbox, 'confidence': conf})
                elif cls == 67:
                    phone_boxes.append(bbox)
                    print(f"[PHONE] detected conf={conf:.2f}")
                elif cls in self._FOOD_CLASSES:
                    food_boxes.append(bbox)
                    print(f"[FOOD] {self._FOOD_CLASSES[cls]} detected conf={conf:.2f}")

        print(f"[DETECT] persons={len(persons)} phones={len(phone_boxes)} food={len(food_boxes)}")

        # ── Helper: check if two boxes are "near" each other ─────────────────
        def _boxes_near(b1, b2, expand=100):
            """True if b2 is within `expand` pixels of b1 (expanded b1)."""
            ax1, ay1, ax2, ay2 = b1
            bx1, by1, bx2, by2 = b2
            ax1e, ay1e = ax1 - expand, ay1 - expand
            ax2e, ay2e = ax2 + expand, ay2 + expand
            return not (bx2 < ax1e or bx1 > ax2e or by2 < ay1e or by1 > ay2e)

        def _overlap_ratio(b1, b2):
            """Fraction of b2 that overlaps with b1."""
            ax1, ay1, ax2, ay2 = b1
            bx1, by1, bx2, by2 = b2
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            inter = (ix2 - ix1) * (iy2 - iy1)
            b2_area = max(1, (bx2 - bx1) * (by2 - by1))
            return inter / b2_area

        # ── Map phones to persons ─────────────────────────────────────────────
        phone_person_indices = set()
        for pb in phone_boxes:
            for i, p in enumerate(persons):
                bx1, by1, bx2, by2 = p['bbox']
                expanded = (bx1, by1, bx2, by2 + 100)  # extend down for desk/lap
                if _overlap_ratio(expanded, pb) > 0.30:
                    phone_person_indices.add(i)
                    print(f"[PHONE] mapped to person {i}")

        # ── Map food to persons ───────────────────────────────────────────────
        food_person_indices = set()
        for fb in food_boxes:
            for i, p in enumerate(persons):
                if _boxes_near(p['bbox'], fb, expand=120):
                    food_person_indices.add(i)
                    print(f"[FOOD] mapped to person {i}")

        # ── Fight detection: persons with high body overlap ───────────────────
        # Two people "fighting" when their person boxes overlap significantly
        # (bodies pressed together) — flag both
        fight_person_indices = set()
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                bi = persons[i]['bbox']
                bj = persons[j]['bbox']
                # Check overlap in both directions
                ov_i = _overlap_ratio(bj, bi)
                ov_j = _overlap_ratio(bi, bj)
                if ov_i > 0.25 or ov_j > 0.25:
                    fight_person_indices.add(i)
                    fight_person_indices.add(j)
                    print(f"[FIGHT] persons {i} and {j} overlap ov_i={ov_i:.2f} ov_j={ov_j:.2f}")

        # ── Build results per person ──────────────────────────────────────────
        # Priority: fight > phone > eating > head_pose
        for i, person in enumerate(persons):
            x1, y1, x2, y2 = person['bbox']
            confidence = person['confidence']
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            zone_id = self.get_zone_id(x_center, y_center, w, h)

            if i in fight_person_indices:
                results.append({
                    "zone_id": zone_id, "pose": "fighting",
                    "possibly_talking": False, "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                })
                continue

            if i in phone_person_indices:
                results.append({
                    "zone_id": zone_id, "pose": "using_phone",
                    "possibly_talking": False, "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                })
                continue

            if i in food_person_indices:
                results.append({
                    "zone_id": zone_id, "pose": "eating",
                    "possibly_talking": False, "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                })
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                results.append({
                    "zone_id": zone_id, "pose": "not_visible",
                    "possibly_talking": False, "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                })
                continue

            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                    data=np.ascontiguousarray(rgb_crop))
            face_result = self.face_landmarker.detect(mp_img)

            if face_result.face_landmarks:
                lm = face_result.face_landmarks[0]
                yaw, pitch = self.calculate_head_pose(lm, crop.shape)
                pose = self.classify_pose(yaw, pitch)
                mar = self.calculate_mar(lm)
                results.append({
                    "zone_id": zone_id, "pose": pose,
                    "possibly_talking": mar > 0.3,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                })
            else:
                results.append({
                    "zone_id": zone_id, "pose": "not_visible",
                    "possibly_talking": False, "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                })

        return results

    def draw_bboxes(self, frame, detections):
        COLOR_MAP = {
            "focused":      (0, 255, 0),      # green
            "looking_away": (0, 255, 255),    # yellow
            "head_down":    (0, 165, 255),    # orange
            "using_phone":  (0, 0, 255),      # red
            "eating":       (255, 165, 0),    # blue
            "fighting":     (0, 0, 180),      # dark red — flashing handled in UI
            "not_visible":  (128, 128, 128),  # grey
        }
        LABEL_MAP = {
            "focused":      "Focused",
            "looking_away": "Looking Away",
            "head_down":    "Head Down",
            "using_phone":  "Using Phone",
            "eating":       "Eating",
            "fighting":     "FIGHT DETECTED",
            "not_visible":  "Not Visible",
        }
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            pose = det["pose"]
            color = COLOR_MAP.get(pose, (128, 128, 128))
            label = LABEL_MAP.get(pose, pose)
            thickness = 3 if pose == "fighting" else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label_y = y1 - 10 if y1 > 20 else y2 + 18
            cv2.putText(frame, f"Zone {det['zone_id']}: {label}",
                        (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
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
