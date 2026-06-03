import cv2
import json
import time
import base64
import numpy as np
import threading
import os
from collections import defaultdict, deque

# ── Env loading: reads .env file directly so it works even if Django
#    didn't load python-decouple / django-environ ──────────────────────────────
def _load_env_file():
    """Load .env from project root (same dir as manage.py) into os.environ."""
    for candidate in [
        os.path.join(os.path.dirname(__file__), '..', '.env'),   # app/../.env
        os.path.join(os.path.dirname(__file__), '..', '..', '.env'),
        '.env',
    ]:
        path = os.path.abspath(candidate)
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, _, v = line.partition('=')
                        k = k.strip(); v = v.strip().strip('"').strip("'")
                        os.environ.setdefault(k, v)
            print(f'[ENV] Loaded {path}')
            return
    print('[ENV] No .env file found — relying on system environment')

_load_env_file()

import urllib3
import threading as _threading
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

# OpenCV's CascadeClassifier detectMultiScale is not thread-safe on Windows.
# All Haar cascade calls must hold this lock.
CV2_CASCADE_LOCK = _threading.Lock()

# ── Colour / label maps (exported for views.py) ───────────────────────────────
COLOR_MAP = {
    'focused':      (0,  200,  60),   # green
    'looking_away': (0,  165, 255),   # orange
    'head_down':    (0,  165, 255),   # orange
    'distracted':   (0,  165, 255),   # orange
    'using_phone':  (0,    0, 220),   # red
    'eating_food':  (0,    0, 220),   # red
    'not_visible':  (120, 120, 120),
}
LABEL_MAP = {
    'focused':      'Focused',
    'looking_away': 'Looking Away',
    'head_down':    'Head Down',
    'distracted':   'Distracted',
    'using_phone':  'Using Phone',
    'eating_food':  'Eating Food',
    'not_visible':  'Not Visible',
}

ALERT_POSES      = {'using_phone', 'eating_food'}
DISTRACTED_POSES = {'looking_away', 'head_down', 'distracted'}
STORE_POSES      = ALERT_POSES | DISTRACTED_POSES

# ── 3-D model points for solvePnP (used when face landmarks are available) ────
_MODEL_PTS = np.array([
    ( 0.0,    0.0,   0.0),   # nose tip      — landmark 30
    ( 0.0,  -63.6, -12.5),   # chin          — landmark 8
    (-43.3,  32.7, -26.0),   # left eye out  — landmark 36
    ( 43.3,  32.7, -26.0),   # right eye out — landmark 45
    (-28.9, -28.9, -24.1),   # left mouth    — landmark 48
    ( 28.9, -28.9, -24.1),   # right mouth   — landmark 54
], dtype=np.float64)


class ClassroomBehaviorDetector:
    """
    Detects student behaviours from BGR frames.

    Head-pose uses OpenCV Haar cascades (zero extra downloads):
      frontal cascade fires  → focused
      profile cascade fires  → looking_away
      neither fires          → head_down
    """

    FIGHT_IOU_THRESH   = 0.45   # min IoU for fighting (DEPRECATED - not used)
    FIGHT_MOTION_THRESH = 800   # min pixel-diff in overlap zone (DEPRECATED - not used)
    YAW_THRESH         = 20     # fallback degrees if solvePnP is used
    PITCH_THRESH       = -18

    def __init__(self, camera_url, camera_id,
                 server_url='http://localhost:8000',
                 alert_cooldown=120,
                 whatsapp_admin=None):
        self.camera_url     = camera_url
        self.camera_id      = camera_id
        self.server_url     = server_url
        self.alert_cooldown = alert_cooldown
        self.whatsapp_admin = whatsapp_admin or os.environ.get('ADMIN_WHATSAPP', '')

        self.yolo_model     = None
        self.face_recognizer = None
        self.known_students  = []

        # OpenCV Haar cascades — always available, no download
        self._frontal_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self._frontal_alt     = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self._profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml')
        print('[OK] Haar cascades loaded (frontal + profile)')

        # Motion buffer for fighting validation (last 2 frames)
        self._prev_gray  = None

        self.last_alert_time = defaultdict(float)
        self.running = False
        self.thread  = None

        self._load_models()

    # ─────────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_models(self):
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolo11s.pt')
            print('[OK] YOLO loaded')
        except Exception as e:
            print(f'[WARN] YOLO: {e}')

        # face_recognition / dlib is NOT loaded here.
        # The caller uses face_recognition_helper.StudentFaceRecognizer
        # with its DLIB_LOCK to prevent Windows thread-safety crashes.

    def _load_known_students(self):
        import requests as _req
        try:
            resp = _req.get(f'{self.server_url}/api/students/encodings/',
                            timeout=5, verify=False)
            self.known_students = resp.json()
            print(f'[OK] {len(self.known_students)} student encodings loaded')
        except Exception as e:
            print(f'[WARN] Could not load students: {e}')

    # ─────────────────────────────────────────────────────────────────────────
    # Head-pose classification using Haar cascades
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_head_pose(self, gray_frame, x1, y1, x2, y2):
        """
        Returns 'focused' | 'looking_away' | 'head_down'

        Strategy:
          1. Crop the person box, run frontal face detector
             → any frontal face found = FOCUSED
          2. Run profile face detector (left-facing)
             → found = LOOKING AWAY
          3. Flip crop horizontally, run profile again (right-facing)
             → found = LOOKING AWAY
          4. Neither → HEAD DOWN (looking at desk / not visible)
        """
        h, w = gray_frame.shape
        px1 = max(0, x1); py1 = max(0, y1)
        px2 = min(w, x2); py2 = min(h, y2)
        crop = gray_frame[py1:py2, px1:px2]
        if crop.size == 0:
            return 'not_visible'

        # Scale down for speed if crop is large
        scale = 1.0
        ch, cw = crop.shape
        if cw > 200:
            scale = 200 / cw
            crop  = cv2.resize(crop, (int(cw*scale), int(ch*scale)))

        def _detect(cascade, img, scale_factor=1.15, min_neighbors=3):
            with CV2_CASCADE_LOCK:
                faces = cascade.detectMultiScale(
                    img, scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(20, 20),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
            return len(faces) > 0

        # 1. Frontal
        if _detect(self._frontal_cascade, crop) or _detect(self._frontal_alt, crop):
            return 'focused'

        # 2. Profile (left or right)
        if _detect(self._profile_cascade, crop):
            return 'looking_away'
        if _detect(self._profile_cascade, cv2.flip(crop, 1)):
            return 'looking_away'

        # 3. Neither — head down / not visible
        return 'head_down'

    # ─────────────────────────────────────────────────────────────────────────
    # Motion check for fighting validation
    # ─────────────────────────────────────────────────────────────────────────

    def _motion_in_zone(self, gray_frame, ix1, iy1, ix2, iy2):
        """
        Returns True if pixel difference in the overlap zone between
        current and previous frame exceeds FIGHT_MOTION_THRESH.
        Prevents flagging two stationary students as fighting.
        """
        if self._prev_gray is None:
            return False
        try:
            h, w = gray_frame.shape
            zx1, zy1 = max(0,ix1), max(0,iy1)
            zx2, zy2 = min(w,ix2), min(h,iy2)
            if zx2 <= zx1 or zy2 <= zy1:
                return False
            curr_zone = gray_frame[zy1:zy2, zx1:zx2].astype(np.int16)
            prev_zone = self._prev_gray[zy1:zy2, zx1:zx2].astype(np.int16)
            if curr_zone.shape != prev_zone.shape:
                return False
            diff = np.sum(np.abs(curr_zone - prev_zone))
            return int(diff) > self.FIGHT_MOTION_THRESH
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Main detection
    # ─────────────────────────────────────────────────────────────────────────

    def _overlaps(self, pb, ob):
        """Check if object box overlaps with person box."""
        return not (ob[2] < pb[0] or ob[0] > pb[2] or
                    ob[3] < pb[1] or ob[1] > pb[3])

    def detect(self, frame):
        """
        Analyse a BGR frame. Returns list of detection dicts:
          type, bbox, confidence, color, label, is_alert, is_distracted
        """
        if self.yolo_model is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            results = self.yolo_model.predict(frame, verbose=False, conf=0.4)
        except Exception as e:
            print(f'[ERROR] YOLO: {e}')
            return []

        person_boxes = []   # (x1,y1,x2,y2,conf)
        object_boxes = []   # (x1,y1,x2,y2,obj_type)

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cid  = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cid == 0:
                    person_boxes.append((x1, y1, x2, y2, conf))
                elif cid == 67 and conf > 0.50:           # cell phone
                    object_boxes.append((x1, y1, x2, y2, 'using_phone'))
                elif cid in range(46, 56) and conf > 0.50:  # food
                    object_boxes.append((x1, y1, x2, y2, 'eating_food'))

        # ── Classify each person ─────────────────────────────────────────────
        detections = []
        for i, (x1, y1, x2, y2, conf) in enumerate(person_boxes):

            # Object overlap → phone / food
            obj_type = next(
                (ob[4] for ob in object_boxes if self._overlaps((x1,y1,x2,y2), ob)),
                None
            )
            if obj_type:
                det_type = obj_type
            else:
                # Head pose via Haar cascades
                det_type = self._classify_head_pose(gray, x1, y1, x2, y2)

            detections.append({
                'type':          det_type,
                'bbox':          (x1, y1, x2, y2),
                'confidence':    conf,
                'color':         COLOR_MAP.get(det_type, (120,120,120)),
                'label':         LABEL_MAP.get(det_type, det_type),
                'is_alert':      det_type in ALERT_POSES,
                'is_distracted': det_type in DISTRACTED_POSES,
            })

        # Update motion buffer
        self._prev_gray = gray.copy()

        return detections

    # Keep backward-compat alias
    def _detect_behaviors(self, frame):
        return self.detect(frame)

    # ─────────────────────────────────────────────────────────────────────────
    # Face recognition
    # ─────────────────────────────────────────────────────────────────────────

    def _recognize_face(self, frame, bbox):
        """Returns (student_id, name, roll_no) or (None, 'Unknown', '')."""
        if self.face_recognizer is None or not self.known_students:
            return None, 'Unknown', ''
        try:
            x1, y1, x2, y2 = bbox
            # Use upper-body region (top 60%) where face is likely
            mid_y = y1 + int((y2 - y1) * 0.55)
            crop  = frame[y1:mid_y, x1:x2]
            if crop.size == 0:
                crop = frame[y1:y2, x1:x2]
            rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            encs = self.face_recognizer.face_encodings(
                rgb, num_jitters=1, model='small')
            if not encs:
                return None, 'Unknown', ''
            detected  = encs[0]
            best_dist = 1.0
            best      = None
            for s in self.known_students:
                try:
                    known = np.array(json.loads(s['encoding']))
                    d = self.face_recognizer.face_distance([known], detected)[0]
                    if d < best_dist:
                        best_dist, best = d, s
                except Exception:
                    continue
            if best_dist < 0.55 and best:
                return best['id'], best['name'], best.get('roll_no', '')
            return None, 'Unknown', ''
        except Exception as e:
            print(f'[ERROR] face_recog: {e}')
            return None, 'Unknown', ''

    # ─────────────────────────────────────────────────────────────────────────
    # Draw detections on frame
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_detections(self, frame, detections):
        """
        Draw rectangles and labels on frame for all detections.
        Returns annotated frame.
        """
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = det['color']
            label = det['label']
            conf = det['confidence']
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            text = f"{label} ({conf:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Background for text
            bg_x1 = x1
            bg_y1 = max(0, y1 - text_size[1] - 4)
            bg_x2 = x1 + text_size[0] + 4
            bg_y2 = y1
            cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            
            # Text
            cv2.putText(annotated, text, (x1 + 2, y1 - 2), font, 
                       font_scale, (255, 255, 255), thickness)
        
        return annotated

    # ─────────────────────────────────────────────────────────────────────────
    # Report to Django API (DB save + WhatsApp)
    # ─────────────────────────────────────────────────────────────────────────

    def _report_incident(self, detection, frame, student_id, student_name, roll_no, all_detections=None):
        import requests as _req
        try:
            # Draw rectangles on frame if all_detections provided
            if all_detections:
                annotated_frame = self._draw_detections(frame, all_detections)
            else:
                annotated_frame = frame
            
            _, buf = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            snap_b64 = base64.b64encode(buf).decode()
            tag = f'{student_name} ({roll_no})' if student_id else 'Unknown person'
            resp = _req.post(
                f'{self.server_url}/classroom/api/incidents/report/',
                json={
                    'student_id':    student_id,
                    'camera_id':     self.camera_id,
                    'incident_type': detection['type'],
                    'confidence':    round(detection['confidence'], 3),
                    'snapshot':      snap_b64,
                    'student_name':  student_name,
                    'roll_no':       roll_no,
                    'description':   f"{detection['label']} — {tag}",
                    'send_whatsapp': detection['is_alert'],
                },
                timeout=10, verify=False,
            )
            print(f"[INCIDENT] {detection['label']} | {tag} | "
                  f"http={resp.status_code}")
        except Exception as e:
            print(f'[ERROR] report_incident: {e}')

    # ─────────────────────────────────────────────────────────────────────────
    # Background loop (standalone camera mode)
    # ─────────────────────────────────────────────────────────────────────────

    def _detection_loop(self):
        cap = cv2.VideoCapture(self.camera_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f'[FATAL] Cannot open: {self.camera_url}')
            return
        print('[OK] Camera opened')
        frame_count = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print('[WARN] Frame read failed, reconnecting…')
                time.sleep(2)
                cap = cv2.VideoCapture(self.camera_url)
                continue

            frame_count += 1
            detections = self.detect(frame)

            for det in detections:
                if not (det['is_alert'] or det['is_distracted']):
                    continue
                now = time.time()
                key = det['type']
                if (now - self.last_alert_time.get(key, 0)) < self.alert_cooldown:
                    continue
                sid, name, roll = self._recognize_face(frame, det['bbox'])
                self._report_incident(det, frame, sid, name, roll, all_detections=detections)
                self.last_alert_time[key] = now

            if frame_count % 300 == 0:
                print(f'[INFO] Frame {frame_count} | {len(detections)} detections')

        cap.release()
        print('[OK] Detection stopped')

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print('[OK] Behavior detection started')

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print('[OK] Behavior detection stopped')