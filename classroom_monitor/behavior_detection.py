import cv2
import json
import time
import base64
import numpy as np
import threading
import os
from collections import defaultdict, deque

# ── Fight detection ──────────────────────────────────────────────────────────
from .fight_detection import FightDetector

# ── Env loading ───────────────────────────────────────────────────────────────
def _load_env_file():
    """Load .env from project root into os.environ."""
    for candidate in [
        os.path.join(os.path.dirname(__file__), '..', '.env'),
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

# OpenCV's CascadeClassifier.detectMultiScale is not thread-safe on Windows.
CV2_CASCADE_LOCK = _threading.Lock()

# ── Colour / label maps ───────────────────────────────────────────────────────
COLOR_MAP = {
    'focused':      (0,  200,  60),
    'looking_away': (0,  165, 255),
    'head_down':    (0,  165, 255),
    'distracted':   (0,  165, 255),
    'using_phone':  (0,    0, 220),
    'eating_food':  (0,    0, 220),
    'fighting':     (0,    0, 255),
    'not_visible':  (120, 120, 120),
}
LABEL_MAP = {
    'focused':      'Focused',
    'looking_away': 'Looking Away',
    'head_down':    'Head Down',
    'distracted':   'Distracted',
    'using_phone':  'Using Phone',
    'eating_food':  'Eating Food',
    'fighting':     'FIGHT',
    'not_visible':  'Not Visible',
}

ALERT_POSES      = {'using_phone', 'eating_food', 'fighting'}
DISTRACTED_POSES = {'looking_away', 'head_down', 'distracted'}
STORE_POSES      = ALERT_POSES | DISTRACTED_POSES

# ── 3-D model points for solvePnP ────────────────────────────────────────────
_MODEL_PTS = np.array([
    ( 0.0,    0.0,   0.0),
    ( 0.0,  -63.6, -12.5),
    (-43.3,  32.7, -26.0),
    ( 43.3,  32.7, -26.0),
    (-28.9, -28.9, -24.1),
    ( 28.9, -28.9, -24.1),
], dtype=np.float64)


class ClassroomBehaviorDetector:
    """
    Detects student behaviours from BGR frames.

    FIXES APPLIED:
    ─────────────────────────────────────────────────────────────────────────
    BUG A (behavior_detection.py — _classify_head_pose):
      Two students talking face-to-face are both FRONTAL in a wide crop.
      The Haar frontal cascade detects BOTH faces in the FULL person crop,
      so the system correctly labels them 'focused' — which is WRONG for
      two students talking off-task.

      Root cause: the person crop at scale ≤ 200 px is too large; it often
      contains the NEIGHBOUR's face as well (desk rows are close together).

      FIX 1 — narrow the crop horizontally to ±40% of width around centre,
               so only the target student's face region is examined.
      FIX 2 — reduce minNeighbors from 3 to 4 for the frontal cascade to
               reduce neighbour-face false positives.
      FIX 3 — if TWO or more frontal faces are detected in the crop, treat
               the student as 'distracted' (talking/interacting), not focused.

    BUG B (behavior_detection.py — _filter_head_down_temporal):
      The tracker key is the raw bbox tuple (x1,y1,x2,y2).  YOLO bbox
      coordinates jitter ±2–5 px every frame even for a stationary person,
      so each frame creates a NEW key.  The 30-second timer NEVER accumulates
      and head_down is NEVER reported.
      FIX: snap bbox to a 20-px grid before using as key, absorbing jitter.

    BUG C (behavior_detection.py — _filter_head_down_temporal):
      When a student briefly lifts their head (1 frame focused) and looks
      back down, the tracker is NOT reset — it keeps the old first_seen
      timestamp.  This means intermittent head-downers accumulate time even
      when not continuously looking down.
      FIX: store a `last_seen` timestamp; reset first_seen if gap > 3 seconds.

    BUG D (behavior_detection.py — detect, phone detection):
      YOLO phone confidence threshold is 0.50 but YOLO11s-style models
      often output phone detections at 0.35-0.45 when the phone is partially
      occluded (held in lap, face down).  This silently drops many phone events.
      FIX: lower phone and food confidence thresholds to 0.35.

    BUG E (behavior_detection.py — detect, object-person overlap):
      _overlaps() does a full bounding-box intersection check but a phone on a
      desk between two students can overlap BOTH person boxes, labelling the
      wrong person as 'using_phone'.
      FIX: pick the person whose box centre is CLOSEST to the object centre.

    BUG F (behavior_detection.py — _detection_loop, alert key):
      The cooldown key is `det['type']` (e.g. 'using_phone').  Multiple
      students using phones simultaneously share the same key, so only the
      first one is ever reported.
      FIX: use (det['type'], bbox_tuple) as the cooldown key so each person
      has their own independent cooldown.
    ─────────────────────────────────────────────────────────────────────────
    """

    YAW_THRESH   = 20
    PITCH_THRESH = -18

    def __init__(self, camera_url, camera_id,
                 server_url='http://localhost:8000',
                 alert_cooldown=120,
                 whatsapp_admin=None):
        self.camera_url     = camera_url
        self.camera_id      = camera_id
        self.server_url     = server_url
        self.alert_cooldown = alert_cooldown
        self.whatsapp_admin = whatsapp_admin or os.environ.get('ADMIN_WHATSAPP', '')

        self.yolo_model      = None
        self.face_recognizer = None
        self.known_students  = []

        self._frontal_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self._frontal_alt     = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self._profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml')
        print('[OK] Haar cascades loaded (frontal + profile)')

        self._prev_gray = None

        # BUG B FIX: head_down tracker — bbox snapped to grid before use
        self.head_down_tracker: dict = {}
        self.HEAD_DOWN_MIN_SECONDS = 30
        self.HEAD_DOWN_GRID        = 20   # snap bbox coords to this grid (px)
        self.HEAD_DOWN_GAP_RESET   = 3.0  # seconds gap before resetting timer

        self.last_alert_time: dict = defaultdict(float)
        self.running = False
        self.thread  = None

        self._load_models()

    # ─────────────────────────────────────────────────────────────────────────
    def _load_models(self):
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolo11s.pt')
            print('[OK] YOLO loaded')
        except Exception as e:
            print(f'[WARN] YOLO: {e}')

        self.fight_detector = FightDetector(fps=30)
        print('[OK] Fight detector initialized')

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
    # Head-pose classification
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_head_pose(self, gray_frame, x1, y1, x2, y2):
        """
        Returns 'focused' | 'looking_away' | 'head_down' | 'distracted'

        BUG A FIX:
          • Narrow crop horizontally to avoid picking up neighbour's face.
          • Raise minNeighbors to 4 to reduce spurious frontal detections.
          • If 2+ frontal faces appear in the crop → 'distracted' (talking).
        """
        h, w = gray_frame.shape

        # ── BUG A FIX: narrow horizontal crop to ±40 % of person width ──────
        pw      = x2 - x1
        margin  = int(pw * 0.40)
        cx      = (x1 + x2) // 2
        nx1     = max(0,  cx - margin)
        nx2     = min(w,  cx + margin)
        py1     = max(0,  y1)
        py2     = min(h,  y2)
        crop    = gray_frame[py1:py2, nx1:nx2]

        if crop.size == 0:
            return 'not_visible'

        # Scale down for speed
        scale = 1.0
        ch, cw = crop.shape
        if cw > 160:
            scale = 160 / cw
            crop  = cv2.resize(crop, (int(cw * scale), int(ch * scale)))

        def _detect_faces(cascade, img, min_neighbors=4):
            """Returns list of detected face rects."""
            with CV2_CASCADE_LOCK:
                faces = cascade.detectMultiScale(
                    img,
                    scaleFactor=1.15,
                    minNeighbors=min_neighbors,  # BUG A FIX: raised from 3→4
                    minSize=(18, 18),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
            if len(faces) == 0:
                return []
            return list(faces)

        # 1. Check frontal faces
        faces_default = _detect_faces(self._frontal_cascade, crop)
        faces_alt     = _detect_faces(self._frontal_alt,     crop)
        all_frontal   = faces_default + faces_alt

        if len(all_frontal) > 0:
            # BUG A FIX: 2+ frontal detections → students talking to each other
            if len(all_frontal) >= 2:
                return 'distracted'
            return 'focused'

        # 2. Profile (left-facing then right-facing)
        if len(_detect_faces(self._profile_cascade, crop)) > 0:
            return 'looking_away'
        if len(_detect_faces(self._profile_cascade, cv2.flip(crop, 1))) > 0:
            return 'looking_away'

        # 3. Neither → head down
        return 'head_down'

    # ─────────────────────────────────────────────────────────────────────────
    # Motion check
    # ─────────────────────────────────────────────────────────────────────────

    def _motion_in_zone(self, gray_frame, ix1, iy1, ix2, iy2):
        if self._prev_gray is None:
            return False
        try:
            h, w = gray_frame.shape
            zx1, zy1 = max(0, ix1), max(0, iy1)
            zx2, zy2 = min(w, ix2), min(h, iy2)
            if zx2 <= zx1 or zy2 <= zy1:
                return False
            curr = gray_frame[zy1:zy2, zx1:zx2].astype(np.int16)
            prev = self._prev_gray[zy1:zy2, zx1:zx2].astype(np.int16)
            if curr.shape != prev.shape:
                return False
            return int(np.sum(np.abs(curr - prev))) > 800
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _overlaps(self, pb, ob):
        """Axis-aligned bounding-box overlap."""
        return not (ob[2] < pb[0] or ob[0] > pb[2] or
                    ob[3] < pb[1] or ob[1] > pb[3])

    def _object_centre(self, box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def _person_centre(self, pb):
        return ((pb[0] + pb[2]) / 2.0, (pb[1] + pb[3]) / 2.0)

    def _dist_sq(self, a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    # ─────────────────────────────────────────────────────────────────────────
    # Main detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect(self, frame):
        """
        Analyse a BGR frame. Returns list of detection dicts.
        """
        if self.yolo_model is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            results = self.yolo_model.predict(frame, verbose=False, conf=0.35)
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
                # BUG D FIX: lower phone / food confidence threshold 0.50 → 0.35
                elif cid == 67 and conf > 0.35:
                    object_boxes.append((x1, y1, x2, y2, 'using_phone'))
                elif cid in range(46, 56) and conf > 0.35:
                    object_boxes.append((x1, y1, x2, y2, 'eating_food'))

        # ── BUG E FIX: assign each object to the NEAREST overlapping person ──
        # Build a mapping: person_index → object_type (highest-priority / nearest)
        person_object: dict[int, str] = {}
        for ob in object_boxes:
            ob_centre = self._object_centre(ob)
            best_dist = float('inf')
            best_idx  = -1
            for idx, (x1, y1, x2, y2, _conf) in enumerate(person_boxes):
                if self._overlaps((x1, y1, x2, y2), ob):
                    d = self._dist_sq(ob_centre, self._person_centre((x1, y1, x2, y2)))
                    if d < best_dist:
                        best_dist = d
                        best_idx  = idx
            if best_idx >= 0:
                # 'using_phone' takes priority over 'eating_food'
                existing = person_object.get(best_idx)
                if existing != 'using_phone':
                    person_object[best_idx] = ob[4]

        # ── Pose-based phone detection (lap/partial occlusion fallback) ───────
        # YOLO frequently misses phones held in the lap or face-down because the
        # object is partially occluded.  Secondary signal: a student whose head
        # is tilted DOWN (chin toward chest, looking at lap) AND has motion in
        # the lower-centre hand region is very likely using a phone.
        # This fires only when YOLO has NOT already assigned 'using_phone'.
        if self._prev_gray is not None:
            for idx, (x1, y1, x2, y2, _conf) in enumerate(person_boxes):
                if person_object.get(idx) == 'using_phone':
                    continue   # already detected by YOLO, skip
                pose = self._classify_head_pose(gray, x1, y1, x2, y2)
                if pose not in ('head_down',):
                    continue   # head must be tilted down
                # Check for motion in the lap zone: bottom-centre 30 % of box
                bw = x2 - x1
                bh = y2 - y1
                lap_x1 = x1 + int(bw * 0.25)
                lap_x2 = x1 + int(bw * 0.75)
                lap_y1 = y1 + int(bh * 0.65)
                lap_y2 = y2
                if self._motion_in_zone(gray, lap_x1, lap_y1, lap_x2, lap_y2):
                    person_object[idx] = 'using_phone'

        # ── Classify each person ──────────────────────────────────────────────
        detections = []
        for i, (x1, y1, x2, y2, conf) in enumerate(person_boxes):

            obj_type = person_object.get(i)
            if obj_type:
                det_type = obj_type
            else:
                det_type = self._classify_head_pose(gray, x1, y1, x2, y2)

            if det_type == 'head_down':
                det_type = self._filter_head_down_temporal((x1, y1, x2, y2))
                if det_type is None:
                    det_type = 'focused'

            detections.append({
                'type':          det_type,
                'bbox':          (x1, y1, x2, y2),
                'confidence':    conf,
                'color':         COLOR_MAP.get(det_type, (120, 120, 120)),
                'label':         LABEL_MAP.get(det_type, det_type),
                'is_alert':      det_type in ALERT_POSES,
                'is_distracted': det_type in DISTRACTED_POSES,
            })

        # ── Fight detection ───────────────────────────────────────────────────
        # Filter out abnormally WIDE person boxes before fight detection.
        # When two adjacent students are merged into one YOLO box the aspect
        # ratio (width/height) is >> 1.  A normal standing/sitting person has
        # w/h ≈ 0.35–0.65.  Skip boxes with w/h > 1.0 from fight input so the
        # tracker doesn't pair them with their own "other half".
        fight_input_boxes = [
            b for b in person_boxes
            if (b[2] - b[0]) / max(1, b[3] - b[1]) < 1.0
        ]
        if self.fight_detector and len(fight_input_boxes) > 1:
            fight_interactions = self.fight_detector.process_frame(frame, fight_input_boxes)
            for fight_info in fight_interactions:
                bbox_a = self.fight_detector.get_track_bbox(fight_info['person_a_id'])
                bbox_b = self.fight_detector.get_track_bbox(fight_info['person_b_id'])

                if bbox_a and bbox_b:
                    for fight_bbox in [bbox_a, bbox_b]:
                        # Upgrade existing detection to fighting if bbox matches,
                        # otherwise append new entry.
                        upgraded = False
                        for det in detections:
                            if det['bbox'] == fight_bbox:
                                det['type']          = 'fighting'
                                det['color']         = COLOR_MAP['fighting']
                                det['label']         = LABEL_MAP['fighting']
                                det['is_alert']      = True
                                det['is_distracted'] = False
                                det['fight_info']    = fight_info
                                det['confidence']    = fight_info['confidence']
                                upgraded = True
                                break
                        if not upgraded:
                            detections.append({
                                'type':          'fighting',
                                'bbox':          fight_bbox,
                                'confidence':    fight_info['confidence'],
                                'color':         COLOR_MAP['fighting'],
                                'label':         LABEL_MAP['fighting'],
                                'is_alert':      True,
                                'is_distracted': False,
                                'fight_info':    fight_info,
                            })
        else:
            if self.fight_detector:
                self.fight_detector.process_frame(frame, [])

        self._prev_gray = gray.copy()
        return detections

    def _detect_behaviors(self, frame):
        return self.detect(frame)

    # ─────────────────────────────────────────────────────────────────────────
    # Head-down temporal filtering
    # ─────────────────────────────────────────────────────────────────────────

    def _snap_bbox(self, bbox):
        """
        BUG B FIX: snap all coordinates to a grid to absorb YOLO jitter.
        Two detections within HEAD_DOWN_GRID pixels of each other map to the
        same key, so the 30-second timer accumulates correctly.
        """
        g = self.HEAD_DOWN_GRID
        return tuple(round(v / g) * g for v in bbox)

    def _filter_head_down_temporal(self, bbox):
        """
        Only report head_down if continuous for HEAD_DOWN_MIN_SECONDS.

        BUG B FIX: use snapped bbox as key so YOLO jitter doesn't break accumulation.
        BUG C FIX: reset first_seen when gap since last observation exceeds 3 seconds.
        """
        bbox_key = self._snap_bbox(bbox)
        now = time.time()

        if bbox_key in self.head_down_tracker:
            info = self.head_down_tracker[bbox_key]

            # BUG C FIX: check for gap (student briefly looked up)
            gap = now - info['last_seen']
            if gap > self.HEAD_DOWN_GAP_RESET:
                # Reset: they looked up and back down — start fresh
                self.head_down_tracker[bbox_key] = {
                    'first_seen': now,
                    'last_seen':  now,
                }
                return None

            # Update last-seen timestamp
            info['last_seen'] = now
            elapsed = now - info['first_seen']

            if elapsed >= self.HEAD_DOWN_MIN_SECONDS:
                return 'head_down'
            return None
        else:
            self.head_down_tracker[bbox_key] = {
                'first_seen': now,
                'last_seen':  now,
            }
            return None

    def _cleanup_head_down_tracker(self):
        """Remove stale entries (no update in 5 minutes)."""
        now = time.time()
        stale = [k for k, v in self.head_down_tracker.items()
                 if now - v['last_seen'] > 300]
        for k in stale:
            del self.head_down_tracker[k]

    # ─────────────────────────────────────────────────────────────────────────
    # Draw detections
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_detections(self, frame, detections):
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = det['color']
            label = det['label']
            conf  = det['confidence']

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text      = f"{label} ({conf:.2f})"
            font      = cv2.FONT_HERSHEY_SIMPLEX
            fscale    = 0.5
            thickness = 1
            tw, th    = cv2.getTextSize(text, font, fscale, thickness)[0]
            bg_y1     = max(0, y1 - th - 4)
            cv2.rectangle(annotated, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 2, y1 - 2),
                        font, fscale, (255, 255, 255), thickness)
        return annotated

    # ─────────────────────────────────────────────────────────────────────────
    # Report to Django API
    # ─────────────────────────────────────────────────────────────────────────

    def _report_incident(self, detection, frame, student_id, student_name, roll_no,
                         all_detections=None):
        import requests as _req
        try:
            annotated = self._draw_detections(frame, all_detections) if all_detections else frame
            _, buf    = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            snap_b64  = base64.b64encode(buf).decode()

            if detection['type'] == 'fighting':
                fight_info = detection.get('fight_info', {})
                other_id   = fight_info.get('person_b_id', 'unknown')
                tag        = f'{student_name} ({roll_no}) vs student_{other_id}'
                severity   = '🚨 CRITICAL'
            else:
                tag      = f'{student_name} ({roll_no})' if student_id else 'Unknown person'
                severity = '⚠️'

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
                    'description':   f"{severity} {detection['label']} — {tag}",
                    'send_whatsapp': detection['is_alert'],
                },
                timeout=10, verify=False,
            )
            print(f"[INCIDENT] {detection['label']} | {tag} | http={resp.status_code}")
        except Exception as e:
            print(f'[ERROR] report_incident: {e}')

    # ─────────────────────────────────────────────────────────────────────────
    # Face recognition
    # ─────────────────────────────────────────────────────────────────────────

    def _recognize_face(self, frame, bbox):
        """Returns (student_id, name, roll_no) or (None, 'Unknown', '')."""
        if self.face_recognizer is None or not self.known_students:
            return None, 'Unknown', ''
        try:
            x1, y1, x2, y2 = bbox
            mid_y = y1 + int((y2 - y1) * 0.55)
            crop  = frame[y1:mid_y, x1:x2]
            if crop.size == 0:
                crop = frame[y1:y2, x1:x2]
            rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            encs = self.face_recognizer.face_encodings(rgb, num_jitters=1, model='small')
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
    # Background loop
    # ─────────────────────────────────────────────────────────────────────────

    def _detection_loop(self):
        cap = cv2.VideoCapture(self.camera_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f'[FATAL] Cannot open: {self.camera_url}')
            return
        print('[OK] Camera opened')
        frame_count  = 0
        fight_cooldown: dict = defaultdict(float)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print('[WARN] Frame read failed, reconnecting…')
                time.sleep(2)
                cap = cv2.VideoCapture(self.camera_url)
                continue

            frame_count += 1
            detections = self.detect(frame)

            fight_detections = [d for d in detections if d['type'] == 'fighting']
            other_detections = [d for d in detections if d['type'] != 'fighting']

            # ── Non-fight incidents ───────────────────────────────────────────
            for det in other_detections:
                if not (det['is_alert'] or det['is_distracted']):
                    continue
                now = time.time()
                # BUG F FIX: per-person cooldown key includes bbox
                key = (det['type'], det['bbox'])
                if (now - self.last_alert_time.get(key, 0)) < self.alert_cooldown:
                    continue
                sid, name, roll = self._recognize_face(frame, det['bbox'])
                self._report_incident(det, frame, sid, name, roll,
                                      all_detections=detections)
                self.last_alert_time[key] = now

            # ── Fight incidents ───────────────────────────────────────────────
            for det in fight_detections:
                fight_info = det.get('fight_info', {})
                pair_key   = tuple(sorted([
                    fight_info.get('person_a_id', 0),
                    fight_info.get('person_b_id', 0),
                ]))
                now = time.time()
                if (now - fight_cooldown.get(pair_key, 0)) < 60:
                    continue
                sid, name, roll = self._recognize_face(frame, det['bbox'])
                self._report_incident(det, frame, sid, name, roll,
                                      all_detections=detections)
                fight_cooldown[pair_key] = now

            if frame_count % 300 == 0:
                print(f'[INFO] Frame {frame_count} | {len(detections)} detections')
                self._cleanup_head_down_tracker()

        cap.release()
        print('[OK] Detection stopped')

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread  = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print('[OK] Behavior detection started')

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print('[OK] Behavior detection stopped')