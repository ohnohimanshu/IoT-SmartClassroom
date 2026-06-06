"""
Classroom Behavior Detection
=============================

FIX: Phone detection was firing for students writing in notebooks.

ROOT CAUSE:
  From a high-angle camera, a writing student and a phone-in-lap student
  share almost identical signals:
    • head_down         → both look down
    • lap_motion        → writing hand also creates flow in lower bbox zone
    • arm_inward        → both postures pull arms inward

  The previous PosePhoneDetector required any 2-of-3 of these, which means
  writers constantly triggered it.

CORRECT DISTINGUISHING SIGNALS (writing vs phone):

  1. FLOW ZONE RATIO (most reliable):
     Writing: flow is concentrated in the UPPER-MIDDLE zone of the person
              bbox (hand near desk surface, roughly 30-60% of height).
     Phone:   flow is concentrated in the LOWER zone (lap, 60-85% of height).
     → Compute ratio: lower_flow / (upper_flow + ε)
       If ratio > 1.4 → phone candidate. If ratio < 0.8 → writing.

  2. FLOW VARIANCE (writing is rhythmic, phone is erratic):
     Writing produces consistent periodic flow (pen strokes).
     Scrolling/tapping produces irregular bursts.
     → Track variance of lap-zone flow over last 20 frames.
       High variance with low mean → tapping (phone).
       Low variance with moderate mean → writing.

  3. STATIC HOLD DETECTION:
     Phone users hold the device still for seconds between interactions.
     Writers almost never hold completely still — pen is always moving.
     → If lap_flow < 0.3 px/frame for > 15 consecutive frames while
       head is down → static hold = phone candidate.

  4. YOLO phone (class 67) remains the highest-confidence signal.
     Lowered threshold to 0.28 to catch partially-visible phones.

  DECISION:
    • YOLO phone detected                              → using_phone (high conf)
    • flow_zone_ratio > 1.4 AND (variance_score OR static_hold) → using_phone
    • Otherwise                                        → head_down / focused

  This specifically rejects the writing pattern:
    Writing = upper-zone flow high + lower-zone flow moderate + periodic variance
    Phone   = upper-zone flow low  + lower-zone flow present  + erratic/static
"""

import cv2
import json
import time
import base64
import numpy as np
import threading
import os
from collections import defaultdict, deque

from .fight_detection import FightDetector

# ── Env loading ───────────────────────────────────────────────────────────────
def _load_env_file():
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
                        os.environ.setdefault(k.strip(),
                                              v.strip().strip('"').strip("'"))
            print(f'[ENV] Loaded {path}')
            return
    print('[ENV] No .env file found')

_load_env_file()

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

CV2_CASCADE_LOCK = threading.Lock()

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


# ─────────────────────────────────────────────────────────────────────────────
# Per-person flow history tracker
# ─────────────────────────────────────────────────────────────────────────────

class PersonFlowHistory:
    """
    Maintains a rolling history of flow measurements for one person bbox.
    Keyed by snapped bbox so YOLO jitter doesn't create new entries.
    """
    GRID = 20   # snap px

    def __init__(self):
        # key → dict with deques for upper/mid/lower flow
        self._store: dict = {}

    def _snap(self, bbox):
        g = self.GRID
        return tuple(round(v / g) * g for v in bbox)

    def update(self, bbox, upper_flow: float, mid_flow: float, lower_flow: float):
        key = self._snap(bbox)
        if key not in self._store:
            self._store[key] = {
                'upper':      deque(maxlen=30),
                'mid':        deque(maxlen=30),
                'lower':      deque(maxlen=30),
                'last_seen':  time.time(),
                'still_frames': 0,   # consecutive frames with near-zero lap flow
            }
        s = self._store[key]
        s['upper'].append(upper_flow)
        s['mid'].append(mid_flow)
        s['lower'].append(lower_flow)
        s['last_seen'] = time.time()

        # Track static-hold frames (lap nearly still while head is down)
        if lower_flow < 0.30:
            s['still_frames'] += 1
        else:
            s['still_frames'] = 0

        return key

    def get(self, bbox):
        return self._store.get(self._snap(bbox))

    def cleanup(self):
        now   = time.time()
        stale = [k for k, v in self._store.items()
                 if now - v['last_seen'] > 60]
        for k in stale:
            del self._store[k]


# ─────────────────────────────────────────────────────────────────────────────
# Pose-based phone detector  (rewritten — writing-safe)
# ─────────────────────────────────────────────────────────────────────────────

class PosePhoneDetector:
    """
    Detects phone use when YOLO cannot see the phone directly.

    Only fires when flow pattern matches phone-in-lap, NOT writing.
    Key discriminator: ZONE RATIO (lower vs upper-mid flow).

    Writing  → upper-mid zone flow dominates (hand near desk).
    Phone    → lower zone flow matches or exceeds upper-mid (hand near lap).
    """

    # Zone boundaries (fraction of bbox height)
    UPPER_TOP    = 0.00   # top of head
    UPPER_BOTTOM = 0.45   # desk-level hand zone top boundary
    MID_TOP      = 0.30   # desk-level hand zone
    MID_BOTTOM   = 0.60   # desk surface / lap boundary
    LAP_TOP      = 0.58   # lap zone start
    LAP_BOTTOM   = 0.90   # lap zone end (avoid feet noise)

    # Thresholds
    ZONE_RATIO_THRESH   = 1.4    # lower/mid_flow ratio to flag phone
    VARIANCE_TAP_THRESH = 0.35   # flow variance that indicates tapping
    STATIC_HOLD_FRAMES  = 18     # consecutive near-zero lap frames = holding still
    MIN_LAP_FLOW        = 0.25   # minimum lap flow to even consider (no movement = not phone)

    def __init__(self):
        self._flow  = None
        self._fh    = 0
        self._fw    = 0
        self.history = PersonFlowHistory()

    def set_flow(self, flow_map, fh: int, fw: int):
        self._flow = flow_map
        self._fh   = fh
        self._fw   = fw

    def _zone_flow(self, x1, y1, x2, y2, top_frac, bot_frac) -> float:
        """Mean flow magnitude in a horizontal zone of the person bbox."""
        if self._flow is None:
            return 0.0
        h  = y2 - y1
        zy1 = max(0,        int(y1 + h * top_frac))
        zy2 = min(self._fh, int(y1 + h * bot_frac))
        # Use centre 70% of width to avoid background on sides
        w   = x2 - x1
        zx1 = max(0,        int(x1 + w * 0.15))
        zx2 = min(self._fw, int(x2 - w * 0.15))
        if zy2 <= zy1 or zx2 <= zx1:
            return 0.0
        region = self._flow[zy1:zy2, zx1:zx2]
        if region.size == 0:
            return 0.0
        return float(np.mean(np.sqrt(region[..., 0]**2 + region[..., 1]**2)))

    def analyze(self, bbox, head_pose: str):
        """
        Returns (is_phone: bool, confidence: float, reason: str)

        Must only be called when head_pose == 'head_down'.
        """
        x1, y1, x2, y2 = bbox

        # Compute zone flows
        upper_flow = self._zone_flow(x1, y1, x2, y2,
                                      self.UPPER_TOP, self.UPPER_BOTTOM)
        mid_flow   = self._zone_flow(x1, y1, x2, y2,
                                      self.MID_TOP, self.MID_BOTTOM)
        lap_flow   = self._zone_flow(x1, y1, x2, y2,
                                      self.LAP_TOP, self.LAP_BOTTOM)

        # Update rolling history
        key = self.history.update(bbox, upper_flow, mid_flow, lap_flow)
        info = self.history.get(bbox)

        signals = []

        # ── Signal 1: ZONE RATIO ──────────────────────────────────────────────
        # Phone → lap_flow / mid_flow > ZONE_RATIO_THRESH
        # Writing → mid_flow >= lap_flow (hand near desk, not lap)
        mid_ref   = mid_flow + 1e-4
        zone_ratio = lap_flow / mid_ref

        # ALSO check that mid (writing zone) is NOT dominant
        # If mid_flow is high → definitely writing, veto signal 1
        writing_dominant = mid_flow > 1.2 and mid_flow > lap_flow * 1.2

        if zone_ratio >= self.ZONE_RATIO_THRESH and not writing_dominant:
            signals.append(('zone_ratio', 0.50))

        # ── Signal 2: TAP VARIANCE in lap zone ───────────────────────────────
        # Scrolling/tapping: low mean + high variance (irregular bursts)
        # Writing: moderate mean + LOW variance (periodic strokes)
        if info and len(info['lower']) >= 10:
            lap_vals     = list(info['lower'])
            lap_mean     = float(np.mean(lap_vals))
            lap_variance = float(np.var(lap_vals))
            # Phone tap signature: variance high relative to mean
            tap_score = lap_variance / (lap_mean + 0.1)
            if tap_score > self.VARIANCE_TAP_THRESH and lap_mean > self.MIN_LAP_FLOW:
                signals.append(('tap_variance', 0.30))

        # ── Signal 3: STATIC HOLD ─────────────────────────────────────────────
        # Person holds phone still for several seconds between interactions.
        # Writers almost never go completely still.
        # BUT: only fire this if lap_flow is also non-zero (not just sitting).
        if (info and
                info['still_frames'] >= self.STATIC_HOLD_FRAMES and
                lap_flow > 0.10):
            # Extra guard: make sure upper flow is also low
            # (a still head-down student who stopped writing briefly
            #  would have no upper flow either — but then zone_ratio
            #  would also be near 1.0 and signal 1 would not have fired)
            if upper_flow < 0.6:
                signals.append(('static_hold', 0.20))

        # ── Decision ──────────────────────────────────────────────────────────
        total_weight = sum(w for _, w in signals)

        # Require zone_ratio signal PLUS at least one corroborating signal,
        # OR zone_ratio alone if very strong (ratio > 2.5)
        has_zone_ratio = any(n == 'zone_ratio' for n, _ in signals)
        strong_ratio   = zone_ratio > 2.5 and not writing_dominant

        is_phone = (has_zone_ratio and total_weight >= 0.70) or strong_ratio

        confidence = min(0.82, total_weight) if is_phone else 0.0
        reason     = '+'.join(n for n, _ in signals) if signals else 'none'

        return is_phone, confidence, reason

    def cleanup(self):
        self.history.cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# Main detector
# ─────────────────────────────────────────────────────────────────────────────

class ClassroomBehaviorDetector:

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
        print('[OK] Haar cascades loaded')

        self._prev_gray = None
        self._flow      = None
        self._fh        = 0
        self._fw        = 0

        # Head-down temporal filter
        self.head_down_tracker:    dict = {}
        self.HEAD_DOWN_MIN_SECONDS = 30
        self.HEAD_DOWN_GRID        = 20
        self.HEAD_DOWN_GAP_RESET   = 3.0

        self.last_alert_time: dict = defaultdict(float)
        self.running  = False
        self.thread   = None

        self.phone_detector = PosePhoneDetector()
        self._load_models()

    # ── Model loading ─────────────────────────────────────────────────────────

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
            r = _req.get(f'{self.server_url}/api/students/encodings/',
                         timeout=5, verify=False)
            self.known_students = r.json()
            print(f'[OK] {len(self.known_students)} encodings')
        except Exception as e:
            print(f'[WARN] students: {e}')

    # ── Optical flow ──────────────────────────────────────────────────────────

    def _update_flow(self, gray):
        if (self._prev_gray is not None and
                self._prev_gray.shape == gray.shape):
            try:
                self._flow = cv2.calcOpticalFlowFarneback(
                    self._prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
            except Exception:
                self._flow = None
        else:
            self._flow = None
        self._fh, self._fw = gray.shape

    # ── Head-pose classification ──────────────────────────────────────────────

    def _classify_head_pose(self, gray_frame, x1, y1, x2, y2) -> str:
        h, w   = gray_frame.shape
        pw     = x2 - x1
        margin = int(pw * 0.40)
        cx     = (x1 + x2) // 2
        nx1    = max(0, cx - margin)
        nx2    = min(w, cx + margin)
        crop   = gray_frame[max(0, y1):min(h, y2), nx1:nx2]

        if crop.size == 0:
            return 'not_visible'

        cw = crop.shape[1]
        if cw > 160:
            scale = 160 / cw
            crop  = cv2.resize(crop,
                               (int(cw * scale),
                                int(crop.shape[0] * scale)))

        def _det(cascade, img, mn=4):
            with CV2_CASCADE_LOCK:
                faces = cascade.detectMultiScale(
                    img, scaleFactor=1.15, minNeighbors=mn,
                    minSize=(18, 18), flags=cv2.CASCADE_SCALE_IMAGE)
            return [] if len(faces) == 0 else list(faces)

        frontal = _det(self._frontal_cascade, crop) + \
                  _det(self._frontal_alt,     crop)
        if frontal:
            return 'distracted' if len(frontal) >= 2 else 'focused'

        if (_det(self._profile_cascade, crop) or
                _det(self._profile_cascade, cv2.flip(crop, 1))):
            return 'looking_away'

        return 'head_down'

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _overlaps(self, pb, ob):
        return not (ob[2] < pb[0] or ob[0] > pb[2] or
                    ob[3] < pb[1] or ob[1] > pb[3])

    def _centre(self, b):
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def _dist_sq(self, a, b):
        return (a[0] - b[0])**2 + (a[1] - b[1])**2

    # ── Head-down temporal filter ─────────────────────────────────────────────

    def _snap_bbox(self, bbox):
        g = self.HEAD_DOWN_GRID
        return tuple(round(v / g) * g for v in bbox)

    def _filter_head_down_temporal(self, bbox):
        key = self._snap_bbox(bbox)
        now = time.time()
        if key in self.head_down_tracker:
            info = self.head_down_tracker[key]
            if now - info['last_seen'] > self.HEAD_DOWN_GAP_RESET:
                self.head_down_tracker[key] = {'first_seen': now,
                                               'last_seen':  now}
                return None
            info['last_seen'] = now
            if now - info['first_seen'] >= self.HEAD_DOWN_MIN_SECONDS:
                return 'head_down'
            return None
        self.head_down_tracker[key] = {'first_seen': now, 'last_seen': now}
        return None

    def _cleanup_head_down_tracker(self):
        now   = time.time()
        stale = [k for k, v in self.head_down_tracker.items()
                 if now - v['last_seen'] > 300]
        for k in stale:
            del self.head_down_tracker[k]

    # ── Main detection ────────────────────────────────────────────────────────

    def detect(self, frame) -> list:
        if self.yolo_model is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._update_flow(gray)
        self.phone_detector.set_flow(self._flow, self._fh, self._fw)

        # ── YOLO ─────────────────────────────────────────────────────────────
        try:
            results = self.yolo_model.predict(frame, verbose=False, conf=0.30)
        except Exception as e:
            print(f'[ERROR] YOLO: {e}')
            return []

        person_boxes = []
        object_boxes = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cid  = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cid == 0:
                    person_boxes.append((x1, y1, x2, y2, conf))
                elif cid == 67 and conf > 0.28:        # phone — low threshold
                    object_boxes.append((x1, y1, x2, y2, 'using_phone'))
                elif cid in range(46, 56) and conf > 0.30:
                    object_boxes.append((x1, y1, x2, y2, 'eating_food'))

        # ── Assign YOLO objects → nearest overlapping person ─────────────────
        person_object: dict = {}
        for ob in object_boxes:
            ob_c = self._centre(ob)
            best_d, best_i = float('inf'), -1
            for i, (x1, y1, x2, y2, _) in enumerate(person_boxes):
                if self._overlaps((x1, y1, x2, y2), ob):
                    d = self._dist_sq(ob_c, self._centre((x1, y1, x2, y2)))
                    if d < best_d:
                        best_d, best_i = d, i
            if best_i >= 0:
                existing = person_object.get(best_i)
                if existing is None or existing != 'using_phone':
                    person_object[best_i] = ob[4]

        # ── Classify each person ──────────────────────────────────────────────
        detections = []
        for i, (x1, y1, x2, y2, conf) in enumerate(person_boxes):

            yolo_obj = person_object.get(i)
            if yolo_obj:
                # YOLO directly saw the phone/food
                det_type = yolo_obj
                det_conf = conf
            else:
                head_pose = self._classify_head_pose(gray, x1, y1, x2, y2)

                if head_pose == 'head_down':
                    # ── POSE-BASED PHONE CHECK ────────────────────────────────
                    # Only fires when flow pattern matches phone, not writing.
                    is_phone, phone_conf, reason = self.phone_detector.analyze(
                        (x1, y1, x2, y2), head_pose
                    )
                    if is_phone:
                        det_type = 'using_phone'
                        det_conf = phone_conf
                        print(f'[PHONE] pose @ ({x1},{y1}) signals={reason}')
                    else:
                        # Temporal gate: only label head_down after 30 s
                        filtered = self._filter_head_down_temporal(
                            (x1, y1, x2, y2))
                        det_type = filtered if filtered else 'focused'
                        det_conf = conf
                else:
                    det_type = head_pose
                    det_conf = conf

            detections.append({
                'type':          det_type,
                'bbox':          (x1, y1, x2, y2),
                'confidence':    det_conf,
                'color':         COLOR_MAP.get(det_type, (120, 120, 120)),
                'label':         LABEL_MAP.get(det_type, det_type),
                'is_alert':      det_type in ALERT_POSES,
                'is_distracted': det_type in DISTRACTED_POSES,
            })

        # ── Fight detection ───────────────────────────────────────────────────
        fight_input = [
            b for b in person_boxes
            if (b[2] - b[0]) / max(1, b[3] - b[1]) < 1.0
        ]

        if self.fight_detector and len(fight_input) > 1:
            fight_interactions = self.fight_detector.process_frame(
                frame, fight_input)

            for fi in fight_interactions:
                bbox_a = self.fight_detector.get_track_bbox(fi['person_a_id'])
                bbox_b = self.fight_detector.get_track_bbox(fi['person_b_id'])
                if not (bbox_a and bbox_b):
                    continue
                for fight_bbox in [bbox_a, bbox_b]:
                    upgraded = False
                    for det in detections:
                        if det['bbox'] == fight_bbox:
                            det.update({
                                'type':          'fighting',
                                'color':         COLOR_MAP['fighting'],
                                'label':         LABEL_MAP['fighting'],
                                'is_alert':      True,
                                'is_distracted': False,
                                'fight_info':    fi,
                                'confidence':    fi['confidence'],
                            })
                            upgraded = True
                            break
                    if not upgraded:
                        detections.append({
                            'type':          'fighting',
                            'bbox':          fight_bbox,
                            'confidence':    fi['confidence'],
                            'color':         COLOR_MAP['fighting'],
                            'label':         LABEL_MAP['fighting'],
                            'is_alert':      True,
                            'is_distracted': False,
                            'fight_info':    fi,
                        })
        else:
            if self.fight_detector:
                self.fight_detector.process_frame(frame, [])

        self._prev_gray = gray.copy()
        return detections

    def _detect_behaviors(self, frame):
        return self.detect(frame)

    # ── Draw ──────────────────────────────────────────────────────────────────

    def _draw_detections(self, frame, detections):
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = det['color']
            label = det['label']
            conf  = det['confidence']
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"{label} ({conf:.2f})"
            tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            bg_y1  = max(0, y1 - th - 4)
            cv2.rectangle(out, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, text, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out

    # ── Report ────────────────────────────────────────────────────────────────

    def _report_incident(self, detection, frame, student_id,
                          student_name, roll_no, all_detections=None):
        import requests as _req
        try:
            annotated = (self._draw_detections(frame, all_detections)
                         if all_detections else frame)
            _, buf    = cv2.imencode('.jpg', annotated,
                                     [cv2.IMWRITE_JPEG_QUALITY, 80])
            snap_b64  = base64.b64encode(buf).decode()

            if detection['type'] == 'fighting':
                fi    = detection.get('fight_info', {})
                other = fi.get('person_b_id', 'unknown')
                tag   = f'{student_name} ({roll_no}) vs student_{other}'
                sev   = '🚨 CRITICAL'
            else:
                tag = (f'{student_name} ({roll_no})'
                       if student_id else 'Unknown')
                sev = '⚠️'

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
                    'description':   f"{sev} {detection['label']} — {tag}",
                    'send_whatsapp': detection['is_alert'],
                },
                timeout=10, verify=False,
            )
            print(f"[INCIDENT] {detection['label']} | {tag} | {resp.status_code}")
        except Exception as e:
            print(f'[ERROR] report_incident: {e}')

    # ── Face recognition ──────────────────────────────────────────────────────

    def _recognize_face(self, frame, bbox):
        if self.face_recognizer is None or not self.known_students:
            return None, 'Unknown', ''
        try:
            x1, y1, x2, y2 = bbox
            mid_y = y1 + int((y2 - y1) * 0.55)
            crop  = frame[y1:mid_y, x1:x2]
            if crop.size == 0:
                crop = frame[y1:y2, x1:x2]
            rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            encs = self.face_recognizer.face_encodings(
                rgb, num_jitters=1, model='small')
            if not encs:
                return None, 'Unknown', ''
            det    = encs[0]
            best_d = 1.0
            best   = None
            for s in self.known_students:
                try:
                    known = np.array(json.loads(s['encoding']))
                    d     = self.face_recognizer.face_distance([known], det)[0]
                    if d < best_d:
                        best_d, best = d, s
                except Exception:
                    continue
            if best_d < 0.55 and best:
                return best['id'], best['name'], best.get('roll_no', '')
            return None, 'Unknown', ''
        except Exception as e:
            print(f'[ERROR] face_recog: {e}')
            return None, 'Unknown', ''

    # ── Detection loop ────────────────────────────────────────────────────────

    def _detection_loop(self):
        cap = cv2.VideoCapture(self.camera_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f'[FATAL] Cannot open: {self.camera_url}')
            return
        print('[OK] Camera opened')
        frame_count    = 0
        fight_cooldown: dict = defaultdict(float)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print('[WARN] Frame read fail, reconnecting…')
                time.sleep(2)
                cap = cv2.VideoCapture(self.camera_url)
                continue

            frame_count += 1
            detections  = self.detect(frame)

            fight_dets = [d for d in detections if d['type'] == 'fighting']
            other_dets = [d for d in detections if d['type'] != 'fighting']

            for det in other_dets:
                if not (det['is_alert'] or det['is_distracted']):
                    continue
                now = time.time()
                key = (det['type'], det['bbox'])
                if now - self.last_alert_time.get(key, 0) < self.alert_cooldown:
                    continue
                sid, name, roll = self._recognize_face(frame, det['bbox'])
                self._report_incident(det, frame, sid, name, roll,
                                      all_detections=detections)
                self.last_alert_time[key] = now

            for det in fight_dets:
                fi       = det.get('fight_info', {})
                pair_key = tuple(sorted([fi.get('person_a_id', 0),
                                         fi.get('person_b_id', 0)]))
                now = time.time()
                if now - fight_cooldown.get(pair_key, 0) < 60:
                    continue
                sid, name, roll = self._recognize_face(frame, det['bbox'])
                self._report_incident(det, frame, sid, name, roll,
                                      all_detections=detections)
                fight_cooldown[pair_key] = now

            if frame_count % 300 == 0:
                print(f'[INFO] Frame {frame_count} | {len(detections)} dets')
                self._cleanup_head_down_tracker()
                self.phone_detector.cleanup()

        cap.release()
        print('[OK] Detection stopped')

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread  = threading.Thread(target=self._detection_loop,
                                        daemon=True)
        self.thread.start()
        print('[OK] Behavior detection started')

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print('[OK] Behavior detection stopped')