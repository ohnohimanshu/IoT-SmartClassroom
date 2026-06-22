import cv2
import json
import time
import base64
import numpy as np
import threading
import os
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


from classroom_monitor.constants import (
    COLOR_MAP, LABEL_MAP, ALERT_POSES, DISTRACTED_POSES, FACE_MATCH_TOLERANCE,
)


def _http_verify_ssl() -> bool:
    """Use HTTP_VERIFY_SSL=false only for dev/self-signed camera certs."""
    return os.environ.get('HTTP_VERIFY_SSL', 'true').strip().lower() not in (
        'false', '0', 'no',
    )


# ── Environment Loading ──────────────────────────────────────────────────────
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
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            print(f'[ENV] Loaded {path}')
            return
    print('[ENV] No .env file found')

_load_env_file()

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')


# ── Data Structures ──────────────────────────────────────────────────────────
@dataclass
class TrackedPerson:
    track_id: int
    bbox: Tuple[int, int, int, int]
    keypoints: Optional[np.ndarray] = None
    behavior_history: deque = field(default_factory=lambda: deque(maxlen=16))
    last_seen: float = 0.0
    last_final_behavior: str = 'focused'
    keypoint_history: deque = field(default_factory=lambda: deque(maxlen=20))
    last_raw_confidence: float = 0.0


@dataclass
class DetectionResult:
    type: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    color: Tuple[int, int, int]
    label: str
    is_alert: bool
    is_distracted: bool
    track_id: Optional[int] = None
    fight_info: Optional[Dict] = None


# ── Temporal Behavior Engine ─────────────────────────────────────────────────
class TemporalBehaviorEngine:
    # Phone detection
    PHONE_LAP_HEIGHT_FRACTION    = 0.60   # hands below 60% of person height = lap/desk level
    LOW_CONFIDENCE_THRESHOLD     = 0.5
    HEAD_DOWN_CONSECUTIVE_FRAMES = 3      # ~0.3 s at 10 FPS to detect head-down

    # Eating detection
    EATING_HAND_TO_MOUTH_THRESHOLD = 0.15

    # Fight detection — skeleton heuristic (always available, no CNN needed)
    FIGHT_PROXIMITY_RATIO       = 1.5    # candidate gate: centers within 1.5x scale_ref
    FIGHT_WRIST_CROSS_RATIO     = 0.25   # wrist-to-wrist proximity (strong contact signal)
    FIGHT_LIMB_CROSS_RATIO      = 0.32   # elbow/shoulder proximity during grappling
    FIGHT_SKELETON_CONFIRMS     = 4      # sustained contact+motion before flagging
    FIGHT_CNN_CONFIRMS          = 2      # CNN confirmations before flagging
    FIGHT_SCORE_THRESHOLD       = 0.72   # min skeleton score to count as a confirmation tick
    FIGHT_CONFIRM_WINDOW_SEC    = 1.5    # grappling/contact needs slightly longer window

    # Alert smoothing
    ALERT_CONFIRM_FRAMES  = 2            # phone/fight: 2-in-a-row confirms
    NORMAL_CONFIRM_FRAMES = 3            # focused/distracted: 3-in-a-row

    def __init__(self):
        self.tracked_people: Dict[int, TrackedPerson] = {}
        self.cleanup_threshold = 2.0
        self.low_confidence_counters: Dict[int, int] = {}

        # Pairwise fight state
        self.fight_pairs: Dict[Tuple[int, int], Dict] = {}
        self.fight_pair_detectors: Dict[Tuple[int, int], object] = {}
        self._cnn_available = self._check_cnn_available()

    @staticmethod
    def _check_cnn_available() -> bool:
        try:
            from classroom_monitor.fight_detection_3dcnn import FightDetector3DCNN
            FightDetector3DCNN(device='auto', sequence_length=16)
            return True
        except Exception as e:
            print(f'[BEHAVIOR] 3D CNN unavailable ({e}), skeleton-only fight detection')
            return False

    def update_person(self, track_id: int, bbox: Tuple, keypoints: Optional[np.ndarray], timestamp: float):
        if track_id not in self.tracked_people:
            p = TrackedPerson(
                track_id=track_id, bbox=bbox, keypoints=keypoints, last_seen=timestamp)
            self.tracked_people[track_id] = p
        else:
            p = self.tracked_people[track_id]
            p.bbox, p.keypoints, p.last_seen = bbox, keypoints, timestamp

        if keypoints is not None and keypoints.size > 0:
            p.keypoint_history.append((timestamp, keypoints.copy()))

    def cleanup_stale(self, current_time: float):
        stale = [tid for tid, p in self.tracked_people.items()
                 if current_time - p.last_seen > self.cleanup_threshold]
        for tid in stale:
            del self.tracked_people[tid]
            self.low_confidence_counters.pop(tid, None)

        stale_pairs = [k for k in list(self.fight_pairs.keys())
                       if k[0] not in self.tracked_people or k[1] not in self.tracked_people]
        for k in stale_pairs:
            self.fight_pairs.pop(k, None)
            self.fight_pair_detectors.pop(k, None)

    # ── Head Pose ─────────────────────────────────────────────────────────────
    def _calculate_head_pose(self, person: TrackedPerson) -> str:
        """Returns 'focused', 'looking_away', or 'head_down' (internal signal only)."""
        kp = person.keypoints
        if kp is None or kp.size == 0 or len(kp) < 3:
            return 'focused'
        try:
            nose, left_eye, right_eye = kp[0], kp[1], kp[2]
            if any(len(pt) < 3 for pt in (nose, left_eye, right_eye)):
                return 'focused'

            tid = person.track_id
            nose_ok = nose[2] >= self.LOW_CONFIDENCE_THRESHOLD and nose[0] != 0.0
            left_eye_ok = left_eye[2] >= self.LOW_CONFIDENCE_THRESHOLD and left_eye[0] != 0.0
            right_eye_ok = right_eye[2] >= self.LOW_CONFIDENCE_THRESHOLD and right_eye[0] != 0.0

            # Head down is suspected if the nose is obscured OR both eyes are obscured
            head_down_suspected = (not nose_ok) or (not left_eye_ok and not right_eye_ok)

            if head_down_suspected:
                self.low_confidence_counters[tid] = self.low_confidence_counters.get(tid, 0) + 1
                return 'head_down' if self.low_confidence_counters[tid] >= self.HEAD_DOWN_CONSECUTIVE_FRAMES else 'focused'
            else:
                self.low_confidence_counters[tid] = 0

            # If both eyes are visible, check for horizontal head rotation (yaw) or vertical tilt (pitch)
            if left_eye_ok and right_eye_ok:
                inter_eye = np.linalg.norm(left_eye[:2] - right_eye[:2])
                if inter_eye > 0.1:
                    # Yaw ratio: nose offset from eyes center, normalized by inter-eye distance
                    eyes_center_x = (left_eye[0] + right_eye[0]) / 2.0
                    yaw_ratio = abs(nose[0] - eyes_center_x) / inter_eye
                    if yaw_ratio > 0.35:
                        return 'looking_away'

                    # Pitch (drop) ratio for head down
                    drop_ratio = (nose[1] - (left_eye[1] + right_eye[1]) / 2) / inter_eye
                    if drop_ratio > 0.45:
                        return 'head_down'

                    # Small inter-eye distance relative to bbox height indicates profile view
                    x1, y1, x2, y2 = person.bbox
                    bbox_h = y2 - y1
                    if bbox_h > 10 and inter_eye < bbox_h * 0.09:
                        return 'looking_away'
            else:
                # One eye visible (profile/participation glance) — do not auto-flag distracted.
                if nose_ok:
                    return 'focused'

            return 'focused'
        except Exception as exc:
            print(f'[HEAD] pose error track {person.track_id}: {exc}')
            return 'focused'

    # ── Shared helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _point_near_book(pt, bbox_h: float, book_detections: List[Tuple]) -> bool:
        for (bx1, by1, bx2, by2, _) in book_detections:
            bc = np.array([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0])
            if np.linalg.norm(np.array(pt) - bc) / bbox_h < 0.25:
                return True
        return False

    @staticmethod
    def _hands_spread_writing_posture(wrist_pts, bbox_h: float) -> bool:
        """Two hands low but horizontally separated — typical writing, not phone."""
        if len(wrist_pts) != 2 or bbox_h <= 0:
            return False
        spread = np.linalg.norm(wrist_pts[0] - wrist_pts[1]) / bbox_h
        return spread > 0.28

    def _head_down_like(self, person: TrackedPerson, head_pose: str) -> bool:
        """True when the student is looking down at desk/lap (writing/reading posture)."""
        if head_pose == 'head_down':
            return True
        kp = person.keypoints
        if kp is None or len(kp) < 3:
            return False
        try:
            nose, left_eye, right_eye = kp[0], kp[1], kp[2]
            if (len(nose) >= 3 and len(left_eye) >= 3 and len(right_eye) >= 3 and
                    nose[2] >= 0.4 and left_eye[2] >= 0.4 and right_eye[2] >= 0.4):
                inter_eye = np.linalg.norm(left_eye[:2] - right_eye[:2])
                if inter_eye > 0.1:
                    drop_ratio = (nose[1] - (left_eye[1] + right_eye[1]) / 2) / inter_eye
                    return drop_ratio > 0.35
        except Exception:
            pass
        return False

    def _is_writing_posture(self, person: TrackedPerson,
                            head_pose: str = '',
                            book_detections: Optional[List[Tuple]] = None) -> bool:
        """
        Skeleton-only writing/reading at desk or notebook on lap.
        Covers the common case: head down, one hand with pen, notebook not detected by YOLO.
        """
        book_detections = book_detections or []
        if self._head_down_is_writing(person, book_detections):
            return True

        if not head_pose:
            head_pose = self._calculate_head_pose(person)
        if not self._head_down_like(person, head_pose):
            return False

        kp = person.keypoints
        x1, y1, x2, y2 = person.bbox
        bbox_h = y2 - y1
        if kp is None or kp.size == 0 or len(kp) <= 10 or bbox_h <= 0:
            return False

        lap_thresh = y1 + bbox_h * 0.52   # notebook-on-lap zone (above strict phone lap line)
        wrists = []
        for idx in (9, 10):
            w = kp[idx]
            if len(w) >= 3 and w[2] >= 0.35 and w[0] != 0.0:
                wrists.append(w[:2])

        if not wrists:
            return False

        low_wrists = [w for w in wrists if w[1] > lap_thresh]
        if not low_wrists:
            return False

        # Single visible hand at notebook (pen writing) — most notebook FPs look like this
        if len(wrists) == 1 or len(low_wrists) == 1:
            return True

        if self._hands_spread_writing_posture(wrists, bbox_h):
            return True

        # Two hands on lap but not cupped together → writing / holding paper
        if len(low_wrists) == 2:
            spread = np.linalg.norm(low_wrists[0] - low_wrists[1]) / bbox_h
            if spread > 0.14:
                return True

        return False

    def _head_down_is_writing(self, person: TrackedPerson,
                              book_detections: Optional[List[Tuple]] = None) -> bool:
        """True if a head-down posture is plausibly writing/reading (book/notebook
        near a hand or near the face) rather than generic distraction."""
        book_detections = book_detections or []
        if not book_detections:
            return False
        kp = person.keypoints
        x1, y1, x2, y2 = person.bbox
        bbox_h = y2 - y1
        if bbox_h <= 0:
            return False
        points = []
        if kp is not None and kp.size > 0 and len(kp) > 10:
            for idx in (0, 9, 10):  # nose, left wrist, right wrist
                w = kp[idx]
                if len(w) >= 3 and w[2] >= 0.3 and w[0] != 0.0:
                    points.append(w[:2])
        if not points:
            return False
        if kp is not None and len(kp) > 10:
            wrists = []
            for idx in (9, 10):
                w = kp[idx]
                if len(w) >= 3 and w[2] >= 0.3 and w[0] != 0.0:
                    wrists.append(w[:2])
            if self._hands_spread_writing_posture(wrists, bbox_h):
                return True
        return any(self._point_near_book(pt, bbox_h, book_detections) for pt in points)

    # ── Phone Detection ───────────────────────────────────────────────────────
    def _detect_phone_usage(self, person: TrackedPerson,
                            phone_detections: List[Tuple],
                            head_pose: str,
                            book_detections: Optional[List[Tuple]] = None) -> Tuple[bool, float]:
        x1, y1, x2, y2 = person.bbox
        bbox_h = y2 - y1
        if bbox_h <= 0:
            return False, 0.0

        kp = person.keypoints
        book_detections = book_detections or []
        head_is_down = (head_pose == 'head_down')

        def _book_near(pt) -> bool:
            return self._point_near_book(pt, bbox_h, book_detections)

        def _wrist_ok(w, min_conf=0.4) -> bool:
            return w is not None and len(w) >= 3 and w[2] >= min_conf and w[0] != 0.0

        # Notebook / pen writing — never classify as phone from skeleton alone
        if self._is_writing_posture(person, head_pose, book_detections):
            return False, 0.0

        # ── Strategy A: YOLO-detected phone object near a wrist (most reliable) ──
        if phone_detections and kp is not None and kp.size > 0 and len(kp) > 10:
            wrist_prox_thresh = 0.18 if head_is_down else 0.25
            min_phone_conf = 0.50 if head_is_down else 0.25
            for (px1, py1, px2, py2, conf) in phone_detections:
                if conf < min_phone_conf:
                    continue
                ph, pw = py2 - py1, px2 - px1
                if max(pw, ph) / bbox_h > 0.35:   # ignore huge phone-sized rectangles (TVs etc.)
                    continue
                pc = np.array([(px1 + px2) / 2.0, (py1 + py2) / 2.0])
                if _book_near(pc):
                    continue
                # Check if any wrist is near the detected phone
                for idx in (9, 10):
                    w = kp[idx]
                    if _wrist_ok(w, 0.3) and np.linalg.norm(w[:2] - pc) / bbox_h < wrist_prox_thresh:
                        print(f'[PHONE] Person {person.track_id}: object near wrist, conf={conf:.2f}')
                        return True, float(conf)

        # ── Strategy B: Skeleton heuristic (cupped-hands / call posture only) ──
        if kp is not None and kp.size > 0 and len(kp) > 10:
            try:
                nose        = kp[0]
                left_ear    = kp[3] if len(kp) > 3 else None
                right_ear   = kp[4] if len(kp) > 4 else None
                left_wrist  = kp[9]
                right_wrist = kp[10]
                lap_thresh  = y1 + bbox_h * self.PHONE_LAP_HEIGHT_FRACTION

                nose_ok    = _wrist_ok(nose, 0.7)
                wrist_pts  = [w[:2] for w in (left_wrist, right_wrist) if _wrist_ok(w, 0.5)]

                # Guard: both hands up near face → writing, not phone
                if nose_ok and len(wrist_pts) == 2:
                    if all(np.linalg.norm(w - nose[:2]) / bbox_h < 0.22 for w in wrist_pts):
                        return False, 0.0

                # Both wrists at lap/desk level + hands close together (cupped-phone posture)
                if len(wrist_pts) == 2:
                    both_low    = all(w[1] > lap_thresh for w in wrist_pts)
                    hands_close = np.linalg.norm(wrist_pts[0] - wrist_pts[1]) / bbox_h < 0.28
                    if both_low and hands_close:
                        if self._hands_spread_writing_posture(wrist_pts, bbox_h):
                            pass  # writing spread — not phone
                        elif not (_book_near(wrist_pts[0]) or _book_near(wrist_pts[1])):
                            if head_is_down:
                                print(f'[PHONE] Person {person.track_id}: cupped-hands lap, head_down={head_is_down}')
                                return True, 0.72

                # Hand-to-ear (phone call) — not when head is down reading/writing
                if head_pose != 'head_down':
                    ear_dists = []
                    for w in wrist_pts:
                        best = None
                        for ear in (left_ear, right_ear):
                            if _wrist_ok(ear, 0.4):
                                d = np.linalg.norm(w - ear[:2]) / bbox_h
                                if best is None or d < best:
                                    best = d
                        if best is not None:
                            ear_dists.append(best)

                    if ear_dists:
                        thresh = 0.20 if len(wrist_pts) == 2 else 0.17
                        if min(ear_dists) < thresh:
                            print(f'[PHONE] Person {person.track_id}: hand-to-ear posture')
                            return True, 0.75

            except Exception as exc:
                print(f'[PHONE] Heuristic error: {exc}')

        return False, 0.0

    # ── Eating Detection ──────────────────────────────────────────────────────
    def _detect_eating(self, person: TrackedPerson, food_detections: List[Tuple]) -> Tuple[bool, float]:
        if not food_detections or person.keypoints is None or person.keypoints.size == 0:
            return False, 0.0
        try:
            kp   = person.keypoints
            nose = kp[0]
            x1, y1, x2, y2 = person.bbox
            bbox_h = y2 - y1

            best_food_conf = 0.0
            food_near_mouth = False
            for (fx1, fy1, fx2, fy2, conf) in food_detections:
                if conf < 0.7:
                    continue
                fc = np.array([(fx1 + fx2) / 2, (fy1 + fy2) / 2])
                if np.linalg.norm(fc - nose[:2]) < bbox_h * 0.2:
                    food_near_mouth = True
                    best_food_conf = max(best_food_conf, conf)

            if not food_near_mouth:
                return False, 0.0

            if len(nose) >= 3 and nose[2] >= 0.8:
                for idx in (9, 10):
                    if len(kp) > idx:
                        w = kp[idx]
                        if len(w) >= 3 and w[2] >= 0.8 and w[0] != 0.0:
                            if np.linalg.norm(w[:2] - nose[:2]) / bbox_h < self.EATING_HAND_TO_MOUTH_THRESHOLD:
                                return True, best_food_conf
        except Exception as exc:
            print(f'[EATING] error track {person.track_id}: {exc}')
        return False, 0.0

    def _limb_contact_evidence(self, a: 'TrackedPerson', b: 'TrackedPerson',
                               scale_ref: float) -> Tuple[float, bool, float, int, bool]:
        """
        Returns (contact_score, strong_wrist_proximity, min_limb_distance,
                 grappling_contacts, boxes_overlap).

        Grappling (arm-wrestling, interlocked arms) uses wrists AND elbows —
        wrist-only checks miss most real fights in classroom footage.
        """
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox
        center_a = np.array([(ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0])
        center_b = np.array([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0])

        boxes_overlap = not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

        wrist_limit = scale_ref * self.FIGHT_WRIST_CROSS_RATIO
        limb_limit = scale_ref * self.FIGHT_LIMB_CROSS_RATIO
        contact_score = 0.0
        strong_wrist = False
        min_limb_d = float('inf')
        grappling_contacts = 0

        if (a.keypoints is None or b.keypoints is None or
                len(a.keypoints) <= 10 or len(b.keypoints) <= 10):
            return 0.0, False, min_limb_d, 0, boxes_overlap

        # COCO: 5/6 shoulder, 7/8 elbow, 9/10 wrist
        a_limbs = [(9, 0.35), (10, 0.35), (7, 0.30), (8, 0.30)]
        b_limbs = [(9, 0.35), (10, 0.35), (7, 0.30), (8, 0.30)]

        try:
            for ai, a_min_conf in a_limbs:
                aw = a.keypoints[ai]
                if len(aw) < 3 or aw[2] < a_min_conf:
                    continue
                limit = wrist_limit if ai >= 9 else limb_limit

                for bi, b_min_conf in b_limbs:
                    bw = b.keypoints[bi]
                    if len(bw) < 3 or bw[2] < b_min_conf:
                        continue
                    d = np.linalg.norm(aw[:2] - bw[:2])
                    min_limb_d = min(min_limb_d, d)
                    if d >= limit:
                        continue

                    grappling_contacts += 1
                    if ai >= 9 and bi >= 9:
                        contact_score += 0.35
                        strong_wrist = True
                    elif ai >= 9 or bi >= 9:
                        # wrist-to-elbow / wrist-to-wrist across people — arm wrestling
                        contact_score += 0.28
                    else:
                        contact_score += 0.18

                # Limb projecting into the other person's bbox (reach/grab)
                if bx1 <= aw[0] <= bx2 and by1 <= aw[1] <= by2:
                    if np.linalg.norm(aw[:2] - center_b) < np.linalg.norm(aw[:2] - center_a) * 0.9:
                        grappling_contacts += 1
                        contact_score += 0.12

            for bi, b_min_conf in b_limbs:
                bw = b.keypoints[bi]
                if len(bw) < 3 or bw[2] < b_min_conf:
                    continue
                if ax1 <= bw[0] <= ax2 and ay1 <= bw[1] <= ay2:
                    if np.linalg.norm(bw[:2] - center_a) < np.linalg.norm(bw[:2] - center_b) * 0.9:
                        grappling_contacts += 1
                        contact_score += 0.12

        except Exception as exc:
            print(f'[FIGHT] limb contact error: {exc}')

        return min(contact_score, 0.85), strong_wrist, min_limb_d, grappling_contacts, boxes_overlap

    def _wrist_contact_evidence(self, a, b, scale_ref):
        """Backward-compatible wrapper."""
        score, strong, min_d, _, _ = self._limb_contact_evidence(a, b, scale_ref)
        return score, strong, min_d

    def _calculate_motion_speed(self, person: TrackedPerson, scale_ref: float) -> float:
        """
        Calculate normalized speed of upper body/arm keypoints (wrists, elbows, shoulders, nose)
        by looking at displacement over multiple past frames in a window of 0.08s to 0.4s.
        Taking the maximum speed over these comparisons prevents aliasing of periodic movements
        while requiring dt >= 0.08s filters out high-frequency tracking jitter.
        """
        history = person.keypoint_history
        if len(history) < 2 or scale_ref <= 0:
            return 0.0

        t_curr, kp_curr = history[-1]
        kps_to_check = [0, 5, 6, 7, 8, 9, 10]  # nose, shoulders, elbows, wrists
        max_speed = 0.0
        has_compared = False

        for t_past, kp_past in history:
            dt = t_curr - t_past
            if 0.08 <= dt <= 0.40:
                has_compared = True
                frame_speeds = []
                for idx in kps_to_check:
                    if idx < len(kp_curr) and idx < len(kp_past):
                        pt_curr = kp_curr[idx]
                        pt_past = kp_past[idx]
                        if len(pt_curr) >= 3 and len(pt_past) >= 3 and pt_curr[2] > 0.35 and pt_past[2] > 0.35:
                            if pt_curr[0] != 0.0 and pt_past[0] != 0.0:
                                dist = np.linalg.norm(pt_curr[:2] - pt_past[:2])
                                speed = dist / (scale_ref * dt)
                                frame_speeds.append(speed)
                if frame_speeds:
                    max_speed = max(max_speed, np.max(frame_speeds))

        if not has_compared:
            # Fallback to the oldest frame in history
            t_past, kp_past = history[0]
            dt = t_curr - t_past
            if dt >= 0.08:
                frame_speeds = []
                for idx in kps_to_check:
                    if idx < len(kp_curr) and idx < len(kp_past):
                        pt_curr = kp_curr[idx]
                        pt_past = kp_past[idx]
                        if len(pt_curr) >= 3 and len(pt_past) >= 3 and pt_curr[2] > 0.35 and pt_past[2] > 0.35:
                            if pt_curr[0] != 0.0 and pt_past[0] != 0.0:
                                dist = np.linalg.norm(pt_curr[:2] - pt_past[:2])
                                speed = dist / (scale_ref * dt)
                                frame_speeds.append(speed)
                if frame_speeds:
                    max_speed = max(max_speed, np.max(frame_speeds))

        return float(max_speed)

    def _skeleton_fight_score(self, a: 'TrackedPerson', b: 'TrackedPerson') -> float:
        """
        Skeleton fight score in [0, 1]. Uses wrist + elbow contact (grappling)
        and mutual upper-body motion. Passing paper still stays below threshold.
        """
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox

        scale_ref = max(
            ((ay2 - ay1) + (by2 - by1)) / 2.0,
            ((ax2 - ax1) + (bx2 - bx1)) / 2.0,
        )
        if scale_ref <= 0:
            return 0.0

        speed_a = self._calculate_motion_speed(a, scale_ref)
        speed_b = self._calculate_motion_speed(b, scale_ref)
        max_speed = max(speed_a, speed_b)
        min_speed = min(speed_a, speed_b)

        contact_score, strong_wrist, min_limb_d, grappling_n, overlap = (
            self._limb_contact_evidence(a, b, scale_ref))

        if contact_score <= 0.0:
            print(f"[FIGHT-SKEL] Pair {a.track_id}-{b.track_id}: no contact evidence, "
                  f"speeds=({speed_a:.2f}, {speed_b:.2f}) -> score=0.00")
            return 0.0

        score = min(contact_score, 0.60)

        # ── Grappling path (arm-wrestling / interlocked arms) ─────────────────
        if grappling_n >= 2 and overlap:
            if grappling_n >= 3:
                # Sustained multi-limb contact — real grappling even if motion is slow
                score += 0.20
            if min_speed >= 0.25 or max_speed >= 0.35:
                score += 0.12
            if min_speed >= 0.45 and max_speed >= 0.65:
                score += 0.15

        elif strong_wrist:
            if max_speed >= 1.5 and min_speed >= 0.8:
                score += 0.45
            elif max_speed >= 1.2 and min_speed >= 0.6:
                score += 0.28
            elif max_speed >= 0.8:
                score += 0.12
        else:
            # Single reach into neighbor — still capped below threshold unless fast
            if max_speed >= 1.6 and min_speed >= 1.2:
                score += 0.30
            elif max_speed >= 1.3 and min_speed >= 1.0:
                score += 0.15
            else:
                score = min(score, 0.45)

        if max_speed < 0.35 and grappling_n < 2:
            score = min(score, 0.35)

        print(f"[FIGHT-SKEL] Pair {a.track_id}-{b.track_id}: contact={contact_score:.2f}, "
              f"grapple={grappling_n}, overlap={overlap}, min_limb_d={min_limb_d:.1f}, "
              f"speeds=({speed_a:.2f}, {speed_b:.2f}), score={score:.2f}")

        return min(score, 1.0)

    def _detect_fighting_pairwise(self, frame: np.ndarray) -> List[Tuple[int, int, float, Dict]]:
        """
        Pairwise fight detection.
        Primary: skeleton heuristic (always available).
        Secondary: 3D CNN on cropped pair region (when available).
        Returns list of (person_a_id, person_b_id, confidence, fight_info).
        """
        if len(self.tracked_people) < 2:
            return []

        people = list(self.tracked_people.values())
        fight_results = []
        candidate_pairs = []

        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                a, b = people[i], people[j]
                ax1, ay1, ax2, ay2 = a.bbox
                bx1, by1, bx2, by2 = b.bbox

                ca = np.array([(ax1 + ax2) / 2, (ay1 + ay2) / 2])
                cb = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])
                dist = np.linalg.norm(ca - cb)

                scale_ref = max(
                    ((ay2 - ay1) + (by2 - by1)) / 2.0,
                    ((ax2 - ax1) + (bx2 - bx1)) / 2.0,
                )
                if scale_ref <= 0:
                    continue

                boxes_overlap = not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)
                if dist > scale_ref * self.FIGHT_PROXIMITY_RATIO and not boxes_overlap:
                    continue

                # Pre-filter: a pair only becomes a fight CANDIDATE if there is
                # actual wrist-level evidence of contact. Two students simply
                # seated next to each other will very often have OVERLAPPING
                # BOUNDING BOXES (shoulders/arms touching) with zero physical
                # contact — bbox overlap on its own is NOT evidence of contact
                # and must never bypass this check.
                has_crossing = False
                if (a.keypoints is not None and b.keypoints is not None and
                        len(a.keypoints) > 10 and len(b.keypoints) > 10):
                    try:
                        _, strong_prox, min_d, grapple_n, overlap = (
                            self._limb_contact_evidence(a, b, scale_ref))
                        limb_gate = scale_ref * self.FIGHT_LIMB_CROSS_RATIO * 1.15
                        has_crossing = (
                            grapple_n >= 2
                            or strong_prox
                            or min_d < scale_ref * self.FIGHT_WRIST_CROSS_RATIO * 1.1
                            or (overlap and grapple_n >= 1 and min_d < limb_gate)
                        )
                    except Exception:
                        has_crossing = False
                if not has_crossing:
                    continue

                candidate_pairs.append((a.track_id, b.track_id, dist, scale_ref))

        if candidate_pairs:
            print(f'[FIGHT] {len(candidate_pairs)} candidate pair(s): '
                  f'{[(a, b, round(d, 1)) for a, b, d, _ in candidate_pairs]}')

        for a_id, b_id, dist, scale_ref in candidate_pairs:
            if a_id not in self.tracked_people or b_id not in self.tracked_people:
                continue

            pa, pb   = self.tracked_people[a_id], self.tracked_people[b_id]
            pair_key = tuple(sorted([a_id, b_id]))

            # ── Skeleton heuristic score (always runs) ────────────────────────
            skel_score = self._skeleton_fight_score(pa, pb)
            print(f'[FIGHT] pair {pair_key}: skeleton_score={skel_score:.2f}')

            if pair_key not in self.fight_pairs:
                self.fight_pairs[pair_key] = {
                    'skel_confirms': 0,
                    'cnn_confirms': 0,
                    'last_skel_time': 0.0,
                    'first_skel_time': 0.0,
                    'last_cnn_time': 0.0,
                }
            state = self.fight_pairs[pair_key]
            now   = time.time()

            # Skeleton path: score >= FIGHT_SCORE_THRESHOLD triggers a confirmation tick.
            # All ticks for a single fight call must land within FIGHT_CONFIRM_WINDOW_SEC
            # of the FIRST tick — if too much time passes without re-confirming, the
            # whole streak resets. This prevents scattered, unrelated jitter spikes
            # spread out over many seconds from ever accumulating into a "confirmed" fight.
            if skel_score >= self.FIGHT_SCORE_THRESHOLD:
                if state['skel_confirms'] == 0 or (now - state['first_skel_time']) > self.FIGHT_CONFIRM_WINDOW_SEC:
                    state['skel_confirms']   = 1
                    state['first_skel_time'] = now
                elif now - state['last_skel_time'] > 0.2:
                    state['skel_confirms'] += 1
                state['last_skel_time'] = now

                if state['skel_confirms'] >= self.FIGHT_SKELETON_CONFIRMS:
                    print(f'[FIGHT] pair {pair_key}: FIGHT confirmed (skeleton)! score={skel_score:.2f}')
                    fight_results.append((a_id, b_id, min(skel_score, 0.85), {
                        'person_a_id': a_id, 'person_b_id': b_id,
                        'confidence': skel_score, 'trigger': 'skeleton',
                        'distance': dist, 'confirmations': state['skel_confirms'],
                    }))
                    continue
            else:
                # Any sub-threshold frame breaks the streak immediately rather than
                # slowly decaying — a real fight produces threshold-crossing contact
                # repeatedly and quickly, not once every couple of seconds.
                state['skel_confirms'] = 0

            # ── 3D CNN path (when available, used for borderline cases) ───────
            if not self._cnn_available:
                continue

            ax1, ay1, ax2, ay2 = pa.bbox
            bx1, by1, bx2, by2 = pb.bbox
            cx1 = max(0, min(ax1, bx1) - 20)
            cy1 = max(0, min(ay1, by1) - 20)
            cx2 = min(frame.shape[1], max(ax2, bx2) + 20)
            cy2 = min(frame.shape[0], max(ay2, by2) + 20)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            if pair_key not in self.fight_pair_detectors:
                try:
                    from classroom_monitor.fight_detection_3dcnn import FightDetector3DCNN
                    self.fight_pair_detectors[pair_key] = FightDetector3DCNN(
                        device='auto', sequence_length=16, confidence_threshold=0.60)
                except Exception as e:
                    print(f'[FIGHT] Error creating pair detector: {e}')
                    continue

            cnn_det = self.fight_pair_detectors[pair_key]
            if not cnn_det.detection_available:
                continue

            cnn_det.add_frame(crop)

            if not cnn_det.is_ready():
                continue

            try:
                is_fighting, conf = cnn_det.predict()
                print(f'[FIGHT] pair {pair_key}: cnn conf={conf:.2f} fighting={is_fighting}')
                if is_fighting and conf > 0.6:
                    if now - state['last_cnn_time'] > 0.3:
                        state['cnn_confirms'] += 1
                        state['last_cnn_time'] = now
                    if state['cnn_confirms'] >= self.FIGHT_CNN_CONFIRMS:
                        fight_results.append((a_id, b_id, conf, {
                            'person_a_id': a_id, 'person_b_id': b_id,
                            'confidence': conf, 'trigger': 'pairwise_3dcnn',
                            'distance': dist, 'confirmations': state['cnn_confirms'],
                        }))
                else:
                    if now - state['last_cnn_time'] > 1.0:
                        state['cnn_confirms'] = max(0, state['cnn_confirms'] - 1)
            except Exception as e:
                print(f'[FIGHT] CNN error: {e}')

        return fight_results

    # ── Evaluate Person ───────────────────────────────────────────────────────
    def evaluate_person(self, track_id: int,
                        phone_detections: List[Tuple],
                        food_detections: List[Tuple],
                        frame: Optional[np.ndarray] = None,
                        fight_override: Optional[Tuple[bool, float, Dict]] = None,
                        book_detections: Optional[List[Tuple]] = None) -> DetectionResult:
        if track_id not in self.tracked_people:
            return DetectionResult(
                type='not_visible', bbox=(0, 0, 0, 0), confidence=0.0,
                color=COLOR_MAP['not_visible'], label=LABEL_MAP['not_visible'],
                is_alert=False, is_distracted=False)

        person = self.tracked_people[track_id]
        fight_detected, fight_confidence, fight_info = False, 0.0, None
        raw_confidence = person.last_raw_confidence

        if fight_override is not None:
            fight_detected, fight_confidence, fight_info = fight_override
        elif frame is not None:
            for a_id, b_id, conf, info in self._detect_fighting_pairwise(frame):
                if track_id in (a_id, b_id):
                    fight_detected, fight_confidence, fight_info = True, conf, info
                    break

        # ── Append raw behavior to history ──────────────────────────────────
        if fight_detected:
            person.behavior_history.append('fighting')
            raw_confidence = fight_confidence
        else:
            # IMPORTANT: merely being a "fight candidate" (i.e. within proximity
            # of another tracked person — true for almost every student in a
            # normal classroom seating arrangement) must NEVER suppress normal
            # behavior classification. Previously this branch did `pass`, which
            # froze phone/eating/focus evaluation for any student sitting near
            # a classmate — which is most students, most of the time. Only an
            # ACTUALLY CONFIRMED fight (handled above) should override the
            # regular checks below.
            raw_head = self._calculate_head_pose(person)
            is_phone, phone_conf = self._detect_phone_usage(person, phone_detections, raw_head, book_detections)
            if is_phone:
                person.behavior_history.append('using_phone')
                raw_confidence = phone_conf
            else:
                is_eating, eat_conf = self._detect_eating(person, food_detections)
                if is_eating:
                    person.behavior_history.append('eating_food')
                    raw_confidence = eat_conf
                elif (raw_head == 'head_down' or self._head_down_like(person, raw_head)) and (
                        self._is_writing_posture(person, raw_head, book_detections)):
                    # Head-down writing / reading at notebook — normal classroom work
                    person.behavior_history.append('focused')
                    raw_confidence = 0.85
                else:
                    person.behavior_history.append('focused' if raw_head == 'focused' else 'distracted')
                    raw_confidence = 0.75 if raw_head == 'focused' else 0.65

        person.last_raw_confidence = raw_confidence

        # ── Temporal smoothing ────────────────────────────────────────────────
        # Alert states (phone, fight, eating): confirm on ALERT_CONFIRM_FRAMES consecutive.
        # Non-alert states: confirm on NORMAL_CONFIRM_FRAMES consecutive or >50% majority.
        # Alert states CAN also win via majority — they are no longer blocked.
        final_behavior = person.last_final_behavior
        history = list(person.behavior_history)

        if len(history) >= 1:
            last = history[-1]

            if last in ALERT_POSES:
                # Alert: confirm after N consecutive frames
                n = min(self.ALERT_CONFIRM_FRAMES, len(history))
                if all(h == last for h in history[-n:]):
                    final_behavior = last
            else:
                # Non-alert: 3-in-a-row or majority over last 6 frames
                if len(history) >= self.NORMAL_CONFIRM_FRAMES:
                    recent = history[-self.NORMAL_CONFIRM_FRAMES:]
                    if len(set(recent)) == 1:
                        final_behavior = recent[0]
                    else:
                        window = history[-6:]
                        top_label, top_count = Counter(window).most_common(1)[0]
                        if top_count / len(window) > 0.5:
                            final_behavior = top_label
                else:
                    final_behavior = last

        person.last_final_behavior = final_behavior

        # ── Build result ─────────────────────────────────────────────────────
        if final_behavior == 'fighting':
            confidence = fight_confidence or raw_confidence or 0.8
            is_alert, is_distracted = True, False
        else:
            confidence = raw_confidence if final_behavior == person.behavior_history[-1] else person.last_raw_confidence
            confidence = confidence or 0.75
            is_alert     = final_behavior in ALERT_POSES
            is_distracted = final_behavior in DISTRACTED_POSES

        return DetectionResult(
            type=final_behavior,
            bbox=person.bbox,
            confidence=confidence,
            color=COLOR_MAP.get(final_behavior, COLOR_MAP['not_visible']),
            label=LABEL_MAP.get(final_behavior, final_behavior),
            is_alert=is_alert,
            is_distracted=is_distracted,
            track_id=track_id,
            fight_info=fight_info if final_behavior == 'fighting' else None,
        )


# ── Shared YOLO models (one load per process) ────────────────────────────────
class _SharedYOLOModels:
    _lock = threading.Lock()
    _pose_model = None
    _object_model = None

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._pose_model is None:
                try:
                    from ultralytics import YOLO
                    cls._pose_model = YOLO('yolo11s-pose.pt')
                    cls._object_model = YOLO('yolo11s.pt')
                    print('[OK] Shared YOLO pose + object models loaded')
                except Exception as e:
                    print(f'[WARN] Shared YOLO model load failed: {e}')
            return cls._pose_model, cls._object_model


# ── Production Stream Processor ──────────────────────────────────────────────
class ProductionStreamProcessor:
    """Thread-isolated stream processor: fixed ring buffer, configurable FPS, auto-reconnect."""

    # COCO class IDs
    _PHONE_CLS = {67}
    _FOOD_CLS  = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55}  # banana..cake (56=chair excluded)
    _BOOK_CLS  = {73}

    def __init__(self, process_fps: int = 10, buffer_size: int = 5):
        self.process_fps    = process_fps
        self.frame_interval = 1.0 / process_fps
        self.frame_buffer: deque = deque(maxlen=buffer_size)
        self.result_buffer: deque = deque(maxlen=2)
        self.yolo_model   = None
        self.object_model = None
        self.running      = False
        self.lock         = threading.Lock()
        self.stop_event   = threading.Event()
        self.phone_detections: List[Tuple] = []
        self.food_detections:  List[Tuple] = []
        self.person_tracks: List[Tuple] = []
        self.behavior_engine = TemporalBehaviorEngine()
        self._ensure_models()

    def _ensure_models(self):
        self.yolo_model, self.object_model = _SharedYOLOModels.get()

    def _load_models(self):
        """Backward-compatible alias."""
        self._ensure_models()

    def _capture_frames(self, camera_url: str):
        cap, reconnect_delay = None, 1.0
        while not self.stop_event.is_set():
            try:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(camera_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    if cap.isOpened():
                        print('[OK] Camera connected')
                        reconnect_delay = 1.0
                    else:
                        time.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 2, 10.0)
                        continue
                ret, frame = cap.read()
                if not ret:
                    cap.release(); cap = None
                    continue
                with self.lock:
                    self.frame_buffer.append((time.time(), frame.copy()))
            except Exception as e:
                print(f'[ERROR] Capture: {e}')
                if cap: cap.release(); cap = None
                time.sleep(1.0)
        if cap: cap.release()

    def _process_frames(self):
        last_t = 0.0
        while not self.stop_event.is_set():
            now = time.time()
            if now - last_t >= self.frame_interval:
                frame, ts = None, now
                with self.lock:
                    if self.frame_buffer:
                        ts, frame = self.frame_buffer[-1]
                        self.frame_buffer.clear()
                if frame is not None:
                    self._process_single_frame(frame, ts)
                    last_t = now
            time.sleep(0.001)

    def _parse_object_detections(self, frame: np.ndarray):
        """Run object model and return (phone_dets, food_dets, book_dets)."""
        phone_dets, food_dets, book_dets = [], [], []
        if self.object_model is None:
            return phone_dets, food_dets, book_dets
        try:
            for result in self.object_model(frame, verbose=False, conf=0.3, iou=0.5):
                if result.boxes is None:
                    continue
                for i in range(len(result.boxes)):
                    cls  = int(result.boxes.cls[i])
                    conf = float(result.boxes.conf[i])
                    x1, y1, x2, y2 = map(int, result.boxes.xyxy[i])
                    det = (x1, y1, x2, y2, conf)
                    if cls in self._PHONE_CLS:
                        phone_dets.append(det)
                    elif cls in self._FOOD_CLS:
                        food_dets.append(det)
                    elif cls in self._BOOK_CLS:
                        book_dets.append(det)
        except Exception as e:
            print(f'[WARN] Object detection failed: {e}')
        return phone_dets, food_dets, book_dets

    def _parse_pose_detections(self, frame: np.ndarray):
        """Run pose model and return list of (track_id, x1, y1, x2, y2, conf, keypoints)."""
        tracks = []
        if self.yolo_model is None:
            return tracks
        try:
            for result in self.yolo_model.track(frame, persist=True, verbose=False,
                                                  conf=0.2, iou=0.5, tracker='bytetrack.yaml'):
                if result.boxes is None:
                    continue
                kp_list = result.keypoints if hasattr(result, 'keypoints') else None
                for i in range(len(result.boxes)):
                    if int(result.boxes.cls[i]) != 0:
                        continue
                    x1, y1, x2, y2 = map(int, result.boxes.xyxy[i])
                    conf     = float(result.boxes.conf[i])
                    track_id = int(result.boxes.id[i]) if result.boxes.id is not None else i
                    kp = None
                    if kp_list is not None and i < len(kp_list):
                        try:
                            kp = kp_list[i].data.cpu().numpy()[0]
                        except Exception:
                            pass
                    tracks.append((track_id, x1, y1, x2, y2, conf, kp))
        except Exception as e:
            print(f'[ERROR] Pose detection: {e}')
        return tracks

    def _run_behavior_evaluation(self, frame, person_tracks, phone_dets, food_dets, book_dets, timestamp):
        """Update engine, run fight detection once, evaluate current-frame persons."""
        active_tids = set()
        for tid, x1, y1, x2, y2, conf, kp in person_tracks:
            active_tids.add(tid)
            self.behavior_engine.update_person(tid, (x1, y1, x2, y2), kp, timestamp)

        self.behavior_engine.cleanup_stale(timestamp)

        fight_results = self.behavior_engine._detect_fighting_pairwise(frame)
        fight_map = {}
        for a_id, b_id, conf, info in fight_results:
            fight_map[a_id] = (conf, info)
            fight_map[b_id] = (conf, info)

        results = []
        for tid in active_tids:
            if tid not in self.behavior_engine.tracked_people:
                continue
            f_conf, f_info = fight_map.get(tid, (0.0, None))
            override = (tid in fight_map, f_conf, f_info) if tid in fight_map else None
            results.append(self.behavior_engine.evaluate_person(
                tid, phone_dets, food_dets, frame=None,
                fight_override=override, book_detections=book_dets))
        return results

    def _process_single_frame(self, frame: np.ndarray, timestamp: float):
        try:
            person_tracks              = self._parse_pose_detections(frame)
            phone_dets, food_dets, book_dets = self._parse_object_detections(frame)
            final_results = self._run_behavior_evaluation(
                frame, person_tracks, phone_dets, food_dets, book_dets, timestamp)
            with self.lock:
                self.phone_detections = phone_dets
                self.food_detections  = food_dets
                self.person_tracks    = [(t[0], t[1], t[2], t[3], t[4], t[5]) for t in person_tracks]
                self.result_buffer.append((timestamp, frame.copy(), final_results))
        except Exception as e:
            print(f'[ERROR] Frame processing: {e}')

    def start(self, camera_url: str):
        if self.running:
            return
        self.running = True
        self.stop_event.clear()
        threading.Thread(target=self._capture_frames, args=(camera_url,), daemon=True).start()
        threading.Thread(target=self._process_frames, daemon=True).start()
        print('[OK] Stream processor started')

    def stop(self):
        self.running = False
        self.stop_event.set()
        print('[OK] Stream processor stopped')

    def get_latest_result(self) -> Tuple[Optional[float], Optional[np.ndarray], List[DetectionResult]]:
        with self.lock:
            return self.result_buffer[-1] if self.result_buffer else (None, None, [])


# ── ClassroomBehaviorDetector (backward-compatible public API) ───────────────
class ClassroomBehaviorDetector:
    """Drop-in replacement maintaining the original public API."""

    def __init__(self, camera_url, camera_id,
                 server_url='http://localhost:8000',
                 alert_cooldown=120,
                 whatsapp_admin=None):
        self.camera_url      = camera_url
        self.camera_id       = camera_id
        self.server_url      = server_url.rstrip('/')
        self.alert_cooldown  = alert_cooldown
        self.whatsapp_admin  = whatsapp_admin or os.environ.get('ADMIN_WHATSAPP', '')
        self._face_recognizer = None
        self.known_students  = []
        self.last_alert_time: dict = defaultdict(float)
        self.running         = False
        self.thread          = None
        self.processor       = ProductionStreamProcessor(process_fps=10)
        self._api_key        = os.environ.get('DETECTION_API_KEY', '').strip()

    @property
    def behavior_engine(self):
        return self.processor.behavior_engine

    @property
    def yolo_model(self):
        return self.processor.yolo_model

    @property
    def object_model(self):
        return self.processor.object_model

    def _init_face_recognition(self):
        """Load student encodings once — DB direct when Django is up, else HTTP."""
        if self._face_recognizer is not None:
            return
        try:
            from classroom_monitor.face_recognition_helper import StudentFaceRecognizer
            rec = StudentFaceRecognizer()
            rec.load_from_db()
            if rec._known_encodings:
                self._face_recognizer = rec
                print(f'[OK] Face recognition: {len(rec._known_encodings)} encodings from DB')
                return
        except Exception as e:
            print(f'[WARN] DB face encodings unavailable ({e}), trying HTTP')

        self._load_known_students_http()

    def _load_known_students_http(self):
        import requests as _req
        url = f'{self.server_url}/camera-attendance/api/students/encodings/'
        headers = {}
        if self._api_key:
            headers['X-Detection-API-Key'] = self._api_key
        try:
            r = _req.get(url, timeout=5, verify=_http_verify_ssl(), headers=headers)
            r.raise_for_status()
            self.known_students = r.json()
            print(f'[OK] {len(self.known_students)} student encodings loaded via HTTP')
        except Exception as e:
            print(f'[WARN] Could not load students: {e}')
            self.known_students = []

    def detect(self, frame) -> List[Dict]:
        """Detect behaviors in a single frame. Returns list of detection dicts."""
        if self.processor.yolo_model is None:
            return []
        try:
            timestamp     = time.time()
            person_tracks = self.processor._parse_pose_detections(frame)
            phone_dets, food_dets, book_dets = self.processor._parse_object_detections(frame)
            results = self.processor._run_behavior_evaluation(
                frame, person_tracks, phone_dets, food_dets, book_dets, timestamp)

            detections = []
            for det in results:
                d = {
                    'type': det.type, 'bbox': det.bbox, 'confidence': det.confidence,
                    'color': det.color, 'label': det.label,
                    'is_alert': det.is_alert, 'is_distracted': det.is_distracted,
                    'track_id': det.track_id,
                }
                if det.fight_info:
                    d['fight_info'] = det.fight_info
                detections.append(d)
            return detections
        except Exception as e:
            print(f'[ERROR] detect(): {e}')
            return []

    def _detect_behaviors(self, frame):
        return self.detect(frame)

    def _draw_detections(self, frame, detections):
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color, label, conf = det['color'], det['label'], det['confidence']
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"{label} ({conf:.2f})"
            tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            bg_y1 = max(0, y1 - th - 4)
            cv2.rectangle(out, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, text, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out

    def _report_incident(self, detection, frame, student_id, student_name, roll_no, all_detections=None):
        import requests as _req
        try:
            annotated = self._draw_detections(frame, all_detections) if all_detections else frame
            _, buf   = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            snap_b64 = base64.b64encode(buf).decode()
            if detection['type'] == 'fighting':
                fi    = detection.get('fight_info', {})
                tag   = f"{student_name} ({roll_no}) vs student_{fi.get('person_b_id', '?')}"
                sev   = 'CRITICAL'
            else:
                tag = f"{student_name} ({roll_no})" if student_id else 'Unknown'
                sev = 'WARNING'
            resp = _req.post(
                f'{self.server_url}/classroom/api/incidents/report/',
                json={
                    'student_id': student_id, 'camera_id': self.camera_id,
                    'incident_type': detection['type'],
                    'confidence': round(detection['confidence'], 3),
                    'snapshot': snap_b64, 'student_name': student_name,
                    'roll_no': roll_no,
                    'description': f"{sev} {detection['label']} — {tag}",
                },
                headers={'X-Detection-API-Key': self._api_key} if self._api_key else {},
                timeout=10, verify=_http_verify_ssl())
            print(f"[INCIDENT] {detection['label']} | {tag} | {resp.status_code}")
        except Exception as e:
            print(f'[ERROR] report_incident: {e}')

    def _recognize_face(self, frame, bbox):
        self._init_face_recognition()
        try:
            x1, y1, x2, y2 = bbox
            mid_y = y1 + int((y2 - y1) * 0.55)
            crop  = frame[y1:mid_y, x1:x2]
            if crop.size == 0:
                crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None, 'Unknown', ''

            if self._face_recognizer is not None:
                sid, name, roll, _dist = self._face_recognizer.match(crop)
                if sid:
                    return sid, name, roll
                return None, name, roll

            if not self.known_students:
                return None, 'Unknown', ''

            import face_recognition as fr
            rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            encs = fr.face_encodings(rgb, num_jitters=1, model='small')
            if not encs:
                return None, 'Unknown', ''
            det    = encs[0]
            best_d, best = 1.0, None
            for s in self.known_students:
                try:
                    d = fr.face_distance([np.array(json.loads(s['encoding']))], det)[0]
                    if d < best_d:
                        best_d, best = d, s
                except Exception:
                    continue
            if best_d < FACE_MATCH_TOLERANCE and best:
                return best['id'], best['name'], best.get('roll_no', '')
            return None, 'Unknown', ''
        except Exception as e:
            print(f'[ERROR] face_recog: {e}')
            return None, 'Unknown', ''

    def _detection_loop(self):
        self.processor.start(self.camera_url)
        frame_count, fight_cooldown, last_ts = 0, defaultdict(float), 0.0

        while self.running:
            ts, frame, det_objs = self.processor.get_latest_result()
            if frame is None or ts <= last_ts:
                time.sleep(0.01)
                continue
            last_ts = ts

            detections = []
            for d in det_objs:
                entry = {
                    'type': d.type, 'bbox': d.bbox, 'confidence': d.confidence,
                    'color': d.color, 'label': d.label,
                    'is_alert': d.is_alert, 'is_distracted': d.is_distracted,
                    'track_id': d.track_id,
                }
                if d.fight_info:
                    entry['fight_info'] = d.fight_info
                detections.append(entry)

            frame_count += 1
            now = time.time()

            fight_dets = [d for d in detections if d['type'] == 'fighting']
            other_dets = [d for d in detections if d['type'] != 'fighting']

            for det in other_dets:
                if not (det['is_alert'] or det['is_distracted']):
                    continue
                tid = det.get('track_id')
                key = (det['type'], tid if tid is not None else tuple(det['bbox']))
                if now - self.last_alert_time.get(key, 0) < self.alert_cooldown:
                    continue
                sid, name, roll = self._recognize_face(frame, det['bbox'])
                self._report_incident(det, frame, sid, name, roll, all_detections=detections)
                self.last_alert_time[key] = now

            for det in fight_dets:
                fi       = det.get('fight_info', {})
                pair_key = tuple(sorted([fi.get('person_a_id', 0), fi.get('person_b_id', 0)]))
                if now - fight_cooldown.get(pair_key, 0) < 60:
                    continue
                sid, name, roll = self._recognize_face(frame, det['bbox'])
                self._report_incident(det, frame, sid, name, roll, all_detections=detections)
                fight_cooldown[pair_key] = now

            if frame_count % 300 == 0:
                print(f'[INFO] Frame {frame_count} | {len(detections)} detections')

        self.processor.stop()

    def start(self):
        if self.running:
            return
        self._init_face_recognition()
        self.running = True
        self.thread  = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print('[OK] Behavior detection started')

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.processor.stop()
        print('[OK] Behavior detection stopped')