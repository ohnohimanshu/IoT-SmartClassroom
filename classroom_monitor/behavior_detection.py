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


# ── Constants ────────────────────────────────────────────────────────────────
COLOR_MAP = {
    'focused':     (0,  200,  60),
    'distracted':  (0,  165, 255),
    'using_phone': (0,    0, 220),
    'eating_food': (0,    0, 220),
    'fighting':    (0,    0, 255),
    'not_visible': (120, 120, 120),
}
LABEL_MAP = {
    'focused':     'Focused',
    'distracted':  'Distracted',
    'using_phone': 'Using Phone',
    'eating_food': 'Eating Food',
    'fighting':    'FIGHT',
    'not_visible': 'Not Visible',
}

ALERT_POSES      = {'using_phone', 'eating_food', 'fighting'}
DISTRACTED_POSES = {'distracted'}
FACE_MATCH_TOLERANCE = 0.52


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
    FIGHT_PROXIMITY_RATIO       = 1.8    # candidate if centers within 1.8x scale_ref
    FIGHT_WRIST_PROXIMITY_RATIO = 0.45   # wrists within 0.45x scale_ref = physical contact
    FIGHT_SKELETON_CONFIRMS     = 2      # consecutive skeleton frames before flagging fight
    FIGHT_CNN_CONFIRMS          = 2      # CNN confirmations before flagging fight

    # Alert smoothing
    ALERT_CONFIRM_FRAMES  = 2            # phone/fight: 2-in-a-row confirms
    NORMAL_CONFIRM_FRAMES = 3            # focused/distracted: 3-in-a-row

    def __init__(self):
        self.tracked_people: Dict[int, TrackedPerson] = {}
        self.cleanup_threshold = 5.0
        self.low_confidence_counters: Dict[int, int] = {}

        # Pairwise fight state
        self.fight_pairs: Dict[Tuple[int, int], Dict] = {}
        self.fight_pair_detectors: Dict[Tuple[int, int], object] = {}
        self._fight_candidate_ids: set = set()

        # Skeleton-based fight confirmation counters (no CNN needed)
        self._skeleton_fight_counters: Dict[Tuple[int, int], int] = {}

        self._init_fight_detector()

    def _init_fight_detector(self):
        try:
            from classroom_monitor.fight_detection_3dcnn import FightDetector3DCNN
            self._fight_detector_3dcnn = FightDetector3DCNN(
                device='auto', sequence_length=16, confidence_threshold=0.60)
            self._cnn_available = True
            print('[BEHAVIOR] 3D CNN fight detector initializing...')
        except Exception as e:
            print(f'[BEHAVIOR] 3D CNN unavailable ({e}), using skeleton-only fight detection')
            self._fight_detector_3dcnn = None
            self._cnn_available = False

    def update_person(self, track_id: int, bbox: Tuple, keypoints: Optional[np.ndarray], timestamp: float):
        if track_id not in self.tracked_people:
            self.tracked_people[track_id] = TrackedPerson(
                track_id=track_id, bbox=bbox, keypoints=keypoints, last_seen=timestamp)
        else:
            p = self.tracked_people[track_id]
            p.bbox, p.keypoints, p.last_seen = bbox, keypoints, timestamp

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
            self._skeleton_fight_counters.pop(k, None)

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
            low_conf = (
                nose[2] < self.LOW_CONFIDENCE_THRESHOLD or
                left_eye[2] < self.LOW_CONFIDENCE_THRESHOLD or
                right_eye[2] < self.LOW_CONFIDENCE_THRESHOLD or
                nose[0] == 0.0 or left_eye[0] == 0.0 or right_eye[0] == 0.0
            )
            if low_conf:
                self.low_confidence_counters[tid] = self.low_confidence_counters.get(tid, 0) + 1
                return 'head_down' if self.low_confidence_counters[tid] >= self.HEAD_DOWN_CONSECUTIVE_FRAMES else 'focused'
            else:
                self.low_confidence_counters[tid] = 0

            inter_eye = np.linalg.norm(left_eye[:2] - right_eye[:2])
            if inter_eye < 0.1:
                return 'focused'

            drop_ratio = (nose[1] - (left_eye[1] + right_eye[1]) / 2) / inter_eye
            if drop_ratio > 0.45:
                return 'head_down'

            x1, y1, x2, y2 = person.bbox
            bbox_h = y2 - y1
            if bbox_h > 10 and inter_eye < bbox_h * 0.05:
                return 'looking_away'

            return 'focused'
        except Exception:
            return 'focused'

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
            for (bx1, by1, bx2, by2, _) in book_detections:
                bc = np.array([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0])
                if np.linalg.norm(np.array(pt) - bc) / bbox_h < 0.25:
                    return True
            return False

        def _wrist_ok(w, min_conf=0.4) -> bool:
            return w is not None and len(w) >= 3 and w[2] >= min_conf and w[0] != 0.0

        # ── Strategy A: YOLO-detected phone object near a wrist (most reliable) ──
        # head_down is NOT required here — a phone is a phone regardless of pose.
        if phone_detections and kp is not None and kp.size > 0 and len(kp) > 10:
            for (px1, py1, px2, py2, conf) in phone_detections:
                if conf < 0.25:          # low floor; wrist-proximity gate filters FPs
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
                    if _wrist_ok(w, 0.3) and np.linalg.norm(w[:2] - pc) / bbox_h < 0.25:
                        print(f'[PHONE] Person {person.track_id}: object near wrist, conf={conf:.2f}')
                        return True, float(conf)
                # Even without a wrist match: if phone bbox overlaps person bbox → likely theirs
                phone_in_bbox = (px1 >= x1 - 10 and py1 >= y1 - 10 and
                                 px2 <= x2 + 10 and py2 <= y2 + 10)
                if phone_in_bbox and conf >= 0.4:
                    print(f'[PHONE] Person {person.track_id}: phone inside person bbox, conf={conf:.2f}')
                    return True, float(conf)

        # ── Strategy B: Skeleton heuristic ──────────────────────────────────────
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

                # Both wrists at lap/desk level, close together
                # head_down raises confidence but is NOT a hard requirement
                if len(wrist_pts) == 2:
                    both_low    = all(w[1] > lap_thresh for w in wrist_pts)
                    hands_close = np.linalg.norm(wrist_pts[0] - wrist_pts[1]) / bbox_h < 0.30
                    if both_low and hands_close:
                        if not (_book_near(wrist_pts[0]) or _book_near(wrist_pts[1])):
                            conf = 0.7 if head_is_down else 0.55
                            print(f'[PHONE] Person {person.track_id}: cupped-hands lap, head_down={head_is_down}')
                            return True, conf

                elif len(wrist_pts) == 1:
                    lone  = wrist_pts[0]
                    other = right_wrist if _wrist_ok(left_wrist, 0.5) else left_wrist
                    other_hidden = (
                        other is None or len(other) < 3 or
                        other[2] < 0.25 or (other[0] == 0.0 and other[1] == 0.0)
                    )
                    # Single wrist at lap level + other wrist hidden
                    # head_down not required, but raises confidence
                    if lone[1] > lap_thresh and other_hidden and not _book_near(lone):
                        conf = 0.6 if head_is_down else 0.45
                        print(f'[PHONE] Person {person.track_id}: single-hand lap, head_down={head_is_down}')
                        return True, conf

                # Hand-to-ear (phone call posture)
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
        except Exception:
            pass
        return False, 0.0

    # ── Fight Detection ───────────────────────────────────────────────────────
    def add_frame_for_fight_detection(self, frame: np.ndarray):
        if self._fight_detector_3dcnn is not None:
            self._fight_detector_3dcnn.add_frame(frame)

    def _skeleton_fight_score(self, a: 'TrackedPerson', b: 'TrackedPerson') -> float:
        """
        Pure skeleton heuristic fight score [0..1] — works without any CNN.
        Returns > 0 when physical-contact signals are present between person a and b.
        """
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox

        scale_ref = max(
            ((ay2 - ay1) + (by2 - by1)) / 2.0,
            ((ax2 - ax1) + (bx2 - bx1)) / 2.0,
        )
        if scale_ref <= 0:
            return 0.0

        ca = np.array([(ax1 + ax2) / 2, (ay1 + ay2) / 2])
        cb = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])
        center_dist = np.linalg.norm(ca - cb) / scale_ref

        boxes_overlap = not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

        score = 0.0

        # Bounding-box overlap is the strongest physical-contact signal
        if boxes_overlap:
            score += 0.5
        elif center_dist < self.FIGHT_PROXIMITY_RATIO:
            score += max(0.0, (self.FIGHT_PROXIMITY_RATIO - center_dist) / self.FIGHT_PROXIMITY_RATIO) * 0.3

        # Wrist proximity between the two people
        if (a.keypoints is not None and b.keypoints is not None and
                len(a.keypoints) > 10 and len(b.keypoints) > 10):
            try:
                wrist_limit = scale_ref * self.FIGHT_WRIST_PROXIMITY_RATIO
                min_wrist_d = float('inf')
                for ai in (9, 10):
                    for bi in (9, 10):
                        aw = a.keypoints[ai]
                        bw = b.keypoints[bi]
                        # Only use wrists with reasonable confidence
                        if len(aw) >= 3 and len(bw) >= 3 and aw[2] > 0.2 and bw[2] > 0.2:
                            d = np.linalg.norm(aw[:2] - bw[:2])
                            min_wrist_d = min(min_wrist_d, d)
                if min_wrist_d < wrist_limit:
                    score += 0.4 * (1.0 - min_wrist_d / wrist_limit)
                # Also check if one person's wrist is inside the other's bbox
                for ai in (9, 10):
                    aw = a.keypoints[ai]
                    if len(aw) >= 3 and aw[2] > 0.2:
                        if bx1 <= aw[0] <= bx2 and by1 <= aw[1] <= by2:
                            score += 0.3  # A's wrist is inside B's body
                for bi in (9, 10):
                    bw = b.keypoints[bi]
                    if len(bw) >= 3 and bw[2] > 0.2:
                        if ax1 <= bw[0] <= ax2 and ay1 <= bw[1] <= ay2:
                            score += 0.3  # B's wrist is inside A's body
            except Exception:
                pass

        return min(score, 1.0)

    def _detect_fighting_pairwise(self, frame: np.ndarray) -> List[Tuple[int, int, float, Dict]]:
        """
        Pairwise fight detection.
        Primary: skeleton heuristic (always available).
        Secondary: 3D CNN on cropped pair region (when available).
        Returns list of (person_a_id, person_b_id, confidence, fight_info).
        """
        if len(self.tracked_people) < 2:
            self._fight_candidate_ids = set()
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

                candidate_pairs.append((a.track_id, b.track_id, dist, scale_ref))

        self._fight_candidate_ids = {tid for a, b, _, _ in candidate_pairs for tid in (a, b)}

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
                    'last_cnn_time': 0.0,
                }
            state = self.fight_pairs[pair_key]
            now   = time.time()

            # Skeleton path: score >= 0.5 triggers a confirmation tick
            if skel_score >= 0.5:
                if now - state['last_skel_time'] > 0.2:
                    state['skel_confirms'] += 1
                    state['last_skel_time'] = now

                if state['skel_confirms'] >= self.FIGHT_SKELETON_CONFIRMS:
                    print(f'[FIGHT] pair {pair_key}: SKELETON FIGHT confirmed! score={skel_score:.2f}')
                    fight_results.append((a_id, b_id, min(skel_score, 0.85), {
                        'person_a_id': a_id, 'person_b_id': b_id,
                        'confidence': skel_score, 'trigger': 'skeleton',
                        'distance': dist, 'confirmations': state['skel_confirms'],
                    }))
                    continue   # no need to also run CNN for this pair
            else:
                if now - state['last_skel_time'] > 1.5:
                    state['skel_confirms'] = max(0, state['skel_confirms'] - 1)

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
        elif track_id in self._fight_candidate_ids:
            pass  # contact suspected, hold state
        else:
            raw_head = self._calculate_head_pose(person)
            is_phone, phone_conf = self._detect_phone_usage(person, phone_detections, raw_head, book_detections)
            if is_phone:
                person.behavior_history.append('using_phone')
            else:
                is_eating, _ = self._detect_eating(person, food_detections)
                if is_eating:
                    person.behavior_history.append('eating_food')
                else:
                    person.behavior_history.append('focused' if raw_head == 'focused' else 'distracted')

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
            confidence = fight_confidence or 0.8
            is_alert, is_distracted = True, False
        else:
            confidence = 0.8
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
        self._load_models()

    def _load_models(self):
        try:
            from ultralytics import YOLO
            self.yolo_model   = YOLO('yolo11s-pose.pt')
            self.object_model = YOLO('yolo11s.pt')
            print('[OK] Pose + object detection models loaded')
        except Exception as e:
            print(f'[WARN] Model load failed: {e}')

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
                        ts, frame = self.frame_buffer.pop()
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
        """Update engine, run fight detection once, evaluate all persons."""
        for tid, x1, y1, x2, y2, conf, kp in person_tracks:
            self.behavior_engine.update_person(tid, (x1, y1, x2, y2), kp, timestamp)

        self.behavior_engine.add_frame_for_fight_detection(frame)
        self.behavior_engine.cleanup_stale(timestamp)

        # Fight detection runs once per frame
        fight_results = self.behavior_engine._detect_fighting_pairwise(frame)
        fight_map = {}
        for a_id, b_id, conf, info in fight_results:
            fight_map[a_id] = (conf, info)
            fight_map[b_id] = (conf, info)

        results = []
        for tid in self.behavior_engine.tracked_people:
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
        self.server_url      = server_url
        self.alert_cooldown  = alert_cooldown
        self.whatsapp_admin  = whatsapp_admin or os.environ.get('ADMIN_WHATSAPP', '')
        self.yolo_model      = None
        self.face_recognizer = None
        self.known_students  = []
        self.last_alert_time: dict = defaultdict(float)
        self.running         = False
        self.thread          = None
        self._detect_engine  = TemporalBehaviorEngine()
        self.processor       = ProductionStreamProcessor(process_fps=10)
        self._load_models()

    def _load_models(self):
        try:
            from ultralytics import YOLO
            self.yolo_model   = YOLO('yolo11s-pose.pt')
            self.object_model = YOLO('yolo11s.pt')
            print('[OK] Models loaded')
        except Exception as e:
            print(f'[WARN] YOLO: {e}')
            self.yolo_model = self.object_model = None

    def _load_known_students(self):
        import requests as _req
        try:
            r = _req.get(f'{self.server_url}/api/students/encodings/', timeout=5, verify=False)
            self.known_students = r.json()
            print(f'[OK] {len(self.known_students)} student encodings loaded')
        except Exception as e:
            print(f'[WARN] Could not load students: {e}')

    def detect(self, frame) -> List[Dict]:
        """Detect behaviors in a single frame. Returns list of detection dicts."""
        if self.yolo_model is None:
            return []
        try:
            timestamp     = time.time()
            person_tracks = self.processor._parse_pose_detections(frame)
            phone_dets, food_dets, book_dets = self.processor._parse_object_detections(frame)

            for tid, x1, y1, x2, y2, conf, kp in person_tracks:
                self._detect_engine.update_person(tid, (x1, y1, x2, y2), kp, timestamp)
            self._detect_engine.cleanup_stale(timestamp)
            self._detect_engine.add_frame_for_fight_detection(frame)

            fight_results = self._detect_engine._detect_fighting_pairwise(frame)
            fight_map = {}
            for a_id, b_id, conf, info in fight_results:
                fight_map[a_id] = (conf, info)
                fight_map[b_id] = (conf, info)

            detections = []
            for tid in self._detect_engine.tracked_people:
                f_conf, f_info = fight_map.get(tid, (0.0, None))
                override = (tid in fight_map, f_conf, f_info) if tid in fight_map else None
                det = self._detect_engine.evaluate_person(
                    tid, phone_dets, food_dets, frame=None,
                    fight_override=override, book_detections=book_dets)
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
                    'send_whatsapp': detection['is_alert'],
                },
                timeout=10, verify=False)
            print(f"[INCIDENT] {detection['label']} | {tag} | {resp.status_code}")
        except Exception as e:
            print(f'[ERROR] report_incident: {e}')

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
            encs = self.face_recognizer.face_encodings(rgb, num_jitters=1, model='small')
            if not encs:
                return None, 'Unknown', ''
            det    = encs[0]
            best_d, best = 1.0, None
            for s in self.known_students:
                try:
                    d = self.face_recognizer.face_distance([np.array(json.loads(s['encoding']))], det)[0]
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