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
    behavior_history: deque = field(default_factory=lambda: deque(maxlen=20))  # Increased for better smoothing
    last_seen: float = 0.0
    last_final_behavior: str = 'focused'
    keypoint_history: deque = field(default_factory=lambda: deque(maxlen=30))  # Increased for motion analysis
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


# ── Temporal Behavior Engine ─────────────────────────────────────────────────
class TemporalBehaviorEngine:
    # Phone detection thresholds
    PHONE_LAP_HEIGHT_FRACTION    = 0.60   # Hands below 60% of bbox height for lap detection
    PHONE_CUPPED_SPREAD_MAX      = 0.22   # Wrists close together (cupped / lap phone)
    PHONE_SINGLE_HAND_Y_MIN      = 0.50   # Single-hand phone — wider vertical window
    PHONE_SINGLE_HAND_Y_MAX      = 0.90
    PHONE_WRIST_PROXIMITY_RATIO  = 0.45   # Phone center within 45% bbox_h of wrist
    PHONE_SIZE_MAX_RATIO         = 0.85   # Phone max dimension < 85% of person bbox_h
    PHONE_SIZE_MIN_RATIO         = 0.03   # Phone min dimension > 3% of person bbox_h
    PHONE_BBOX_EXPAND            = 0.20   # Expand person bbox by 20% when checking containment

    # Strategy-1 (YOLO) spread veto used to fire on a flat 0.32 with no
    # escape hatch — vetoing real one-handed phone detections whenever the
    # off-hand happened to be visible and naturally spread away from the
    # body. It now reuses PHONE_TWO_HAND_SPREAD_VETO below as the trigger
    # point (consistent with Strategy 2's own definition of "spread enough
    # to suspect writing") and pairs it with a tight-grip + idle-far-hand
    # override, so the FIX is the override carve-out, not a loosened
    # absolute number that would also blind it to genuine two-hand writing.
    PHONE_TIGHT_GRIP_RATIO       = 0.18   # phone within this of a WRIST = hand is gripping it

    # Strategy-2 (pose-only fallback) two-hand writing-posture veto. Also
    # reused by Strategy 1 (see above) as the spread-veto trigger point.
    PHONE_TWO_HAND_SPREAD_VETO   = 0.30
    # When both wrists ARE visible but one sits far off to the side (resting,
    # not gripping anything) while the other stays centered, treat it as a
    # one-handed hold rather than forcing the symmetric two-hand veto.
    PHONE_IDLE_HAND_OFFSET_RATIO = 0.35
    PHONE_SINGLE_HAND_CENTER_RATIO = 0.28  # phone-holding hand stays roughly centered
    WRITING_DESK_Y_MIN           = 0.70   # Writing starts lower (clean separation)
    LOW_CONFIDENCE_THRESHOLD     = 0.5    # Confidence threshold for keypoints
    HEAD_DOWN_CONSECUTIVE_FRAMES = 3

    # Eating detection
    EATING_HAND_TO_MOUTH_THRESHOLD = 0.15

    # Alert smoothing — 6 frames @ 10 FPS = 0.6 s continuous detection to confirm
    ALERT_CONFIRM_FRAMES       = 6
    NORMAL_CONFIRM_FRAMES      = 5
    # Hand raise is a transient gesture — confirm faster to avoid missing it
    HAND_RAISE_CONFIRM_FRAMES  = 3

    # Hand raise keypoint confidence thresholds
    # Raised arms are an unusual YOLO training pose → real-world confidence is 0.35–0.5
    HAND_RAISE_WRIST_CONF      = 0.35   # was 0.6 — far too strict for raised-arm poses
    HAND_RAISE_SHOULDER_CONF   = 0.40   # was 0.6 — shoulder can be partially occluded
    HAND_RAISE_ELBOW_CONF      = 0.35   # new: elbow fallback when wrist is occluded

    def __init__(self):
        self.tracked_people: Dict[int, TrackedPerson] = {}
        self.cleanup_threshold = 2.0
        self.low_confidence_counters: Dict[int, int] = {}
        self.lock = threading.Lock()  # Added thread safety

    def update_person(self, track_id: int, bbox: Tuple, keypoints: Optional[np.ndarray], timestamp: float):
        with self.lock:  # Thread-safe update
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
        with self.lock:  # Thread-safe cleanup
            stale = [tid for tid, p in self.tracked_people.items()
                     if current_time - p.last_seen > self.cleanup_threshold]
            for tid in stale:
                del self.tracked_people[tid]
                self.low_confidence_counters.pop(tid, None)

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

            head_down_suspected = (not nose_ok) or (not left_eye_ok and not right_eye_ok)

            if head_down_suspected:
                self.low_confidence_counters[tid] = self.low_confidence_counters.get(tid, 0) + 1
                return 'head_down' if self.low_confidence_counters[tid] >= self.HEAD_DOWN_CONSECUTIVE_FRAMES else 'focused'
            else:
                self.low_confidence_counters[tid] = 0

            if left_eye_ok and right_eye_ok:
                inter_eye = np.linalg.norm(left_eye[:2] - right_eye[:2])
                if inter_eye > 0.1:
                    eyes_center_x = (left_eye[0] + right_eye[0]) / 2.0
                    yaw_ratio = abs(nose[0] - eyes_center_x) / inter_eye
                    if yaw_ratio > 0.6:  # Very large turn to be distracted
                        return 'looking_away'

                    drop_ratio = (nose[1] - (left_eye[1] + right_eye[1]) / 2) / inter_eye
                    if drop_ratio > 0.6:  # Very low head for head_down
                        return 'head_down'

                    bbox_h = person.bbox[3] - person.bbox[1]
                    if bbox_h > 10 and inter_eye < bbox_h * 0.06:
                        return 'looking_away'
            else:
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
            if np.linalg.norm(np.array(pt) - bc) / bbox_h < 0.3:  # Larger area for book
                return True
        return False

    @staticmethod
    def _hands_spread_writing_posture(wrist_pts, bbox_h: float) -> bool:
        """Two hands low but horizontally separated — typical writing, not phone."""
        if len(wrist_pts) != 2 or bbox_h <= 0:
            return False
        spread = np.linalg.norm(wrist_pts[0] - wrist_pts[1]) / bbox_h
        return spread > 0.3

    @staticmethod
    def _wrist_lateral_offset(wrist, x1: float, x2: float) -> float:
        """How far a wrist is from body center, normalized by bbox width."""
        bbox_w = max(x2 - x1, 1.0)
        center_x = (x1 + x2) / 2.0
        return abs(wrist[0] - center_x) / bbox_w

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
                    nose[2] >= 0.5 and left_eye[2] >= 0.5 and right_eye[2] >= 0.5):
                inter_eye = np.linalg.norm(left_eye[:2] - right_eye[:2])
                if inter_eye > 0.1:
                    drop_ratio = (nose[1] - (left_eye[1] + right_eye[1]) / 2) / inter_eye
                    return drop_ratio > 0.4
        except Exception:
            pass
        return False

    # ── Wrist Motion Analysis for Writing vs Phone ────────────────────────────
    def _calculate_wrist_motion_variance(self, person: TrackedPerson) -> Tuple[bool, float]:
        """
        Calculate wrist motion variance over time independently per wrist.
        Writing = sustained high variance (pen movement), Phone = lower or burst variance.
        Returns (is_definitive_writing, confidence).

        Note: scrolling a phone produces similar wrist motion to writing. We therefore
        only return True when motion variance is VERY high (>400, not 150) AND head is down,
        to avoid suppressing phone detection.
        """
        if len(person.keypoint_history) < 10:
            return False, 0.0

        per_wrist: Dict[int, list] = {9: [], 10: []}
        for _, kp in person.keypoint_history:
            if kp is not None and len(kp) > 10:
                for idx in (9, 10):
                    w = kp[idx]
                    if len(w) >= 3 and w[2] >= 0.5 and w[0] != 0.0:
                        per_wrist[idx].append(w[:2].copy())

        variances = []
        for positions in per_wrist.values():
            if len(positions) >= 8:
                arr = np.array(positions)
                variances.append(float(np.var(arr[:, 0]) + np.var(arr[:, 1])))

        if not variances:
            return False, 0.0

        total_variance = float(np.mean(variances))
        # Raised threshold: >400 (was 150) to avoid phone-scrolling false positives.
        # At 150 a student scrolling Instagram would suppress phone detection every time.
        is_definitive_writing = total_variance > 400
        confidence = min(total_variance / 800.0, 0.95)
        return is_definitive_writing, confidence

    def _is_writing_posture(self, person: TrackedPerson,
                            head_pose: str = '',
                            book_detections: Optional[List[Tuple]] = None) -> Tuple[bool, float]:
        """
        Definitive writing detection — returns True only when we are highly confident.
        Deliberately conservative so we do NOT suppress phone detection on ambiguous cases.

        Writing is definitively confirmed when:
          - A book/notebook is detected near the hands, OR
          - Wrist motion variance is very high (>400 px²) AND head is down

        Suggestive writing (single-hand at desk, spread hands) returns False here —
        those cases are handled in _detect_phone_usage as a confidence reducer, not blocker.
        """
        book_detections = book_detections or []

        if not head_pose:
            head_pose = self._calculate_head_pose(person)
        head_is_down = self._head_down_like(person, head_pose)

        kp = person.keypoints
        x1, y1, x2, y2 = person.bbox
        bbox_h = y2 - y1
        if kp is None or kp.size == 0 or len(kp) <= 10 or bbox_h <= 0:
            return False, 0.0

        wrists = []
        for idx in (9, 10):
            w = kp[idx]
            if len(w) >= 3 and w[2] >= 0.5 and w[0] != 0.0:
                wrists.append(w[:2])

        if not wrists:
            return False, 0.0

        # Definitive signal 1: book/notebook detected near hands
        low_wrists = [w for w in wrists if w[1] > y1 + bbox_h * 0.60]
        if low_wrists:
            book_nearby = any(self._point_near_book(w, bbox_h, book_detections) for w in low_wrists)
            if book_nearby:
                return True, 0.85

        # Definitive signal 2: very high wrist motion + head down
        # (raised variance threshold in _calculate_wrist_motion_variance to 400)
        motion_writing, motion_conf = self._calculate_wrist_motion_variance(person)
        if motion_writing and head_is_down:
            return True, motion_conf

        return False, 0.0

    # ── Phone bbox containment check ──────────────────────────────────────────
    def _phone_in_person_bbox(self, px1: float, py1: float, px2: float, py2: float,
                              person: TrackedPerson) -> bool:
        """Return True if the phone bbox overlaps the (expanded) person bbox."""
        bx1, by1, bx2, by2 = person.bbox
        bh, bw = by2 - by1, bx2 - bx1
        exp_x = bw * self.PHONE_BBOX_EXPAND
        exp_y = bh * self.PHONE_BBOX_EXPAND
        epx1, epy1 = bx1 - exp_x, by1 - exp_y
        epx2, epy2 = bx2 + exp_x, by2 + exp_y
        # IoU-style overlap check
        ix1, iy1 = max(px1, epx1), max(py1, epy1)
        ix2, iy2 = min(px2, epx2), min(py2, epy2)
        return ix1 < ix2 and iy1 < iy2

    # ── Single-hand phone scoring (pose-only fallback) ─────────────────────────
    def _score_single_hand_phone(self, wrist: np.ndarray, head_is_down: bool,
                                  motion_conf: float, bbox_h: float, y1: float,
                                  x1: float, x2: float) -> Tuple[bool, float]:
        """
        Score a ONE-handed phone hold from a single wrist keypoint.

        Used in two situations:
          1. Only one wrist keypoint is confident at all (the other hand is
             occluded, off-screen, or simply at the student's side where the
             pose model doesn't track it well).
          2. Both wrists are confident, but one is clearly idle/resting off
             to the side (see the caller) — only the active wrist is passed
             in here.

        Deliberately stricter than the two-hand cupped case (threshold 0.58
        vs 0.55, lower base score) since there is only one corroborating
        hand position instead of two. Still conservative: requires the wrist
        to be in the lap/desk zone before any score accrues at all.
        """
        y_frac = (wrist[1] - y1) / bbox_h
        if not (self.PHONE_SINGLE_HAND_Y_MIN < y_frac < self.PHONE_SINGLE_HAND_Y_MAX):
            return False, 0.0

        lateral_offset = self._wrist_lateral_offset(wrist, x1, x2)

        score = 0.28  # lower base than two-hand cupped (0.30) — one fewer signal available
        if lateral_offset < self.PHONE_SINGLE_HAND_CENTER_RATIO:
            score += 0.18   # held close to centerline, not reaching sideways for something else
        if head_is_down:
            score += 0.14
        if motion_conf < 0.10:
            score += 0.10   # idle hold rather than active gesture

        if score >= 0.58:
            final_conf = min(score, 0.68)
            print(f'[PHONE] S2 single-hand PASS score={score:.2f} '
                  f'y_frac={y_frac:.2f} lateral={lateral_offset:.2f} '
                  f'head_down={head_is_down} motion_conf={motion_conf:.2f}')
            return True, final_conf

        print(f'[PHONE] S2 single-hand REJECT score={score:.2f} (threshold 0.58) '
              f'y_frac={y_frac:.2f} lateral={lateral_offset:.2f}')
        return False, 0.0


    # ── Phone Detection — Industry-Grade Multi-Signal Fusion ──────────────────
    def _detect_phone_usage(self, person: TrackedPerson,
                            phone_detections: List[Tuple],
                            head_pose: str,
                            book_detections: Optional[List[Tuple]] = None) -> Tuple[bool, float]:
        """
        Industry-grade phone detection using multi-signal fusion.

        Detection pipeline:
          1. YOLO object detection (primary):
             - Phone bbox must OVERLAP person's expanded bbox (containment check).
             - Phone center must be within PHONE_WRIST_PROXIMITY_RATIO*bbox_h of
               any wrist OR elbow keypoint (wider net, lower conf threshold).
             - Size filter relaxed: max 85%, min 3% of bbox_h.
             - Writing gate only applied when book is confirmed near hands
               (pure motion is NOT a veto — scrolling looks like writing).
             - Fused confidence = YOLO conf + proximity bonus + head bonus.
          2. Pose heuristic fallback (secondary, when YOLO has no phone):
             - Two cupped wrists in lap (both low, close together, no book).
             - Single wrist in lap zone, centered, head down, low motion —
               covers the common one-handed-phone case, including when the
               other hand is visible but idle/resting off to the side.
        """
        x1, y1, x2, y2 = person.bbox
        bbox_h = y2 - y1
        bbox_w = x2 - x1
        if bbox_h <= 0 or bbox_w <= 0:
            return False, 0.0

        kp = person.keypoints
        book_detections = book_detections or []
        head_is_down = (head_pose == 'head_down') or self._head_down_like(person, head_pose)
        center_x = (x1 + x2) / 2.0

        def _wrist_ok(w, min_conf=0.45) -> bool:
            return w is not None and len(w) >= 3 and w[2] >= min_conf and w[0] != 0.0

        def _book_near(pt) -> bool:
            return self._point_near_book(pt, bbox_h, book_detections)

        # Collect valid body keypoints: wrists + elbows (wider net catches more views)
        # idx: 7=left_elbow, 8=right_elbow, 9=left_wrist, 10=right_wrist
        body_kps: List[Tuple[int, np.ndarray]] = []
        if kp is not None and len(kp) > 10:
            for idx in (7, 8, 9, 10):
                if idx < len(kp):
                    pt = kp[idx]
                    if _wrist_ok(pt, 0.40):
                        body_kps.append((idx, pt[:2].copy()))

        # ── Strategy 1: YOLO-detected phone object ────────────────────────────
        phone_found = False
        best_conf = 0.0

        if phone_detections:
            for (px1, py1, px2, py2, conf) in phone_detections:
                if conf < 0.25:  # Lowered from 0.35
                    continue

                ph, pw = py2 - py1, px2 - px1

                # Fix 3: Relaxed size filter (was 0.6 max / 0.05 min)
                if max(pw, ph) / bbox_h > self.PHONE_SIZE_MAX_RATIO:
                    continue  # Too large — probably another person
                if min(pw, ph) / bbox_h < self.PHONE_SIZE_MIN_RATIO:
                    continue  # Too small — noise

                # Fix 6: Containment — phone must overlap person's expanded bbox
                if not self._phone_in_person_bbox(px1, py1, px2, py2, person):
                    continue

                pc = np.array([(px1 + px2) / 2.0, (py1 + py2) / 2.0])

                # Fix 2: Proximity to wrists OR elbows; threshold widened to 0.45
                best_proximity = float('inf')
                best_kp_idx = -1
                for kp_idx, kp_pt in body_kps:
                    dist_ratio = float(np.linalg.norm(kp_pt - pc) / bbox_h)
                    if dist_ratio < best_proximity:
                        best_proximity = dist_ratio
                        best_kp_idx = kp_idx

                if best_proximity > self.PHONE_WRIST_PROXIMITY_RATIO:
                    continue  # Phone too far from any hand/elbow

                # Writing gate: veto when book is confirmed near hands OR when
                # both wrists are spread in writing posture (strong structural signal).
                is_writing, _ = self._is_writing_posture(person, head_pose, book_detections)
                if is_writing:
                    print(f'[PHONE] Person {person.track_id}: YOLO phone present but '
                          f'book-confirmed writing posture — skipping')
                    continue

                # Spread veto: both wrists clearly apart AND neither one is
                # tightly gripping the phone = writing, not phone.
                # Only apply when both wrists are detected with reasonable confidence.
                #
                # FIX: a tight grip on the near wrist (best_proximity very small,
                # matched keypoint is a WRIST not an elbow) combined with the FAR
                # wrist sitting well off the body centerline (genuinely idle/
                # resting, not a second hand symmetrically engaged at the desk)
                # is direct evidence of a one-handed phone hold — the previous
                # flat 0.32 spread veto had no escape hatch for this extremely
                # common pose. Requiring BOTH the tight grip AND an idle far
                # hand (rather than tight grip alone) avoids re-opening a
                # different false positive: a pen/eraser near one wrist during
                # genuine two-hand writing, where the far wrist is still
                # forward and engaged rather than resting at the student's side.
                spread_veto = False
                if kp is not None and len(kp) > 10:
                    lw, rw = kp[9], kp[10]
                    if (_wrist_ok(lw, 0.40) and _wrist_ok(rw, 0.40)):
                        wrist_spread = float(
                            np.linalg.norm(lw[:2] - rw[:2]) / bbox_h)
                        if wrist_spread > self.PHONE_TWO_HAND_SPREAD_VETO:
                            tightly_gripped = (
                                best_kp_idx in (9, 10)
                                and best_proximity < self.PHONE_TIGHT_GRIP_RATIO
                            )
                            far_wrist = rw if best_kp_idx == 9 else lw
                            far_is_idle = (
                                _wrist_ok(far_wrist, 0.40)
                                and self._wrist_lateral_offset(far_wrist[:2], x1, x2)
                                    > self.PHONE_IDLE_HAND_OFFSET_RATIO
                            )
                            if not (tightly_gripped and far_is_idle):
                                spread_veto = True
                if spread_veto:
                    print(f'[PHONE] Person {person.track_id}: YOLO phone present but '
                          f'wrists spread in writing posture — skipping')
                    continue

                # Fix 7: Multi-signal confidence fusion
                prox_bonus  = max(0.0, (self.PHONE_WRIST_PROXIMITY_RATIO - best_proximity)
                                  / self.PHONE_WRIST_PROXIMITY_RATIO) * 0.15
                head_bonus  = 0.08 if head_is_down else 0.0
                wrist_bonus = 0.05 if best_kp_idx in (9, 10) else 0.0  # wrist > elbow
                fused_conf  = min(conf + prox_bonus + head_bonus + wrist_bonus, 0.97)
                if fused_conf > best_conf:
                    best_conf = fused_conf
                    phone_found = True
                    print(f'[PHONE] [OK] Person {person.track_id}: YOLO+fusion '
                          f'conf={fused_conf:.2f} (yolo={conf:.2f}, prox={best_proximity:.2f}, '
                          f'kp={best_kp_idx}, head_down={head_is_down})')

        if phone_found:
            return True, best_conf

        # ── Strategy 2: Multi-signal scoring (YOLO found no phone) ──────────────
        #
        # Design principle — PRECISION OVER RECALL:
        #   Pose-only heuristics must be conservative. A missed phone is recoverable
        #   (YOLO handles recall via Strategy 1). A false positive on a focused student
        #   erodes trust in the whole system.
        #
        # What went wrong before (false positives on writing students):
        #   - Heuristic B (single centered wrist in lap) fired on any pen-holding hand.
        #   - Heuristic C (mid-body wrist) fired on virtually every writing student
        #     because writing-on-desk wrists sit at exactly 35–75% bbox height.
        #   - No veto for spread wrists (writing = hands spread; phone = hands cupped).
        #   - No veto for high wrist-motion variance (writing = constant pen movement).
        #
        # What went wrong AFTER that fix (false negatives — real phones missed):
        #   - Single-wrist detection was removed entirely instead of being fixed,
        #     so `if len(wrist_pts) < 2: return False` rejected every one-handed
        #     phone user — by far the common case, since the off-hand is usually
        #     resting at the student's side or simply outside the pose model's
        #     confident range.
        #   - Veto V2 used a flat two-wrist spread check with no distinction
        #     between "two hands genuinely spread for writing" and "one centered
        #     hand holding a phone while the other idle hand happens to be
        #     spread away" — the latter is a completely normal one-handed-phone
        #     pose that got vetoed too.
        #
        # Current approach — vetoes first, then route to the right scoring path:
        #
        #   Veto V1: Book/notebook detected near any visible wrist → definitively writing
        #   Veto V2: Active wrist-motion variance (pen-on-paper)   → definitively writing
        #            (checked up front since it applies to either path below)
        #
        #   Then:
        #     • 1 wrist visible             → _score_single_hand_phone() on it
        #     • 2 wrists, one clearly idle   → _score_single_hand_phone() on the
        #       active one (idle = lateral offset > 0.35; active = offset < 0.28)
        #     • 2 wrists, both engaged       → symmetric two-hand path:
        #         Veto: spread > 0.30 × bbox_h (notebook posture)
        #         Score (threshold 0.55): +0.22 cupped-in-lap, +0.12 head down,
        #         +0.10 low motion, −0.12 moderately spread (0.22–0.30)
        #
        #   Single-hand scoring is intentionally stricter (threshold 0.58 — see
        #   _score_single_hand_phone) since there's only one corroborating hand
        #   position instead of two.

        if kp is not None and kp.size > 0 and len(kp) > 10:
            try:
                left_wrist  = kp[9]
                right_wrist = kp[10]
                lap_thresh  = y1 + bbox_h * self.PHONE_LAP_HEIGHT_FRACTION  # 60% from top
                seat_bottom = y1 + bbox_h * 0.90  # below this = below the seat

                wrist_pts = []
                for w in (left_wrist, right_wrist):
                    if _wrist_ok(w, 0.45):
                        wrist_pts.append(w[:2].copy())

                if not wrist_pts:
                    # No wrist visible at all — nothing to reason from.
                    return False, 0.0

                # ── Veto V1: Book/notebook near any visible hand ──────────────
                if any(_book_near(w) for w in wrist_pts):
                    return False, 0.0

                # ── Veto V2: Active wrist motion = pen movement ───────────────
                # is_writing_motion is True when variance > 400 px² — i.e., the wrist
                # has been tracing wide arcs consistent with pen-on-paper movement.
                # Checked up front since it applies regardless of how many wrists
                # are visible.
                is_writing_motion, motion_conf = self._calculate_wrist_motion_variance(person)
                if is_writing_motion:
                    return False, 0.0

                # ── Only one wrist visible ─────────────────────────────────────
                # Can't use the two-wrist spread discriminator with just one
                # point, so reason from this wrist alone via the dedicated
                # single-hand scorer instead of rejecting outright.
                if len(wrist_pts) == 1:
                    return self._score_single_hand_phone(
                        wrist_pts[0], head_is_down, motion_conf, bbox_h, y1, x1, x2)

                # ── Both wrists visible — check for an asymmetric one-handed
                # hold before forcing the symmetric two-hand path below. ──────
                # Holding a phone in one hand while the other rests naturally
                # at the student's side produces exactly this pattern: one
                # wrist near the body centerline, one wrist well off to the
                # side, and a wide overall spread that a flat threshold alone
                # would mistake for two-hand writing.
                spread = float(np.linalg.norm(wrist_pts[0] - wrist_pts[1]) / bbox_h)
                offsets = [self._wrist_lateral_offset(w, x1, x2) for w in wrist_pts]
                active_idx = int(np.argmin(offsets))
                idle_idx = 1 - active_idx
                one_hand_idle = (
                    spread > self.PHONE_TWO_HAND_SPREAD_VETO
                    and offsets[idle_idx] > self.PHONE_IDLE_HAND_OFFSET_RATIO
                    and offsets[active_idx] < self.PHONE_SINGLE_HAND_CENTER_RATIO
                )
                if one_hand_idle:
                    return self._score_single_hand_phone(
                        wrist_pts[active_idx], head_is_down, motion_conf, bbox_h, y1, x1, x2)

                # ── Veto: Wrists spread apart — genuine two-hand writing posture ──
                # Two hands on a notebook are always spread; two hands holding a
                # phone are always cupped. This is the strongest discriminator
                # once the asymmetric one-idle-hand case above has been ruled out.
                if spread > self.PHONE_TWO_HAND_SPREAD_VETO:
                    return False, 0.0

                # ── Multi-signal confidence scoring (symmetric two-hand case) ──
                score = 0.30  # base — must be lifted by corroborating signals

                # Signal S1 (strongest): Both wrists cupped in lap zone
                both_in_lap = all(lap_thresh < w[1] < seat_bottom for w in wrist_pts)
                if spread < self.PHONE_CUPPED_SPREAD_MAX and both_in_lap:
                    score += 0.22
                    print(f'[PHONE] Person {person.track_id}: S2-cupped '
                          f'spread={spread:.2f} score={score:.2f}')

                # Signal S2: Head tilted down toward lap
                if head_is_down:
                    score += 0.12

                # Signal S3: Very low wrist motion (idle phone-holding vs active writing)
                # motion_conf = total_variance / 800; < 0.10 → variance < 80 px² → very still
                if motion_conf < 0.10:
                    score += 0.10

                # Penalty: moderately spread wrists (partial writing indicator)
                if spread > 0.22:
                    score -= 0.12

                if score >= 0.55:
                    final_conf = min(score, 0.72)
                    print(f'[PHONE] Person {person.track_id}: S2 multi-signal PASS '
                          f'score={score:.2f} spread={spread:.2f} '
                          f'head_down={head_is_down} motion_conf={motion_conf:.2f}')
                    return True, final_conf
                else:
                    print(f'[PHONE] Person {person.track_id}: S2 multi-signal REJECT '
                          f'score={score:.2f} (threshold 0.55)')

            except Exception as exc:
                print(f'[PHONE] Heuristic error: {exc}')

        return False, 0.0

    # ── Hand Raise Detection ─────────────────────────────────────────────────
    def _detect_hand_raise(self, person: TrackedPerson) -> Tuple[bool, float]:
        """
        Industry-grade hand raise detection using a 3-tier signal cascade:

        Tier 1 — Wrist vs shoulder (primary):  wrist[y] < shoulder[y] - thresh
          • Thresholds lowered from 0.6 → 0.35/0.40 because raised-arm poses are
            under-represented in YOLO training data; real cameras yield 0.35–0.50.

        Tier 2 — Elbow vs shoulder (fallback when wrist is occluded):
          • Elbow above shoulder is a strong partial-raise signal.

        Tier 3 — Wrist above nose/eyes (very high raise, absolute):
          • When a hand is raised completely above the head, the wrist Y will be
            less than the nose Y — catch this with a permissive conf threshold.

        Confidence is boosted proportionally to how far above the shoulder the
        wrist is, giving the smoothing engine a strong signal.
        """
        if person.keypoints is None or person.keypoints.size == 0:
            return False, 0.0
        try:
            kp = person.keypoints
            x1, y1, x2, y2 = person.bbox
            bbox_h = y2 - y1
            bbox_w = x2 - x1
            if bbox_h <= 0 or bbox_w <= 0 or len(kp) < 11:
                return False, 0.0

            def _kp_ok(pt, min_conf: float) -> bool:
                return (pt is not None and len(pt) >= 3
                        and float(pt[2]) >= min_conf and float(pt[0]) != 0.0)

            left_shoulder  = kp[5]
            right_shoulder = kp[6]
            left_elbow     = kp[7]
            right_elbow    = kp[8]
            left_wrist     = kp[9]
            right_wrist    = kp[10]
            # Nose for absolute height check (Tier 3)
            nose = kp[0] if len(kp) > 0 else None

            # Minimum raise = 8% of bbox height above the shoulder (same as before)
            raise_thresh_min = bbox_h * 0.08

            best_conf = 0.0
            raised    = False

            # --- Tier 1: wrist above shoulder (both sides) ---
            for wrist, shoulder, side in (
                (left_wrist, left_shoulder, 'L'),
                (right_wrist, right_shoulder, 'R'),
            ):
                if not (_kp_ok(wrist, self.HAND_RAISE_WRIST_CONF) and
                        _kp_ok(shoulder, self.HAND_RAISE_SHOULDER_CONF)):
                    continue
                clearance = float(shoulder[1]) - float(wrist[1])  # +ve = wrist is higher
                if clearance >= raise_thresh_min:
                    # Boost confidence proportionally: full boost at clearance = 30% bbox_h
                    boost = min(clearance / (bbox_h * 0.30), 1.0) * 0.20
                    conf  = min(float(wrist[2]) + boost, 0.95)
                    if conf > best_conf:
                        best_conf = conf
                        raised    = True
                    print(f'[HAND RAISE] Person {person.track_id}: T1 {side} '
                          f'clearance={clearance:.1f}px conf={conf:.2f}')

            # --- Tier 2: elbow above shoulder (fallback when wrist occluded) ---
            if not raised:
                for elbow, shoulder, side in (
                    (left_elbow, left_shoulder, 'L'),
                    (right_elbow, right_shoulder, 'R'),
                ):
                    if not (_kp_ok(elbow, self.HAND_RAISE_ELBOW_CONF) and
                            _kp_ok(shoulder, self.HAND_RAISE_SHOULDER_CONF)):
                        continue
                    clearance = float(shoulder[1]) - float(elbow[1])
                    if clearance >= raise_thresh_min:
                        conf = min(float(elbow[2]) * 0.80, 0.75)  # lower ceiling for elbow
                        if conf > best_conf:
                            best_conf = conf
                            raised    = True
                        print(f'[HAND RAISE] Person {person.track_id}: T2 elbow {side} '
                              f'clearance={clearance:.1f}px conf={conf:.2f}')

            # --- Tier 3: wrist above nose (very high / full arm raise) ---
            if not raised and nose is not None and _kp_ok(nose, 0.40):
                for wrist, side in ((left_wrist, 'L'), (right_wrist, 'R')):
                    if not _kp_ok(wrist, 0.30):  # very permissive — arm occluded by body
                        continue
                    if float(wrist[1]) < float(nose[1]):  # wrist above nose
                        conf = min(float(wrist[2]) + 0.15, 0.85)
                        if conf > best_conf:
                            best_conf = conf
                            raised    = True
                        print(f'[HAND RAISE] Person {person.track_id}: T3 wrist-above-nose '
                              f'{side} conf={conf:.2f}')

            return raised, best_conf
        except Exception as exc:
            print(f'[HAND RAISE] error track {person.track_id}: {exc}')
        return False, 0.0

    # ── Eating Detection ──────────────────────────────────────────────────────
    def _detect_eating(self, person: TrackedPerson, food_detections: List[Tuple]) -> Tuple[bool, float]:
        if not food_detections or person.keypoints is None or person.keypoints.size == 0:
            return False, 0.0
        try:
            kp = person.keypoints
            x1, y1, x2, y2 = person.bbox
            bbox_h = y2 - y1
            if bbox_h <= 0:
                return False, 0.0

            # Use mouth estimate: COCO kp[0]=nose, kp[3]=left_ear, kp[4]=right_ear
            # Best mouth proxy is nose (kp[0]) shifted slightly downward
            nose = kp[0]
            if len(nose) < 3 or nose[2] < 0.5:
                return False, 0.0
            mouth_pt = np.array([nose[0], nose[1] + bbox_h * 0.04])  # slight downward offset

            best_food_conf = 0.0
            food_near_mouth = False
            for (fx1, fy1, fx2, fy2, conf) in food_detections:
                if conf < 0.5:  # was 0.75 — too strict, reduced recall significantly
                    continue
                fc = np.array([(fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0])
                if np.linalg.norm(fc - mouth_pt) < bbox_h * 0.25:  # slightly larger proximity
                    food_near_mouth = True
                    best_food_conf = max(best_food_conf, conf)

            if not food_near_mouth:
                return False, 0.0

            # Check at least one wrist is near the mouth (hand-to-mouth gesture)
            for idx in (9, 10):
                if len(kp) > idx:
                    w = kp[idx]
                    if (len(w) >= 3 and w[2] >= 0.5 and w[0] != 0.0 and  # was 0.8 — too strict
                            np.linalg.norm(w[:2] - mouth_pt) / bbox_h < self.EATING_HAND_TO_MOUTH_THRESHOLD):
                        return True, best_food_conf
        except Exception as exc:
            print(f'[EATING] error track {person.track_id}: {exc}')
        return False, 0.0

    # ── Evaluate Person ───────────────────────────────────────────────────────
    def evaluate_person(self, track_id: int,
                        phone_detections: List[Tuple],
                        food_detections: List[Tuple],
                        frame: Optional[np.ndarray] = None,
                        book_detections: Optional[List[Tuple]] = None) -> DetectionResult:
        with self.lock:  # Thread-safe evaluation
            if track_id not in self.tracked_people:
                return DetectionResult(
                    type='not_visible', bbox=(0, 0, 0, 0), confidence=0.0,
                    color=COLOR_MAP['not_visible'], label=LABEL_MAP['not_visible'],
                    is_alert=False, is_distracted=False)

            person = self.tracked_people[track_id]
            raw_confidence = person.last_raw_confidence

            raw_head = self._calculate_head_pose(person)
            is_hand_raised, hand_conf = self._detect_hand_raise(person)
            if is_hand_raised:
                person.behavior_history.append('hand_raised')
                raw_confidence = hand_conf
            else:
                is_phone, phone_conf = self._detect_phone_usage(person, phone_detections, raw_head, book_detections)
                if is_phone:
                    person.behavior_history.append('using_phone')
                    raw_confidence = phone_conf
                else:
                    is_eating, eat_conf = self._detect_eating(person, food_detections)
                    if is_eating:
                        person.behavior_history.append('eating_food')
                        raw_confidence = eat_conf
                    else:
                        person.behavior_history.append('focused' if raw_head == 'focused' else 'distracted')
                        raw_confidence = 0.75 if raw_head == 'focused' else 0.65

            person.last_raw_confidence = raw_confidence

            final_behavior = person.last_final_behavior
            history = list(person.behavior_history)

            if len(history) >= self.NORMAL_CONFIRM_FRAMES:
                last = history[-1]

                if last == 'hand_raised':
                    # Hand raise is a transient gesture — confirm in 3 frames (0.3 s @ 10 FPS)
                    # to avoid missing it while still filtering single-frame noise.
                    n = min(self.HAND_RAISE_CONFIRM_FRAMES, len(history))
                    if all(h == last for h in history[-n:]):
                        final_behavior = last
                elif last in ALERT_POSES:
                    # Other alert behaviors (phone, eating) use the full 6-frame confirmation
                    n = min(self.ALERT_CONFIRM_FRAMES, len(history))
                    if all(h == last for h in history[-n:]):
                        final_behavior = last
                    # else: keep previous final_behavior (don't flip on a single frame)
                else:
                    recent = history[-self.NORMAL_CONFIRM_FRAMES:]
                    if len(set(recent)) == 1:
                        final_behavior = recent[0]
                    else:
                        window = history[-10:]
                        top_label, top_count = Counter(window).most_common(1)[0]
                        if top_count / len(window) > 0.6:
                            final_behavior = top_label
                        # else: keep previous stable label — avoid flickering
            # If history is too short, keep last_final_behavior (stay stable)

            person.last_final_behavior = final_behavior

            # Always use fresh raw_confidence for the live detection result.
            # last_raw_confidence is only a fallback for frames with no detectable action.
            confidence = raw_confidence if raw_confidence > 0.0 else (person.last_raw_confidence or 0.75)
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
    # 67 = cell phone, 76 = remote (YOLOv8/11 frequently confuses phones with remotes)
    _PHONE_CLS = {67, 76}
    _FOOD_CLS  = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55}  # banana..cake
    _BOOK_CLS  = {73}

    def __init__(self, process_fps: int = 10, buffer_size: int = 5):
        self.process_fps    = process_fps
        self.frame_interval = 1.0 / process_fps
        self.frame_buffer: deque = deque(maxlen=buffer_size)
        self.result_buffer: deque = deque(maxlen=2)
        self.yolo_model   = None
        self.object_model = None
        self.roboflow_model = None
        self.running      = False
        self.lock         = threading.Lock()
        self.stop_event   = threading.Event()
        self.phone_detections: List[Tuple] = []
        self.food_detections:  List[Tuple] = []
        self.person_tracks: List[Tuple] = []
        self.behavior_engine = TemporalBehaviorEngine()
        self.fight_detector = None
        self._ensure_models()
        self._init_fight_detector()

    def _ensure_models(self):
        self.yolo_model, self.object_model = _SharedYOLOModels.get()
        self._init_roboflow_model()

    def _init_roboflow_model(self):
        """Initialize Roboflow model once at startup (not per-frame)."""
        api_key = os.environ.get('ROBOFLOW_API_KEY', '').strip()
        if not api_key or self.roboflow_model is not None:
            return
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=api_key)
            project = rf.workspace().project("classroom-cell-phone-detection")
            self.roboflow_model = project.version(18).model
            print('[OK] Roboflow model loaded')
        except ImportError:
            print('[WARN] roboflow not installed — phone fallback disabled')
        except Exception as e:
            print(f'[WARN] Roboflow init failed: {e}')

    def _init_fight_detector(self):
        try:
            from classroom_monitor.fight_detection_3dcnn import FightDetector3DCNN
            self.fight_detector = FightDetector3DCNN()
            print('[OK] Fight detector initialized')
        except Exception as e:
            print(f'[WARN] Fight detector initialization failed: {e}')
            self.fight_detector = None

    def _load_models(self):
        """Alias kept for backward compatibility."""
        self._ensure_models()

    def _capture_frames(self, camera_url: str):
        cap, reconnect_delay = None, 1.0
        while not self.stop_event.is_set():
            try:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(camera_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    if cap.isOpened():
                        print('[OK] Camera connected')
                        reconnect_delay = 1.0
                    else:
                        cap.release()
                        cap = None
                        time.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 2, 10.0)
                        continue
                ret, frame = cap.read()
                if not ret:
                    print('[WARN] Camera read failed — reconnecting')
                    cap.release()
                    cap = None
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 10.0)
                    continue
                reconnect_delay = 1.0  # reset on successful read
                with self.lock:
                    self.frame_buffer.append((time.time(), frame.copy()))
                # Throttle capture loop to ~60 fps max to avoid CPU spin
                time.sleep(0.016)
            except Exception as e:
                print(f'[ERROR] Capture: {e}')
                if cap:
                    cap.release()
                    cap = None
                time.sleep(1.0)
        if cap:
            cap.release()

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
        """Run object models: YOLO first (low conf for phones), Roboflow as backup."""
        phone_dets, food_dets, book_dets = [], [], []

        if self.object_model is not None:
            try:
                # Run at low global conf so we catch phones; we filter per-class below.
                for result in self.object_model(frame, verbose=False, conf=0.20, iou=0.45):
                    if result.boxes is None:
                        continue
                    for i in range(len(result.boxes)):
                        cls  = int(result.boxes.cls[i])
                        conf = float(result.boxes.conf[i])
                        x1, y1, x2, y2 = map(int, result.boxes.xyxy[i])
                        det = (x1, y1, x2, y2, conf)
                        if cls in self._PHONE_CLS:
                            # Lower bar for phone/remote: accept from conf >= 0.20
                            phone_dets.append(det)
                        elif cls in self._FOOD_CLS and conf >= 0.30:
                            food_dets.append(det)
                        elif cls in self._BOOK_CLS and conf >= 0.30:
                            book_dets.append(det)
            except Exception as e:
                print(f'[WARN] YOLO object detection failed: {e}')

        # Roboflow fallback — only attempt if model is already loaded OR key is present,
        # but do NOT re-initialize the model object every frame when phone_dets is empty.
        if not phone_dets and self.roboflow_model is not None:
            try:
                result = self.roboflow_model.predict(frame, confidence=25, overlap=30).json()
                for prediction in result.get('predictions', []):
                    x1 = int(prediction['x'] - prediction['width'] / 2)
                    y1 = int(prediction['y'] - prediction['height'] / 2)
                    x2 = int(prediction['x'] + prediction['width'] / 2)
                    y2 = int(prediction['y'] + prediction['height'] / 2)
                    conf = prediction['confidence']
                    phone_dets.append((x1, y1, x2, y2, conf))
                if phone_dets:
                    print(f'[ROBOFLOW] Found {len(phone_dets)} phone(s)')
            except Exception as e:
                print(f'[WARN] Roboflow inference failed: {e}')

        return phone_dets, food_dets, book_dets

    def _parse_pose_detections(self, frame: np.ndarray):
        """Run pose model and return list of (track_id, x1, y1, x2, y2, conf, keypoints)."""
        tracks = []
        if self.yolo_model is None:
            return tracks
        try:
            for result in self.yolo_model.track(frame, persist=True, verbose=False,
                                                  conf=0.3, iou=0.5, tracker='bytetrack.yaml'):
                if result.boxes is None:
                    continue
                kp_list = result.keypoints if hasattr(result, 'keypoints') else None
                boxes_id = result.boxes.id  # may be None on first frame
                for i in range(len(result.boxes)):
                    if int(result.boxes.cls[i]) != 0:
                        continue
                    x1, y1, x2, y2 = map(int, result.boxes.xyxy[i])
                    conf = float(result.boxes.conf[i])
                    # Fallback to negative index when tracker hasn't assigned an ID yet.
                    # Negative IDs will be cleaned up in cleanup_stale and never collide
                    # with real tracker IDs (which are always positive integers).
                    if boxes_id is not None and i < len(boxes_id) and boxes_id[i] is not None:
                        track_id = int(boxes_id[i])
                    else:
                        track_id = -(i + 1)  # provisional negative ID
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

    def _run_behavior_evaluation(self, frame, person_tracks, phone_dets, food_dets, book_dets, timestamp, fight_detected=False):
        active_tids = set()
        for tid, x1, y1, x2, y2, conf, kp in person_tracks:
            active_tids.add(tid)
            self.behavior_engine.update_person(tid, (x1, y1, x2, y2), kp, timestamp)

        self.behavior_engine.cleanup_stale(timestamp)

        results = []
        for tid in active_tids:
            if tid not in self.behavior_engine.tracked_people:
                continue
            if fight_detected:
                person = self.behavior_engine.tracked_people[tid]
                result = DetectionResult(
                    type='fighting',
                    bbox=person.bbox,
                    confidence=0.9,
                    color=(0, 0, 255),
                    label='Fighting!',
                    is_alert=True,
                    is_distracted=False,
                    track_id=tid
                )
                results.append(result)
            else:
                results.append(self.behavior_engine.evaluate_person(
                    tid, phone_dets, food_dets, book_detections=book_dets))
        return results

    def _process_single_frame(self, frame: np.ndarray, timestamp: float):
        try:
            person_tracks              = self._parse_pose_detections(frame)
            phone_dets, food_dets, book_dets = self._parse_object_detections(frame)
            
            fight_detected = False
            if self.fight_detector is not None:
                fight_detected = self.fight_detector.predict(frame)
                if fight_detected:
                    print('[ALERT] Fight detected!')
            
            final_results = self._run_behavior_evaluation(
                frame, person_tracks, phone_dets, food_dets, book_dets, timestamp, fight_detected)
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
                    enc_raw = s['encoding']
                    # Server may return encoding as a JSON string or already-decoded list
                    if isinstance(enc_raw, str):
                        enc_raw = json.loads(enc_raw)
                    d = fr.face_distance([np.array(enc_raw, dtype=np.float64)], det)[0]
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
        frame_count, last_ts = 0, 0.0

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
                detections.append(entry)

            frame_count += 1
            now = time.time()

            for det in detections:
                if not (det['is_alert'] or det['is_distracted']):
                    continue
                tid = det.get('track_id')
                key = (det['type'], tid if tid is not None else tuple(det['bbox']))
                if now - self.last_alert_time.get(key, 0) < self.alert_cooldown:
                    continue
                sid, name, roll = self._recognize_face(frame, det['bbox'])
                self._report_incident(det, frame, sid, name, roll, all_detections=detections)
                self.last_alert_time[key] = now

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