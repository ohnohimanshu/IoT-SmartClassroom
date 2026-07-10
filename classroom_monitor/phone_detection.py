"""
Phone Detection — Time-Series State Engine
==========================================
Replaces brittle full-frame spatial heuristics with a per-track sliding-window
probabilistic accumulator.

Two confidence paths:
  Path A (weight 0.75) — temporal density of YOLO / Roboflow phone object hits
  Path B (weight 0.25) — sustained behavioural anomaly (head-down + centred wrist)

All geometry is expressed relative to the person bounding box (spatial-invariant).
"""

import threading
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from classroom_monitor.behavior_detection_core import TrackedPerson, SharedHelpers
from classroom_monitor.head_pose_detection import HeadPoseDetector


# ── Tunable configuration ─────────────────────────────────────────────────────
PHONE_CONFIRM_THRESHOLD = 0.35   # min combined confidence to fire alert
PATH_A_WEIGHT           = 0.75   # weight for YOLO/RF temporal density
PATH_B_WEIGHT           = 0.25   # weight for head+wrist behavioural signal
WINDOW_SECONDS          = 2.5    # sliding window duration in seconds
STALE_TIMEOUT           = 3.0    # seconds before DetectionState is purged
YOLO_PHONE_CONF         = 0.30   # min YOLO phone confidence to count as a hit
MIN_WRIST_CONF          = 0.40   # min keypoint confidence to use a wrist


# ── Per-track sliding-window state ────────────────────────────────────────────
@dataclass
class DetectionState:
    """Holds per-track sliding-window signal histories for one student."""
    track_id:                int
    window_length:           int
    # True  = phone object detected by YOLO/RF in this frame for this person
    yolo_phone_history:      deque = field(default_factory=deque)
    # True  = head pose was head_down or looking_away this frame
    head_down_history:       deque = field(default_factory=deque)
    # float = wrist-centre proximity score [0,1] this frame (1 = centred on body)
    wrist_proximity_history: deque = field(default_factory=deque)
    last_seen:               float = 0.0

    def __post_init__(self):
        # Re-create deques with correct maxlen after deserialization / init
        self.yolo_phone_history      = deque(self.yolo_phone_history,      maxlen=self.window_length)
        self.head_down_history       = deque(self.head_down_history,       maxlen=self.window_length)
        self.wrist_proximity_history = deque(self.wrist_proximity_history, maxlen=self.window_length)


# ── Phone Detector ────────────────────────────────────────────────────────────
class PhoneDetector:
    """
    Stateful per-track phone detector.

    detect_phone_usage() must be called every processed frame with the current
    track_id and timestamp so the sliding-window histories stay accurate.
    """

    def __init__(self, process_fps: int = 10):
        self._process_fps   = process_fps
        self._window_length = max(1, round(process_fps * WINDOW_SECONDS))
        self._states: Dict[int, DetectionState] = {}
        self._lock          = threading.Lock()
        self.head_pose_detector = HeadPoseDetector()

    # ── State registry ────────────────────────────────────────────────────────

    def _get_or_create_state(self, track_id: int) -> DetectionState:
        with self._lock:
            if track_id not in self._states:
                self._states[track_id] = DetectionState(
                    track_id=track_id,
                    window_length=self._window_length,
                )
            return self._states[track_id]

    def cleanup_stale(self, current_time: float) -> None:
        """Remove DetectionState entries for tracks not seen recently."""
        with self._lock:
            expired = [
                tid for tid, s in self._states.items()
                if current_time - s.last_seen > STALE_TIMEOUT
            ]
            for tid in expired:
                del self._states[tid]
                if expired:
                    print(f'[PHONE] Cleaned up stale states: {expired}')

    # ── Spatial-invariance helpers ────────────────────────────────────────────

    @staticmethod
    def _wrist_centre_proximity(wrist, x1: float, y1: float,
                                x2: float, y2: float) -> float:
        """
        Returns [0, 1] representing how centred the wrist is horizontally
        within the person bbox.
          1.0 = perfectly centred (holding something in front of body)
          0.0 = at the horizontal edge of the bbox (writing at a desk)
        """
        bbox_w = max(x2 - x1, 1.0)
        rel_x  = (wrist[0] - x1) / bbox_w          # 0 = left edge, 1 = right edge
        # Distance from centre (0.5), normalised to [0, 1]
        proximity = 1.0 - abs(rel_x - 0.5) * 2.0
        return float(np.clip(proximity, 0.0, 1.0))

    @staticmethod
    def _wrist_ok(w, min_conf: float = MIN_WRIST_CONF) -> bool:
        return w is not None and len(w) >= 3 and w[2] >= min_conf and w[0] != 0.0

    @staticmethod
    def _phone_size_sane(pw: float, ph: float, bbox_h: float) -> bool:
        """Reject phone bboxes that are absurdly large or tiny vs the person."""
        if bbox_h <= 0:
            return False
        return 0.03 <= min(pw, ph) / bbox_h and max(pw, ph) / bbox_h <= 0.7

    # ── Core detection ────────────────────────────────────────────────────────

    def detect_phone_usage(
        self,
        person:           TrackedPerson,
        phone_detections: List[Tuple],
        head_pose:        str,
        book_detections:  Optional[List[Tuple]] = None,
        track_id:         int = -1,
        timestamp:        float = 0.0,
    ) -> Tuple[bool, float]:
        """
        Returns (is_phone_detected, confidence).

        Updates the track's sliding-window histories and computes the
        two-path probabilistic accumulator score.
        """
        book_detections = book_detections or []
        x1, y1, x2, y2 = person.bbox
        bbox_h = y2 - y1
        bbox_w = x2 - x1
        if bbox_h <= 0 or bbox_w <= 0:
            return False, 0.0

        # Use person.track_id as fallback if caller did not pass track_id
        tid = track_id if track_id >= 0 else person.track_id
        state = self._get_or_create_state(tid)
        state.last_seen = timestamp or time.monotonic()

        # Writing suppression — high-variance wrist motion means student is writing
        is_writing, _ = SharedHelpers.calculate_wrist_motion_variance(person)
        if is_writing:
            state.yolo_phone_history.append(False)
            state.head_down_history.append(False)
            state.wrist_proximity_history.append(0.0)
            return self._accumulate(state)

        # Head-down signal
        head_is_down = (
            head_pose in ('head_down', 'looking_away')
            or self.head_pose_detector.is_head_down_like(person, head_pose)
        )
        state.head_down_history.append(head_is_down)

        # Best wrist proximity signal
        best_proximity = 0.0
        if person.keypoints is not None and len(person.keypoints) > 10:
            for idx in (9, 10):   # left wrist, right wrist
                w = person.keypoints[idx]
                if self._wrist_ok(w):
                    prox = self._wrist_centre_proximity(w, x1, y1, x2, y2)
                    best_proximity = max(best_proximity, prox)
        state.wrist_proximity_history.append(best_proximity)

        # ── Path A: YOLO / Roboflow phone object hit ──────────────────────────
        yolo_hit = False
        yolo_conf = 0.0

        for (px1, py1, px2, py2, conf) in phone_detections:
            if conf < YOLO_PHONE_CONF:
                continue
            pw, ph = px2 - px1, py2 - py1
            if not self._phone_size_sane(pw, ph, bbox_h):
                continue

            pc = np.array([(px1 + px2) / 2.0, (py1 + py2) / 2.0])

            # Phone must spatially overlap or be near the person bbox
            in_x = px1 < x2 + 20 and px2 > x1 - 20
            in_y = py1 < y2 + 20 and py2 > y1 - 20
            if not (in_x and in_y):
                continue

            # Prefer wrist/elbow proximity confirmation
            if person.keypoints is not None and len(person.keypoints) > 10:
                for idx in (9, 10, 7, 8):
                    if idx >= len(person.keypoints):
                        continue
                    w = person.keypoints[idx]
                    if self._wrist_ok(w, 0.4) and np.linalg.norm(w[:2] - pc) / bbox_h < 0.35:
                        yolo_hit  = True
                        yolo_conf = max(yolo_conf, float(conf))
                        break

            # High-confidence YOLO with strong overlap — trust without keypoints
            if not yolo_hit and conf >= 0.55:
                ov_x = max(0, min(px2, x2) - max(px1, x1))
                ov_y = max(0, min(py2, y2) - max(py1, y1))
                if ov_x * ov_y / max(pw * ph, 1) > 0.5:
                    yolo_hit  = True
                    yolo_conf = max(yolo_conf, float(conf))

            if yolo_hit:
                break

        state.yolo_phone_history.append(yolo_hit)

        # ── Path B: Heuristic behavioural signal ──────────────────────────────
        # Only evaluated when keypoints are available and head is down.
        # Does NOT directly fire an alert — feeds into accumulator as Path B weight.
        if (head_is_down
                and person.keypoints is not None
                and len(person.keypoints) > 10):
            try:
                left_wrist  = person.keypoints[9]
                right_wrist = person.keypoints[10]
                wrist_pts = [
                    w[:2] for w in (left_wrist, right_wrist)
                    if self._wrist_ok(w, 0.5)
                ]

                if wrist_pts:
                    def _book_near(pt) -> bool:
                        return SharedHelpers.point_near_book(pt, bbox_h, book_detections)

                    for w in wrist_pts:
                        rel_y  = (w[1] - y1) / bbox_h
                        prox   = self._wrist_centre_proximity(w, x1, y1, x2, y2)

                        # Centred wrist in torso/lap zone, not near a book
                        if (0.45 <= rel_y <= 0.90
                                and prox > 0.55
                                and not _book_near(w)):
                            # Boost path B signal by adding a synthetic phone hit
                            # at lower weight — will be reflected in path_b only
                            state.wrist_proximity_history[-1] = max(
                                state.wrist_proximity_history[-1], prox
                            )
                            break

                    # Cupped-hands pattern (two wrists close together, both low)
                    if len(wrist_pts) == 2:
                        spread = np.linalg.norm(wrist_pts[0] - wrist_pts[1]) / bbox_h
                        both_low = all((w[1] - y1) / bbox_h > 0.50 for w in wrist_pts)
                        if both_low and spread < 0.25:
                            if not any(_book_near(w) for w in wrist_pts):
                                # Amplify proximity score for cupped grip
                                state.wrist_proximity_history[-1] = min(
                                    state.wrist_proximity_history[-1] + 0.3, 1.0
                                )
            except Exception as exc:
                print(f'[PHONE] Path B heuristic error: {exc}')

        return self._accumulate(state)

    def _accumulate(self, state: DetectionState) -> Tuple[bool, float]:
        """Two-path probabilistic accumulator over the sliding window."""
        window = state.window_length or 1

        # Path A — fraction of frames in window with a YOLO/RF phone hit
        path_a = sum(state.yolo_phone_history) / window

        # Path B — head-down density × mean wrist-centre proximity
        head_density = (
            sum(state.head_down_history) / window
            if state.head_down_history else 0.0
        )
        wrist_density = (
            sum(state.wrist_proximity_history) / len(state.wrist_proximity_history)
            if state.wrist_proximity_history else 0.0
        )
        path_b = head_density * wrist_density

        confidence = PATH_A_WEIGHT * path_a + PATH_B_WEIGHT * path_b
        is_phone   = confidence >= PHONE_CONFIRM_THRESHOLD

        if is_phone:
            print(
                f'[PHONE] Track {state.track_id}: '
                f'conf={confidence:.3f} (A={path_a:.2f} B={path_b:.2f})'
            )

        return is_phone, round(confidence, 3)
