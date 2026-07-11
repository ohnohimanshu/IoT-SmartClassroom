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
# Threshold is high: Path B alone (max 0.20) can NEVER fire an alert by itself.
# Only sustained YOLO hits (Path A) can push confidence above threshold.
PHONE_CONFIRM_THRESHOLD = 0.50   # requires ~67% of frames to have a YOLO hit
PATH_A_WEIGHT           = 1.00   # Path A is the only reliable signal
PATH_B_WEIGHT           = 0.00   # Path B disabled — too many false positives in classrooms
WINDOW_SECONDS          = 3.0    # longer window = more evidence required
STALE_TIMEOUT           = 3.0    # seconds before DetectionState is purged
YOLO_PHONE_CONF         = 0.55   # high confidence threshold — reject book/paper misclassifications
MIN_WRIST_CONF          = 0.45   # min keypoint confidence to use a wrist
# Min YOLO hits required in window before Path B can even contribute (future use)
MIN_YOLO_HITS_FOR_PATH_B = 3


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

    def cleanup_stale(self, active_track_ids: set) -> None:
        """Remove DetectionState entries for tracks no longer in the frame.
        Mirrors the same pattern as HeadPoseDetector.cleanup_stale — call
        alongside behavior_engine.cleanup_stale each frame."""
        with self._lock:
            stale = [tid for tid in self._states if tid not in active_track_ids]
            for tid in stale:
                del self._states[tid]
            if stale:
                print(f'[PHONE] Cleaned up stale states: {stale}')

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
        """
        Reject phone bboxes that are:
        - absurdly large or tiny relative to the person
        - nearly square (books/notebooks are square; phones are rectangular)
        """
        if bbox_h <= 0 or pw <= 0 or ph <= 0:
            return False
        # Size check relative to person height
        if not (0.04 <= min(pw, ph) / bbox_h <= 0.55):
            return False
        # Aspect ratio: phone must be at least 1.4:1 (portrait or landscape)
        aspect = max(pw, ph) / min(pw, ph)
        if aspect < 1.4:
            return False
        return True

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

            # Phone centre must be in the lower 65% of the person bbox
            # (rules out books held up to read, papers on desks at head level)
            phone_rel_y = (pc[1] - y1) / bbox_h
            if phone_rel_y < 0.35:
                continue

            # Require wrist/elbow proximity confirmation — tightened to 0.28
            if person.keypoints is not None and len(person.keypoints) > 10:
                for idx in (9, 10, 7, 8):
                    if idx >= len(person.keypoints):
                        continue
                    w = person.keypoints[idx]
                    if self._wrist_ok(w, 0.45) and np.linalg.norm(w[:2] - pc) / bbox_h < 0.28:
                        yolo_hit  = True
                        yolo_conf = max(yolo_conf, float(conf))
                        break
            else:
                # No keypoints at all — only trust very high confidence + strong overlap
                if conf >= 0.70:
                    ov_x = max(0, min(px2, x2) - max(px1, x1))
                    ov_y = max(0, min(py2, y2) - max(py1, y1))
                    if ov_x * ov_y / max(pw * ph, 1) > 0.6:
                        yolo_hit  = True
                        yolo_conf = max(yolo_conf, float(conf))

            if yolo_hit:
                break

        state.yolo_phone_history.append(yolo_hit)

        # ── Path B: disabled (PATH_B_WEIGHT = 0.0) ───────────────────────────
        # Behavioural heuristics (head-down + centred wrist) produce too many
        # false positives in classroom settings where students write with
        # their hands centred on notebooks. Path A (YOLO object hits) is the
        # sole reliable signal. Path B code is retained but does not modify
        # yolo_phone_history and does not amplify wrist_proximity_history.

        return self._accumulate(state)

    def _accumulate(self, state: DetectionState) -> Tuple[bool, float]:
        """
        Path A only accumulator.
        Two-gate approach — both must pass before flagging phone use:
          1. Density: >= PHONE_CONFIRM_THRESHOLD of the full window has YOLO hits.
          2. Recency: at least 2 of the last 4 frames have hits (prevents a
             burst of old detections in a cold window from triggering a flag
             after the object has gone away).
        Neither gate fires until the window is at least half populated.
        """
        window = state.window_length or 1
        hits   = sum(state.yolo_phone_history)
        filled = len(state.yolo_phone_history)

        # Require at least half the window to be populated before deciding
        if filled < max(4, window // 2):
            return False, 0.0

        # Gate 1: overall density
        path_a     = hits / window
        confidence = path_a
        if confidence < PHONE_CONFIRM_THRESHOLD:
            return False, 0.0

        # Gate 2: recent activity — phone must still be present, not just
        # remembered from a burst several seconds ago
        recent = list(state.yolo_phone_history)[-4:]
        if sum(recent) < 2:
            return False, 0.0

        print(
            f'[PHONE] Track {state.track_id}: '
            f'conf={confidence:.3f} hits={hits}/{filled} window={window}'
        )
        return True, round(confidence, 3)
