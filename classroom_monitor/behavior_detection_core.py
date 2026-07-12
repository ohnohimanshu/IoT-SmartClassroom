import threading
import numpy as np
from collections import deque, Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from classroom_monitor.constants import (
    COLOR_MAP, LABEL_MAP, ALERT_POSES, DISTRACTED_POSES,
)

# ── Data Structures ──────────────────────────────────────────────────────────
@dataclass
class TrackedPerson:
    track_id: int
    bbox: Tuple[int, int, int, int]
    keypoints: Optional[np.ndarray] = None
    behavior_history: deque = field(default_factory=lambda: deque(maxlen=20))
    last_seen: float = 0.0
    last_final_behavior: str = 'focused'
    keypoint_history: deque = field(default_factory=lambda: deque(maxlen=30))
    last_raw_confidence: float = 0.0
    # Confidence last observed for each raw behavior label, so that
    # switching final_behavior back to a previously-seen label restores
    # that label's own confidence instead of an unrelated frame's value.
    confidence_by_label: Dict[str, float] = field(default_factory=dict)


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


# ── Shared Helpers ───────────────────────────────────────────────────────────
class SharedHelpers:
    @staticmethod
    def point_near_book(pt, bbox_h: float, book_detections: List[Tuple]) -> bool:
        for (bx1, by1, bx2, by2, _) in book_detections:
            bc = np.array([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0])
            if np.linalg.norm(np.array(pt) - bc) / bbox_h < 0.3:
                return True
        return False

    @staticmethod
    def hands_spread_writing_posture(wrist_pts, bbox_h: float) -> bool:
        if len(wrist_pts) != 2 or bbox_h <= 0:
            return False
        spread = np.linalg.norm(wrist_pts[0] - wrist_pts[1]) / bbox_h
        return spread > 0.3

    @staticmethod
    def wrist_lateral_offset(wrist, x1: float, x2: float) -> float:
        bbox_w = max(x2 - x1, 1.0)
        center_x = (x1 + x2) / 2.0
        return abs(wrist[0] - center_x) / bbox_w

    @staticmethod
    def calculate_wrist_motion_variance(person: TrackedPerson) -> Tuple[bool, float]:
        # Require fewer frames before making a writing judgement (was 10)
        if len(person.keypoint_history) < 6:
            return False, 0.0

        wrist_positions = []
        for _, kp in person.keypoint_history:
            if kp is not None and len(kp) > 10:
                for idx in (9, 10):
                    if idx < len(kp):
                        w = kp[idx]
                        if len(w) >= 3 and w[2] >= 0.4 and w[0] != 0.0:
                            wrist_positions.append(w[:2])

        if len(wrist_positions) < 5:
            return False, 0.0

        wrist_array = np.array(wrist_positions)
        variance_x = np.var(wrist_array[:, 0])
        variance_y = np.var(wrist_array[:, 1])
        total_variance = variance_x + variance_y

        # Higher threshold so we don't suppress phone detection due to slight movement
        is_writing = total_variance > 300   # was 150 — too aggressive
        confidence = min(total_variance / 800, 0.95)
        return is_writing, confidence


# ── Temporal Behavior Engine ─────────────────────────────────────────────────
class TemporalBehaviorEngine:
    LOW_CONFIDENCE_THRESHOLD     = 0.4
    HEAD_DOWN_CONSECUTIVE_FRAMES = 2
    NORMAL_CONFIRM_FRAMES = 3    # was 5
    VOTE_WINDOW    = 8           # trailing window used for the majority vote
    ALERT_MAJORITY = 0.80        # share required for an alert-type label to win the vote
    NORMAL_MAJORITY = 0.50       # share required (strictly greater) for a non-alert label

    def __init__(self):
        self.tracked_people: Dict[int, TrackedPerson] = {}
        self.cleanup_threshold = 2.0
        self.low_confidence_counters: Dict[int, int] = {}
        self.lock = threading.Lock()

    def update_person(self, track_id: int, bbox: Tuple, keypoints: Optional[np.ndarray], timestamp: float):
        with self.lock:
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
        with self.lock:
            stale = [tid for tid, p in self.tracked_people.items()
                     if current_time - p.last_seen > self.cleanup_threshold]
            for tid in stale:
                del self.tracked_people[tid]
                self.low_confidence_counters.pop(tid, None)

    def _is_alert_label(self, label: str) -> bool:
        return label in ALERT_POSES or label == 'hand_raised'

    def _majority_vote(self, history: List[str]) -> Optional[str]:
        """
        Single, unified hysteresis vote over the trailing window of raw
        behavior readings. Applies regardless of what the newest raw
        reading happens to be, so the confirmation bar for an alert label
        is always ALERT_MAJORITY (0.80) — never looser depending on which
        branch happened to run.

        Returns the label that should become the new final_behavior, or
        None if nothing in the window clears its required bar (in which
        case the caller keeps the previous final_behavior).
        """
        if not history:
            return None

        last = history[-1]

        # Fast-response shortcut: N consecutive identical non-alert
        # readings confirm immediately, without waiting to fill the full
        # vote window. Never applies to alert-type labels — those always
        # go through the full ALERT_MAJORITY bar below.
        if not self._is_alert_label(last) and len(history) >= self.NORMAL_CONFIRM_FRAMES:
            recent = history[-self.NORMAL_CONFIRM_FRAMES:]
            if len(set(recent)) == 1:
                return recent[0]

        window = history[-self.VOTE_WINDOW:]
        top_label, top_count = Counter(window).most_common(1)[0]
        share = top_count / len(window)

        if self._is_alert_label(top_label):
            if share >= self.ALERT_MAJORITY:
                return top_label
        else:
            if share > self.NORMAL_MAJORITY:
                return top_label

        return None

    def evaluate_final_behavior(self, person: TrackedPerson, raw_behavior: str, raw_confidence: float) -> Tuple[str, float]:
        """
        Update a tracked person's smoothed behavior state from this frame's
        raw reading and return (final_behavior, confidence).

        Holds self.lock for the full read-modify-write so this is safe to
        call from any thread holding a reference to `person`, even after
        the caller has released the lock it used to originally fetch that
        person from tracked_people (multiple threads can share a camera's
        detector instance).
        """
        with self.lock:
            person.behavior_history.append(raw_behavior)

            # Record this label's confidence BEFORE any fallback lookup,
            # keyed by label rather than overwriting a single scalar, so
            # switching final_behavior back to a previously-seen label
            # restores that label's own last confidence rather than an
            # unrelated frame's value.
            person.confidence_by_label[raw_behavior] = raw_confidence
            person.last_raw_confidence = raw_confidence  # kept for introspection/back-compat

            history = list(person.behavior_history)
            voted = self._majority_vote(history)
            if voted is not None:
                person.last_final_behavior = voted

            final_behavior = person.last_final_behavior

            if final_behavior == raw_behavior:
                confidence = raw_confidence
            else:
                confidence = person.confidence_by_label.get(final_behavior, person.last_raw_confidence)
            confidence = confidence or 0.75

            return final_behavior, confidence