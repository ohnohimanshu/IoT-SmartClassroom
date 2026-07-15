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

    # Raw variance ceiling below which a wrist counts as "phone-still" — used
    # to POSITIVELY require stillness for the posture-only phone heuristic in
    # phone_detection.py, not just to rule out obvious writing motion. Writing
    # involves continuous small wrist strokes even when it doesn't cross the
    # is_writing threshold above; a phone-holding hand is comparatively
    # static. This value has not been calibrated against your real camera
    # footage yet — turn on PHONE_DEBUG=1, watch the printed variance values
    # for a few known-writing and known-phone-holding students, and adjust
    # PHONE_STILLNESS_MAX_VARIANCE in phone_detection.py accordingly if this
    # default is rejecting/accepting too much.
    PHONE_STILLNESS_DEFAULT_MAX_VARIANCE = 60

    @staticmethod
    def wrist_motion_available_and_variance(person: TrackedPerson) -> Tuple[bool, float]:
        """Like calculate_wrist_motion_variance, but returns the raw variance
        value (and whether enough history exists to trust it) instead of a
        writing/not-writing verdict — lets callers apply their own threshold
        (e.g. a stillness requirement) rather than only the writing one."""
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
        total_variance = np.var(wrist_array[:, 0]) + np.var(wrist_array[:, 1])
        return True, float(total_variance)


# ── Temporal Behavior Engine ─────────────────────────────────────────────────
class TemporalBehaviorEngine:
    LOW_CONFIDENCE_THRESHOLD     = 0.4
    HEAD_DOWN_CONSECUTIVE_FRAMES = 2
    ALERT_CONFIRM_FRAMES  = 3    # was 10 — alerts now confirm in ~0.3s at 10fps
    ALERT_MAJORITY        = 0.67
    NORMAL_CONFIRM_FRAMES = 3    # was 5

    # 'hand_raised' gets its own, shorter/looser confirm window than
    # ALERT_POSES (phone/eating). Those guard against false incident
    # reports/emails and should stay strict; a missed or late-confirmed
    # hand-raise has no equivalent cost, and real hand-raises are often
    # only held 1-3s, which the shared ALERT window (at heavy-detection
    # cadence ~2fps) can miss entirely or confirm only as the hand is
    # already coming down.
    HAND_RAISE_CONFIRM_FRAMES = 3
    HAND_RAISE_MAJORITY       = 0.6

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

    def evaluate_final_behavior(self, person: TrackedPerson, raw_behavior: str, raw_confidence: float) -> Tuple[str, float]:
        person.behavior_history.append(raw_behavior)
        prev_raw_confidence = person.last_raw_confidence   # capture BEFORE overwriting below
        person.last_raw_confidence = raw_confidence

        final_behavior = person.last_final_behavior
        history = list(person.behavior_history)

        if history:
            last = history[-1]

            if last == 'hand_raised':
                # Shorter, looser confirm window than ALERT_POSES — see
                # HAND_RAISE_CONFIRM_FRAMES/HAND_RAISE_MAJORITY comment above.
                n = min(self.HAND_RAISE_CONFIRM_FRAMES, len(history))
                recent = history[-n:]
                raise_count = sum(1 for h in recent if h == last)
                if raise_count / n >= self.HAND_RAISE_MAJORITY:
                    final_behavior = last
            else:
                # Step 1: always check every ALERT_POSES label against the
                # strict confirm window, regardless of what the single
                # newest frame happened to read. This used to only run when
                # `last` itself was an alert label — so a phone-use signal
                # that was genuinely dominant across recent frames could
                # still fail to confirm just because the latest sample
                # flickered to something else (a threshold-edge posture
                # read differently for one frame is enough to trigger this).
                # Checking every alert candidate here, every time, removes
                # that dependency on the newest frame specifically.
                promoted = False
                n = min(self.ALERT_CONFIRM_FRAMES, len(history))
                recent_alert_window = history[-n:]
                for candidate in ALERT_POSES:
                    count = sum(1 for h in recent_alert_window if h == candidate)
                    if count / n >= self.ALERT_MAJORITY:
                        final_behavior = candidate
                        promoted = True
                        break  # at most one alert label can dominate a window

                # Step 2: only reachable if no alert label was just
                # confirmed above. This path handles focused/distracted/
                # not_visible and deliberately can NEVER promote an
                # ALERT_POSES label — that's Step 1's job, at Step 1's
                # stricter bar. Previously this branch used a looser >50%
                # majority that had no such restriction, which is what let
                # "distracted" out-vote a real, frequent "using_phone"
                # signal whenever the newest single frame wasn't itself
                # "using_phone".
                if not promoted:
                    if len(history) >= self.NORMAL_CONFIRM_FRAMES:
                        recent = history[-self.NORMAL_CONFIRM_FRAMES:]
                        if len(set(recent)) == 1 and recent[0] not in ALERT_POSES:
                            final_behavior = recent[0]
                        else:
                            window = history[-8:]
                            top_label, top_count = Counter(window).most_common(1)[0]
                            if top_label not in ALERT_POSES and top_count / len(window) > 0.5:
                                final_behavior = top_label
                    elif last not in ALERT_POSES:
                        final_behavior = last

        person.last_final_behavior = final_behavior

        # Report the confidence that actually belongs to the confirmed
        # label: if this frame's raw reading matches what we're reporting,
        # use its confidence; otherwise use whatever confidence was last
        # seen BEFORE this frame overwrote it (captured above), not the
        # value that was just replaced — using the post-overwrite value
        # here always equaled the current frame's confidence regardless of
        # which branch ran, silently defeating the fallback.
        confidence = raw_confidence if final_behavior == history[-1] else prev_raw_confidence
        confidence = confidence or 0.75
        return final_behavior, confidence