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
    normalized_pose_history: deque = field(default_factory=lambda: deque(maxlen=16)) # For LSTM
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
            if np.linalg.norm(np.array(pt) - bc) / bbox_h < 0.26:
                return True
        return False

    @staticmethod
    def normalize_keypoints(keypoints: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Normalize 17 keypoints (x,y) to center (0,0) and scale (torso length).
        If missing, fill with 0.
        Returns flat array of 34 floats.
        """
        out = np.zeros(34, dtype=np.float32)
        if keypoints is None or len(keypoints) < 17:
            return out
            
        # Try to find a stable center (e.g., hips midpoint or shoulders midpoint)
        left_shoulder = keypoints[5][:2] if len(keypoints[5]) >= 2 and keypoints[5][0] != 0 else None
        right_shoulder = keypoints[6][:2] if len(keypoints[6]) >= 2 and keypoints[6][0] != 0 else None
        left_hip = keypoints[11][:2] if len(keypoints[11]) >= 2 and keypoints[11][0] != 0 else None
        right_hip = keypoints[12][:2] if len(keypoints[12]) >= 2 and keypoints[12][0] != 0 else None
        
        valid_centers = []
        if left_shoulder is not None and right_shoulder is not None:
            valid_centers.append((left_shoulder + right_shoulder) / 2)
        if left_hip is not None and right_hip is not None:
            valid_centers.append((left_hip + right_hip) / 2)
            
        if valid_centers:
            center = np.mean(valid_centers, axis=0)
        else:
            x1, y1, x2, y2 = bbox
            center = np.array([(x1 + x2) / 2.0, y1 + (y2 - y1) * 0.5])
            
        # Scale by bounding box height
        bbox_h = max(bbox[3] - bbox[1], 1.0)
        scale = bbox_h
        
        for i in range(17):
            if i < len(keypoints) and keypoints[i][0] != 0:
                out[i*2] = (keypoints[i][0] - center[0]) / scale
                out[i*2+1] = (keypoints[i][1] - center[1]) / scale
                
        return out

    @staticmethod
    def hands_near_book(person: 'TrackedPerson', book_detections: List[Tuple]) -> bool:
        """True if either wrist is near a detected book/notebook. Used to
        distinguish a student writing (head down, hand near notebook) from
        a student who is actually distracted/disengaged (head down, hands
        elsewhere) — pose alone can't tell these apart, this can."""
        if not book_detections or person.keypoints is None or len(person.keypoints) <= 10:
            return False
        bbox_h = person.bbox[3] - person.bbox[1]
        if bbox_h <= 0:
            return False
        for idx in (9, 10):
            if idx >= len(person.keypoints):
                continue
            w = person.keypoints[idx]
            if len(w) >= 3 and w[2] >= 0.4 and w[0] != 0.0:
                if SharedHelpers.point_near_book(w[:2], bbox_h, book_detections):
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
        bbox_h = person.bbox[3] - person.bbox[1]
        normalized_variance = (variance_x + variance_y) / max(bbox_h ** 2, 1)
        # Was 0.0015 originally (too permissive — ordinary fidgeting or a
        # phone-scrolling thumb read as "writing" and silently vetoed both
        # distracted AND phone-use detection). Raising it to 0.0035 then
        # overshot the other way — real, careful handwriting often has
        # smaller wrist motion than that, and hands_near_book essentially
        # never fires as a backstop (COCO's generic "book" class rarely
        # detects an open notebook at a normal writing angle), so genuine
        # writers started getting mislabeled "distracted" instead. 0.0022
        # is a middle point; treat it as a starting value and nudge it
        # after watching a session against your own footage/camera angle.
        is_writing = normalized_variance > 0.0022
        confidence = min(normalized_variance / 0.005, 0.95)
        return is_writing, confidence


# ── Temporal Behavior Engine ─────────────────────────────────────────────────
class TemporalBehaviorEngine:
    LOW_CONFIDENCE_THRESHOLD     = 0.4
    HEAD_DOWN_CONSECUTIVE_FRAMES = 2
    # Both of these used to re-confirm behavior over several more frames on
    # top of the leaky-counter hysteresis already applied upstream in
    # head_pose_detection.py / phone_detection.py. That stacked two
    # independent smoothing delays in series. The leaky counters already
    # guarantee a stable, debounced raw_behavior signal by the time it gets
    # here, so this layer only needs to catch a single additional frame of
    # agreement, not re-run its own multi-frame vote.
    ALERT_CONFIRM_FRAMES  = 1    # was 3
    NORMAL_CONFIRM_FRAMES = 2    # was 3

    def __init__(self):
        self.tracked_people: Dict[int, TrackedPerson] = {}
        self.cleanup_threshold = 8.0
        self.low_confidence_counters: Dict[int, int] = {}
        self.lock = threading.Lock()

    def update_person(self, track_id: int, bbox: Tuple, keypoints: Optional[np.ndarray], timestamp: float, aux_features: Optional[np.ndarray] = None):
        with self.lock:
            if track_id not in self.tracked_people:
                print(f'[TRACK] New track_id {track_id} created')
                p = TrackedPerson(
                    track_id=track_id, bbox=bbox, keypoints=keypoints, last_seen=timestamp)
                self.tracked_people[track_id] = p
            else:
                p = self.tracked_people[track_id]
                p.bbox, p.keypoints, p.last_seen = bbox, keypoints, timestamp

            if keypoints is not None and keypoints.size > 0:
                p.keypoint_history.append((timestamp, keypoints.copy()))
                norm_kp = SharedHelpers.normalize_keypoints(keypoints, bbox)
                
                if aux_features is None:
                    aux_features = np.zeros(4, dtype=np.float32)
                
                # Combine normalized keypoints (34) with aux features (4) = 38
                full_feature = np.concatenate([norm_kp, aux_features])
                p.normalized_pose_history.append(full_feature)

    def cleanup_stale(self, current_time: float):
        with self.lock:
            stale = [tid for tid, p in self.tracked_people.items()
                     if current_time - p.last_seen > self.cleanup_threshold]
            for tid in stale:
                del self.tracked_people[tid]
                self.low_confidence_counters.pop(tid, None)

    def evaluate_final_behavior(self, person: TrackedPerson, raw_behavior: str, raw_confidence: float) -> Tuple[str, float]:
        prev_raw_confidence = person.last_raw_confidence
        person.behavior_history.append(raw_behavior)

        final_behavior = person.last_final_behavior
        history = list(person.behavior_history)

        if len(history) >= 1:
            last = history[-1]
            if last in ALERT_POSES or last == 'hand_raised':
                final_behavior = last
            elif person.last_final_behavior in ALERT_POSES or person.last_final_behavior == 'hand_raised':
                # Hold alert state until N consecutive non-alert frames occur
                recent = history[-6:]
                if len(recent) >= 5 and all(h not in ALERT_POSES and h != 'hand_raised' for h in recent):
                    final_behavior = last
                else:
                    final_behavior = person.last_final_behavior
            else:
                if len(history) >= self.NORMAL_CONFIRM_FRAMES:
                    recent = history[-self.NORMAL_CONFIRM_FRAMES:]
                    if len(set(recent)) == 1:
                        final_behavior = recent[0]
                    else:
                        window = history[-8:]
                        top_label, top_count = Counter(window).most_common(1)[0]
                        if top_count / len(window) > 0.5:
                            final_behavior = top_label
                else:
                    final_behavior = last

        person.last_final_behavior = final_behavior

        confidence = raw_confidence if final_behavior == raw_behavior else prev_raw_confidence
        confidence = confidence or 0.75
        person.last_raw_confidence = raw_confidence
        return final_behavior, confidence