import numpy as np
from typing import List, Tuple, Optional
from classroom_monitor.behavior_detection_core import TrackedPerson, SharedHelpers
from classroom_monitor.head_pose_detection import HeadPoseDetector


class PhoneDetector:
    """
    Single detection path: real object evidence only.

    A phone-shaped object (from your trained classroom_phone_yolo model,
    or any other detector feeding phone_detections) was seen near a hand.
    Confirmed over a short leaky-counter window for light smoothing
    against one-off misdetections, but the underlying signal is always
    real visual evidence — never a pose-only guess.

    Path 2 (guessing phone-vs-writing from wrist position + head pose
    alone, with no object seen) was removed. It was firing on ordinary
    writing far more often than on real phone use: it relied on a
    book/notebook proximity veto to tell the two apart, but that veto
    depends on COCO's generic "book" class, which essentially never
    detects an open spiral notebook or notepad at a normal writing angle.
    With that veto not actually vetoing anything, "wrist in torso/lap
    zone, head down" matched writing students just as easily as phone
    users, producing false "Using Phone" labels. Since a real trained
    phone detector exists (Path 1) and is reliable, it isn't worth
    keeping an ambiguous fallback that produced more false positives
    than confirmed true positives.

    Persistence uses a LEAKY counter (decay by 1 on a miss, not reset to
    0) rather than a strict consecutive-frame counter. Pose/detector
    output is noisy frame to frame, and a hard reset means one bad frame
    in the middle of sustained real phone use can wipe out the whole
    streak and the behavior never confirms.
    """

    # Leaky-counter threshold. At process_fps=10, ~2 net frames of
    # "credit" is roughly ~0.2s of real sustained match — short since
    # Path 1 evidence is already real object detections, not a guess.
    YOLO_HIT_THRESHOLD = 2
    COUNTER_CAP = 20

    def __init__(self):
        # Retained for API compatibility (head-pose based callers/signature
        # elsewhere); no longer used for phone-vs-writing disambiguation
        # since Path 2 was removed.
        self.head_pose_detector = HeadPoseDetector()
        self._yolo_counters = {}

    @classmethod
    def _leaky_update(cls, counters: dict, tid, matched: bool, threshold: int) -> bool:
        val = counters.get(tid, 0)
        val = min(val + 1, cls.COUNTER_CAP) if matched else max(val - 1, 0)
        if val == 0:
            counters.pop(tid, None)
        else:
            counters[tid] = val
        return val >= threshold

    def detect_phone_usage(self, person: TrackedPerson, phone_detections: List[Tuple],
                           head_pose: str, book_detections: Optional[List[Tuple]] = None) -> Tuple[bool, float]:
        # book_detections / head_pose kept in the signature for backward
        # compatibility with the caller in behavior_detection.py, but are
        # no longer used now that Path 2 (the only place that consumed
        # them) has been removed.
        x1, y1, x2, y2 = person.bbox
        bbox_h = y2 - y1
        bbox_w = x2 - x1
        tid = person.track_id
        if bbox_h <= 0 or bbox_w <= 0:
            return False, 0.0

        def _wrist_ok(w, min_conf=0.4) -> bool:
            return w is not None and len(w) >= 3 and w[2] >= min_conf and w[0] != 0.0

        # ── Path 1: real object evidence (only remaining path) ───────────────
        yolo_hit_conf = None
        if phone_detections:
            for (px1, py1, px2, py2, conf) in phone_detections:
                if conf < 0.25:
                    continue

                pc  = np.array([(px1 + px2) / 2.0, (py1 + py2) / 2.0])
                ph  = py2 - py1
                pw  = px2 - px1

                if max(pw, ph) / bbox_h > 0.7 or min(pw, ph) / bbox_h < 0.03:
                    continue

                in_person_x = px1 < x2 + 20 and px2 > x1 - 20
                in_person_y = py1 < y2 + 20 and py2 > y1 - 20
                if not (in_person_x and in_person_y):
                    continue

                if person.keypoints is not None and len(person.keypoints) > 10:
                    for idx in (9, 10, 7, 8):
                        if idx >= len(person.keypoints):
                            continue
                        w = person.keypoints[idx]
                        if _wrist_ok(w, 0.4) and np.linalg.norm(w[:2] - pc) / bbox_h < 0.45:
                            yolo_hit_conf = float(conf)
                            break

                if yolo_hit_conf is None and conf >= 0.45:
                    overlap_x = max(0, min(px2, x2) - max(px1, x1))
                    overlap_y = max(0, min(py2, y2) - max(py1, y1))
                    overlap_area = overlap_x * overlap_y
                    phone_area   = max((px2 - px1) * (py2 - py1), 1)
                    if overlap_area / phone_area > 0.5:
                        yolo_hit_conf = float(conf)

                if yolo_hit_conf is not None:
                    break

        yolo_confirmed = self._leaky_update(self._yolo_counters, tid, yolo_hit_conf is not None, self.YOLO_HIT_THRESHOLD)
        if yolo_hit_conf is not None and yolo_confirmed:
            print(f'[PHONE] Person {tid}: YOLO phone confirmed, conf={yolo_hit_conf:.2f}')
            return True, yolo_hit_conf

        return False, 0.0