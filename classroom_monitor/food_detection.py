import numpy as np
from typing import List, Tuple
from classroom_monitor.behavior_detection_core import TrackedPerson

try:
    # Optional: used to guard the low-evidence fallback branch against
    # food-on-desk-near-book false positives. If unavailable, the fallback
    # simply stays disabled (see detect_eating below) rather than firing
    # without this check.
    from classroom_monitor.shared_helpers import SharedHelpers
    _HAS_BOOK_CHECK = True
except ImportError:
    _HAS_BOOK_CHECK = False


class FoodDetector:
    # How close a wrist must be to the nose/mouth area (relative to bbox height).
    # Tune against real footage; 0.20 is a starting point, not a final value.
    EATING_HAND_TO_MOUTH_THRESHOLD = 0.20
    # How close a food bbox centre must be to the nose/mouth (relative to bbox height)
    FOOD_TO_NOSE_THRESHOLD         = 0.22
    # How close a wrist must be to a SPECIFIC food detection's centre (relative to
    # bbox height) to count as "that hand is holding that food". Tighter than the
    # general hand-to-mouth threshold since it's item-specific, not face-region-general.
    WRIST_TO_FOOD_THRESHOLD        = 0.18
    # Minimum YOLO confidence for a food detection to be considered at all
    FOOD_CONF_MIN                  = 0.40
    # Minimum confidence for the low-evidence (occluded-wrist) fallback path
    FALLBACK_FOOD_CONF_MIN         = 0.60
    FALLBACK_FOOD_TO_NOSE_THRESHOLD = 0.20

    @classmethod
    def detect_eating(cls, person: TrackedPerson, food_detections: List[Tuple]) -> Tuple[bool, float]:
        if not food_detections or person.keypoints is None or person.keypoints.size == 0:
            return False, 0.0
        try:
            kp     = person.keypoints
            nose   = kp[0]
            x1, y1, x2, y2 = person.bbox
            bbox_h = y2 - y1
            if bbox_h <= 0:
                return False, 0.0

            mouth_est = np.array([nose[0], nose[1] + bbox_h * 0.05])

            # Gather visible, confident wrist keypoints once (indices 9, 10 = wrists)
            wrists = []
            for idx in (9, 10):
                if len(kp) > idx:
                    w = kp[idx]
                    if len(w) >= 3 and w[2] >= 0.4 and w[0] != 0.0:
                        wrists.append(w[:2])

            nose_visible = len(nose) >= 3 and nose[2] >= 0.5

            best_food_conf = 0.0

            # Evaluate each food detection independently: it must (a) be near the
            # face AND (b) have a wrist near THAT SPECIFIC detection's centre.
            # This stops an unrelated hand-near-face gesture from combining with
            # a neighbor's food, or a desk object, to falsely confirm eating.
            for (fx1, fy1, fx2, fy2, conf) in food_detections:
                if conf < cls.FOOD_CONF_MIN:
                    continue

                fc = np.array([(fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0])

                near_face = (
                    np.linalg.norm(fc - nose[:2]) / bbox_h < cls.FOOD_TO_NOSE_THRESHOLD or
                    np.linalg.norm(fc - mouth_est) / bbox_h < cls.FOOD_TO_NOSE_THRESHOLD
                )
                if not near_face:
                    continue

                if not (nose_visible and wrists):
                    continue

                wrist_holds_this_food = any(
                    np.linalg.norm(w - fc) / bbox_h < cls.WRIST_TO_FOOD_THRESHOLD
                    for w in wrists
                )
                if wrist_holds_this_food:
                    return True, conf

                best_food_conf = max(best_food_conf, conf)

            # Low-evidence fallback: high-confidence food right at the face but no
            # wrist could be linked to it (e.g. wrist keypoint occluded). Only
            # fires if we can positively rule out a book/desk-surface context;
            # if that check isn't available, the fallback stays off rather than
            # firing unguarded.
            if _HAS_BOOK_CHECK and best_food_conf >= cls.FALLBACK_FOOD_CONF_MIN:
                for (fx1, fy1, fx2, fy2, conf) in food_detections:
                    if conf < cls.FALLBACK_FOOD_CONF_MIN:
                        continue
                    fc = np.array([(fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0])
                    if np.linalg.norm(fc - nose[:2]) / bbox_h < cls.FALLBACK_FOOD_TO_NOSE_THRESHOLD:
                        if not SharedHelpers.point_near_book(fc, person):
                            return True, conf

        except Exception as exc:
            print(f'[EATING] error track {person.track_id}: {exc}')
        return False, 0.0