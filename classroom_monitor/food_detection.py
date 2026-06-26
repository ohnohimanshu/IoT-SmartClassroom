import numpy as np
from typing import List, Tuple
from classroom_monitor.behavior_detection_core import TrackedPerson


class FoodDetector:
    # How close a wrist must be to the nose/mouth area (relative to bbox height)
    EATING_HAND_TO_MOUTH_THRESHOLD = 0.40   # was 0.15 — far too tight for real footage
    # How close food bbox centre must be to the nose (relative to bbox height)
    FOOD_TO_NOSE_THRESHOLD         = 0.35   # was 0.20
    # Minimum YOLO confidence for a food detection to be considered
    FOOD_CONF_MIN                  = 0.40   # was 0.75 — unreachably high

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

            # Also use mouth estimate: slightly below nose
            mouth_est = np.array([nose[0], nose[1] + bbox_h * 0.05])

            best_food_conf  = 0.0
            food_near_mouth = False

            for (fx1, fy1, fx2, fy2, conf) in food_detections:
                if conf < cls.FOOD_CONF_MIN:
                    continue
                fc = np.array([(fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0])
                # Check against both nose and estimated mouth
                if (np.linalg.norm(fc - nose[:2])    / bbox_h < cls.FOOD_TO_NOSE_THRESHOLD or
                        np.linalg.norm(fc - mouth_est) / bbox_h < cls.FOOD_TO_NOSE_THRESHOLD):
                    food_near_mouth = True
                    best_food_conf = max(best_food_conf, conf)

            if not food_near_mouth:
                return False, 0.0

            # Nose keypoint must be reasonably visible
            if len(nose) >= 3 and nose[2] >= 0.5:
                for idx in (9, 10):
                    if len(kp) > idx:
                        w = kp[idx]
                        if (len(w) >= 3 and w[2] >= 0.4 and w[0] != 0.0):
                            dist = np.linalg.norm(w[:2] - nose[:2]) / bbox_h
                            if dist < cls.EATING_HAND_TO_MOUTH_THRESHOLD:
                                return True, best_food_conf

            # Fallback: food very close to face even without clear wrist signal
            # (e.g. student holding food up but wrist keypoint occluded)
            if best_food_conf >= 0.60:
                for (fx1, fy1, fx2, fy2, conf) in food_detections:
                    if conf < 0.60:
                        continue
                    fc = np.array([(fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0])
                    if np.linalg.norm(fc - nose[:2]) / bbox_h < 0.20:
                        return True, conf

        except Exception as exc:
            print(f'[EATING] error track {person.track_id}: {exc}')
        return False, 0.0
