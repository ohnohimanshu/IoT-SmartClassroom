import numpy as np
from typing import Tuple
from classroom_monitor.behavior_detection_core import TrackedPerson


class HandRaiseDetector:
    # Wrist must be this many pixels above the shoulder to count as raised
    # 30px was too coarse at varying camera distances — use relative threshold instead
    WRIST_ABOVE_SHOULDER_FRACTION = 0.08   # fraction of bbox_h above shoulder

    @staticmethod
    def detect_hand_raise(person: TrackedPerson) -> Tuple[bool, float]:
        if person.keypoints is None or person.keypoints.size == 0:
            return False, 0.0
        try:
            kp = person.keypoints
            x1, y1, x2, y2 = person.bbox
            bbox_h = y2 - y1
            if bbox_h <= 0:
                return False, 0.0

            # Pixel threshold: wrist must be this far above its shoulder
            pixel_thresh = bbox_h * HandRaiseDetector.WRIST_ABOVE_SHOULDER_FRACTION

            pairs = [
                (kp[5] if len(kp) > 5 else None,  kp[7] if len(kp) > 7 else None,  kp[9]  if len(kp) > 9  else None),   # left: shoulder, elbow, wrist
                (kp[6] if len(kp) > 6 else None,  kp[8] if len(kp) > 8 else None,  kp[10] if len(kp) > 10 else None),   # right
            ]

            for shoulder, elbow, wrist in pairs:
                if shoulder is None or elbow is None or wrist is None:
                    continue
                if len(shoulder) < 3 or len(elbow) < 3 or len(wrist) < 3:
                    continue
                if shoulder[2] < 0.4 or elbow[2] < 0.4 or wrist[2] < 0.4:
                    continue
                if wrist[0] == 0.0:
                    continue
                # In image coords y increases downward, so raised wrist has smaller y.
                # Require the arm to actually be extended upward — elbow below (i.e.
                # larger y than) the wrist — not just the wrist happening to be near
                # shoulder height, which incidental gestures (adjusting hair, elbow
                # resting on desk with hand near shoulder) can also produce.
                if wrist[1] < shoulder[1] - pixel_thresh and elbow[1] > wrist[1]:
                    return True, float(wrist[2])

        except Exception as exc:
            print(f'[HAND RAISE] error track {person.track_id}: {exc}')
        return False, 0.0