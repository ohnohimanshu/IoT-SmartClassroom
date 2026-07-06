import numpy as np
from typing import Tuple
from classroom_monitor.behavior_detection_core import TrackedPerson


class HeadPoseDetector:
    LOW_CONFIDENCE_THRESHOLD = 0.4       # was 0.5 — more lenient keypoint acceptance
    HEAD_DOWN_CONSECUTIVE_FRAMES = 2     # was 3 — detect head-down sooner

    def __init__(self):
        self.low_confidence_counters = {}

    def calculate_head_pose(self, person: TrackedPerson) -> str:
        kp = person.keypoints
        if kp is None or kp.size == 0 or len(kp) < 3:
            return 'not_visible'
        try:
            nose, left_eye, right_eye = kp[0], kp[1], kp[2]
            if any(len(pt) < 3 for pt in (nose, left_eye, right_eye)):
                return 'not_visible'

            tid = person.track_id
            nose_ok      = nose[2]      >= self.LOW_CONFIDENCE_THRESHOLD and nose[0]      != 0.0
            left_eye_ok  = left_eye[2]  >= self.LOW_CONFIDENCE_THRESHOLD and left_eye[0]  != 0.0
            right_eye_ok = right_eye[2] >= self.LOW_CONFIDENCE_THRESHOLD and right_eye[0] != 0.0

            # Head hidden / face down
            head_down_suspected = (not nose_ok) or (not left_eye_ok and not right_eye_ok)
            if head_down_suspected:
                self.low_confidence_counters[tid] = self.low_confidence_counters.get(tid, 0) + 1
                return 'head_down' if self.low_confidence_counters[tid] >= self.HEAD_DOWN_CONSECUTIVE_FRAMES else 'focused'
            else:
                self.low_confidence_counters.pop(tid, None)

            if left_eye_ok and right_eye_ok:
                inter_eye = np.linalg.norm(left_eye[:2] - right_eye[:2])
                if inter_eye > 0.1:
                    eyes_center_x = (left_eye[0] + right_eye[0]) / 2.0

                    # Yaw: nose offset from eye-centre (was 0.6, lowered to 0.45)
                    yaw_ratio = abs(nose[0] - eyes_center_x) / inter_eye
                    if yaw_ratio > 0.45:
                        return 'looking_away'

                    # Pitch: nose dropped below eye-line (was 0.6, lowered to 0.45)
                    drop_ratio = (nose[1] - (left_eye[1] + right_eye[1]) / 2) / inter_eye
                    if drop_ratio > 0.45:
                        return 'head_down'

                    # Very small inter-eye distance → profile / looking away
                    bbox_h = person.bbox[3] - person.bbox[1]
                    if bbox_h > 10 and inter_eye < bbox_h * 0.07:
                        return 'looking_away'
            else:
                # Only one eye visible — likely profile, treat as looking away
                if nose_ok:
                    return 'looking_away'

            return 'focused'
        except Exception as exc:
            print(f'[HEAD] pose error track {person.track_id}: {exc}')
            return 'focused'

    def is_head_down_like(self, person: TrackedPerson, head_pose: str) -> bool:
        if head_pose == 'head_down':
            return True
        kp = person.keypoints
        if kp is None or len(kp) < 3:
            return False
        try:
            nose, left_eye, right_eye = kp[0], kp[1], kp[2]
            if (len(nose) >= 3 and len(left_eye) >= 3 and len(right_eye) >= 3 and
                    nose[2] >= 0.4 and left_eye[2] >= 0.4 and right_eye[2] >= 0.4):
                inter_eye = np.linalg.norm(left_eye[:2] - right_eye[:2])
                if inter_eye > 0.1:
                    drop_ratio = (nose[1] - (left_eye[1] + right_eye[1]) / 2) / inter_eye
                    return drop_ratio > 0.3   # was 0.4
        except Exception:
            pass
        return False
