import numpy as np
from typing import Tuple
from classroom_monitor.behavior_detection_core import TrackedPerson


class HeadPoseDetector:
    """
    Determines head pose (focused / looking_away / head_down / not_visible)
    from YOLO-pose keypoints (nose, left_eye, right_eye).

    Design principle: a single frame of "nose dropped" or "nose offset" is
    NORMAL classroom behavior (writing, glancing at the board, a blink).
    Only a SUSTAINED deviation should be treated as a real signal — but
    real-time pose estimation is noisy frame to frame (motion blur, brief
    confidence dips, a keypoint jittering a few pixels), so persistence is
    tracked with LEAKY counters: a match increments, a miss only decays by
    one instead of resetting to zero. This tolerates the occasional bad
    frame in the middle of genuinely sustained behavior, unlike a strict
    consecutive-frame counter which one noisy frame can wipe out entirely.
    """

    # Keypoint confidence required to trust a landmark.
    LOW_CONFIDENCE_THRESHOLD = 0.5

    # Leaky-counter thresholds (not raw frame counts — see class docstring).
    # At process_fps=10, ~10 net frames of "credit" roughly corresponds to
    # ~1s of real sustained behavior, tolerant of the odd noisy frame.
    HEAD_DOWN_THRESHOLD = 10
    LOOKING_AWAY_THRESHOLD = 10
    COUNTER_CAP = 20  # prevents unbounded buildup while someone stays down/away

    # Yaw: nose horizontal offset from eye-center, relative to inter-eye
    # distance. Looking at a side-mounted board, or just a normal seated
    # angle relative to the camera, commonly produces yaw in the 0.35-0.55
    # range — that is NOT looking away.
    YAW_LOOKING_AWAY_RATIO = 0.65

    # Pitch: nose vertical drop below the eye-line, relative to inter-eye
    # distance. Writing produces a real, sustained pitch drop that looks
    # identical to "distracted head down" from pose alone.
    PITCH_HEAD_DOWN_RATIO = 0.6

    def __init__(self):
        self.head_down_counters = {}
        self.looking_away_counters = {}

    @classmethod
    def _leaky_update(cls, counters: dict, tid, matched: bool, threshold: int) -> bool:
        """Increment on match, decay by 1 on miss (floor 0). Returns True
        once the counter has crossed `threshold`."""
        val = counters.get(tid, 0)
        val = min(val + 1, cls.COUNTER_CAP) if matched else max(val - 1, 0)
        if val == 0:
            counters.pop(tid, None)
        else:
            counters[tid] = val
        return val >= threshold

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

            # Head hidden / face down far enough that eyes aren't visible.
            head_down_suspected = (not nose_ok) or (not left_eye_ok and not right_eye_ok)
            if head_down_suspected:
                self._leaky_update(self.looking_away_counters, tid, False, self.LOOKING_AWAY_THRESHOLD)
                is_down = self._leaky_update(self.head_down_counters, tid, True, self.HEAD_DOWN_THRESHOLD)
                return 'head_down' if is_down else 'focused'

            if left_eye_ok and right_eye_ok:
                inter_eye = np.linalg.norm(left_eye[:2] - right_eye[:2])
                if inter_eye > 0.1:
                    eyes_center_x = (left_eye[0] + right_eye[0]) / 2.0

                    yaw_ratio = abs(nose[0] - eyes_center_x) / inter_eye
                    drop_ratio = (nose[1] - (left_eye[1] + right_eye[1]) / 2) / inter_eye

                    bbox_h = person.bbox[3] - person.bbox[1]
                    is_profile = bbox_h > 10 and inter_eye < bbox_h * 0.07
                    is_yaw_away = yaw_ratio > self.YAW_LOOKING_AWAY_RATIO or is_profile
                    is_pitch_down = drop_ratio > self.PITCH_HEAD_DOWN_RATIO

                    is_away = self._leaky_update(self.looking_away_counters, tid, is_yaw_away, self.LOOKING_AWAY_THRESHOLD)
                    is_down = self._leaky_update(self.head_down_counters, tid, is_pitch_down, self.HEAD_DOWN_THRESHOLD)

                    if is_yaw_away and is_away:
                        return 'looking_away'
                    if is_pitch_down and is_down:
                        return 'head_down'
                    return 'focused'
            else:
                # Only one eye visible — likely a turned/profile head.
                self._leaky_update(self.head_down_counters, tid, False, self.HEAD_DOWN_THRESHOLD)
                is_away = self._leaky_update(self.looking_away_counters, tid, nose_ok, self.LOOKING_AWAY_THRESHOLD)
                return 'looking_away' if (nose_ok and is_away) else 'focused'

            self._leaky_update(self.head_down_counters, tid, False, self.HEAD_DOWN_THRESHOLD)
            self._leaky_update(self.looking_away_counters, tid, False, self.LOOKING_AWAY_THRESHOLD)
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
                    nose[2] >= self.LOW_CONFIDENCE_THRESHOLD and
                    left_eye[2] >= self.LOW_CONFIDENCE_THRESHOLD and
                    right_eye[2] >= self.LOW_CONFIDENCE_THRESHOLD):
                inter_eye = np.linalg.norm(left_eye[:2] - right_eye[:2])
                if inter_eye > 0.1:
                    drop_ratio = (nose[1] - (left_eye[1] + right_eye[1]) / 2) / inter_eye
                    return drop_ratio > 0.4
        except Exception:
            pass
        return False