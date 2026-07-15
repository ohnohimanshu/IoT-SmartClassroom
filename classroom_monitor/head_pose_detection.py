import numpy as np
from typing import Tuple
from classroom_monitor.behavior_detection_core import TrackedPerson


class HeadPoseDetector:
    """
    Determines head pose (focused / looking_away / head_down / not_visible)
    from YOLO-pose keypoints (nose, left_eye, right_eye).

    Design principle: a single frame of "nose dropped" or "nose offset" is
    NORMAL classroom behavior (writing in a notebook, glancing at the board,
    a blink, a brief head tilt). Only a SUSTAINED deviation over roughly a
    second or more should be treated as a real signal. Every path below
    that can classify someone as distracted requires N consecutive frames,
    not just one.
    """

    # Keypoint confidence required to trust a landmark.
    LOW_CONFIDENCE_THRESHOLD = 0.5

    # Frame counts (not seconds) a pose must persist before it's reported.
    # At process_fps frames/sec, N frames ~= N / process_fps seconds.
    # ProductionStreamProcessor runs at process_fps=10, so 15 frames ~= 1.5s.
    # If you change process_fps, scale these proportionally.
    HEAD_DOWN_CONSECUTIVE_FRAMES = 15
    LOOKING_AWAY_CONSECUTIVE_FRAMES = 15

    # Yaw: nose horizontal offset from eye-center, relative to inter-eye
    # distance. Looking at a side-mounted board, or just a normal seated
    # angle relative to the camera, commonly produces yaw in the 0.35-0.55
    # range — that is NOT looking away. Keep this high enough to only catch
    # a genuine turn of the head away from the front of the room.
    YAW_LOOKING_AWAY_RATIO = 0.65

    # Pitch: nose vertical drop below the eye-line, relative to inter-eye
    # distance. Writing in a notebook produces a real, sustained pitch drop
    # that looks identical to "distracted head down" from pose alone — the
    # consecutive-frame requirement above is what keeps a normal 1-2s glance
    # down from being flagged, while a student who stays down (phone, doze,
    # prolonged disengagement) still gets caught.
    PITCH_HEAD_DOWN_RATIO = 0.6

    def __init__(self):
        self.head_down_counters = {}
        self.looking_away_counters = {}

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
            # This is a strong signal, but still require sustained duration.
            head_down_suspected = (not nose_ok) or (not left_eye_ok and not right_eye_ok)
            if head_down_suspected:
                self.looking_away_counters.pop(tid, None)
                return self._bump(self.head_down_counters, tid, self.HEAD_DOWN_CONSECUTIVE_FRAMES, 'head_down')

            if left_eye_ok and right_eye_ok:
                inter_eye = np.linalg.norm(left_eye[:2] - right_eye[:2])
                if inter_eye > 0.1:
                    eyes_center_x = (left_eye[0] + right_eye[0]) / 2.0

                    yaw_ratio = abs(nose[0] - eyes_center_x) / inter_eye
                    drop_ratio = (nose[1] - (left_eye[1] + right_eye[1]) / 2) / inter_eye

                    bbox_h = person.bbox[3] - person.bbox[1]
                    is_profile = bbox_h > 10 and inter_eye < bbox_h * 0.07

                    if yaw_ratio > self.YAW_LOOKING_AWAY_RATIO or is_profile:
                        self.head_down_counters.pop(tid, None)
                        return self._bump(self.looking_away_counters, tid,
                                           self.LOOKING_AWAY_CONSECUTIVE_FRAMES, 'looking_away')
                    self.looking_away_counters.pop(tid, None)

                    if drop_ratio > self.PITCH_HEAD_DOWN_RATIO:
                        return self._bump(self.head_down_counters, tid,
                                           self.HEAD_DOWN_CONSECUTIVE_FRAMES, 'head_down')
                    self.head_down_counters.pop(tid, None)
                    return 'focused'
            else:
                # Only one eye visible — likely a turned/profile head.
                self.head_down_counters.pop(tid, None)
                if nose_ok:
                    return self._bump(self.looking_away_counters, tid,
                                       self.LOOKING_AWAY_CONSECUTIVE_FRAMES, 'looking_away')
                self.looking_away_counters.pop(tid, None)
                return 'focused'

            self.head_down_counters.pop(tid, None)
            self.looking_away_counters.pop(tid, None)
            return 'focused'
        except Exception as exc:
            print(f'[HEAD] pose error track {person.track_id}: {exc}')
            return 'focused'

    @staticmethod
    def _bump(counters: dict, tid, threshold: int, label: str) -> str:
        """Increment a track's consecutive-frame counter and only report
        `label` once it crosses `threshold`; otherwise treat as focused."""
        counters[tid] = counters.get(tid, 0) + 1
        return label if counters[tid] >= threshold else 'focused'

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