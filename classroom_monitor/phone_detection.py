import numpy as np
from typing import List, Tuple, Optional
from classroom_monitor.behavior_detection_core import TrackedPerson, SharedHelpers
from classroom_monitor.head_pose_detection import HeadPoseDetector


class PhoneDetector:
    PHONE_LAP_HEIGHT_FRACTION  = 0.55   # was 0.65 — catch higher lap positions
    PHONE_CUPPED_SPREAD_MAX    = 0.22   # was 0.15 — slightly wider cupped grip
    PHONE_SINGLE_HAND_Y_MIN    = 0.45   # wrist-relative-y above this = torso zone
    PHONE_SINGLE_HAND_Y_MAX    = 0.85
    WRITING_DESK_Y_MIN         = 0.70   # wrist above this threshold = desk level

    def __init__(self):
        self.head_pose_detector = HeadPoseDetector()

    def detect_phone_usage(self, person: TrackedPerson, phone_detections: List[Tuple],
                           head_pose: str, book_detections: Optional[List[Tuple]] = None) -> Tuple[bool, float]:
        book_detections = book_detections or []
        x1, y1, x2, y2 = person.bbox
        bbox_h = y2 - y1
        bbox_w = x2 - x1
        if bbox_h <= 0 or bbox_w <= 0:
            return False, 0.0

        # Suppress phone detection only when there is clear, high-variance writing motion
        is_writing, _ = SharedHelpers.calculate_wrist_motion_variance(person)
        if is_writing:
            return False, 0.0

        head_is_down = (head_pose in ('head_down', 'looking_away')) or \
                       self.head_pose_detector.is_head_down_like(person, head_pose)

        def _book_near(pt) -> bool:
            return SharedHelpers.point_near_book(pt, bbox_h, book_detections)

        def _wrist_ok(w, min_conf=0.4) -> bool:
            return w is not None and len(w) >= 3 and w[2] >= min_conf and w[0] != 0.0

        # ── Path 1: YOLO detected a phone object ─────────────────────────────
        if phone_detections:
            for (px1, py1, px2, py2, conf) in phone_detections:
                if conf < 0.25:
                    continue

                pc  = np.array([(px1 + px2) / 2.0, (py1 + py2) / 2.0])
                ph  = py2 - py1
                pw  = px2 - px1

                # Sanity-check phone bbox size relative to person
                if max(pw, ph) / bbox_h > 0.7 or min(pw, ph) / bbox_h < 0.03:
                    continue

                # Phone must be inside or near the person bbox
                in_person_x = px1 < x2 + 20 and px2 > x1 - 20
                in_person_y = py1 < y2 + 20 and py2 > y1 - 20
                if not (in_person_x and in_person_y):
                    continue

                # Check proximity to any wrist or elbow (indices 7,8=elbow 9,10=wrist)
                if person.keypoints is not None and len(person.keypoints) > 10:
                    for idx in (9, 10, 7, 8):
                        if idx >= len(person.keypoints):
                            continue
                        w = person.keypoints[idx]
                        if _wrist_ok(w, 0.4) and np.linalg.norm(w[:2] - pc) / bbox_h < 0.35:
                            print(f'[PHONE] Person {person.track_id}: YOLO phone near joint {idx}, conf={conf:.2f}')
                            return True, float(conf)

                # If keypoints are unreliable but phone bbox overlaps person strongly, trust YOLO
                if conf >= 0.55:
                    overlap_x = max(0, min(px2, x2) - max(px1, x1))
                    overlap_y = max(0, min(py2, y2) - max(py1, y1))
                    overlap_area = overlap_x * overlap_y
                    phone_area   = max((px2 - px1) * (py2 - py1), 1)
                    if overlap_area / phone_area > 0.5:
                        print(f'[PHONE] Person {person.track_id}: high-conf YOLO phone inside bbox, conf={conf:.2f}')
                        return True, float(conf)

        # ── Path 2: Heuristic (no YOLO phone, use pose only) ─────────────────
        if person.keypoints is None or person.keypoints.size == 0 or len(person.keypoints) <= 10:
            return False, 0.0

        try:
            left_wrist  = person.keypoints[9]
            right_wrist = person.keypoints[10]
            lap_thresh  = y1 + bbox_h * self.PHONE_LAP_HEIGHT_FRACTION
            desk_thresh = y1 + bbox_h * self.WRITING_DESK_Y_MIN

            wrist_pts = [w[:2] for w in (left_wrist, right_wrist) if _wrist_ok(w, 0.5)]

            if not wrist_pts:
                return False, 0.0

            # Single-hand in torso/lap zone with head down
            if head_is_down and len(wrist_pts) >= 1:
                for w in wrist_pts:
                    rel_y = (w[1] - y1) / bbox_h
                    rel_x_offset = abs(w[0] - (x1 + x2) / 2.0) / bbox_w

                    # Wrist in mid-torso to lap zone, centred (not at desk edge)
                    if (self.PHONE_SINGLE_HAND_Y_MIN <= rel_y <= self.PHONE_SINGLE_HAND_Y_MAX
                            and rel_x_offset < 0.35
                            and w[1] < desk_thresh   # not at desk level
                            and not _book_near(w)):
                        print(f'[PHONE] Person {person.track_id}: single-hand torso heuristic')
                        return True, 0.60

            # Two cupped hands at lap level with head down
            if head_is_down and len(wrist_pts) == 2:
                both_low = all(w[1] > lap_thresh for w in wrist_pts)
                spread   = np.linalg.norm(wrist_pts[0] - wrist_pts[1]) / bbox_h
                if both_low and spread < self.PHONE_CUPPED_SPREAD_MAX:
                    if not any(_book_near(w) for w in wrist_pts):
                        print(f'[PHONE] Person {person.track_id}: cupped hands at lap')
                        return True, 0.55

        except Exception as exc:
            print(f'[PHONE] Heuristic error: {exc}')

        return False, 0.0
