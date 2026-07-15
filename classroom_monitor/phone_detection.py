import numpy as np
from typing import List, Tuple, Optional
from classroom_monitor.behavior_detection_core import TrackedPerson, SharedHelpers
from classroom_monitor.head_pose_detection import HeadPoseDetector


class PhoneDetector:
    """
    Two independent detection paths:

    Path 1 - YOLO actually detected a phone-shaped object near a hand.
             This is real visual evidence, so it only needs light smoothing
             to avoid one-off misclassifications.

    Path 2 - No phone object seen; guess from hand position + head pose
             alone. This is inherently ambiguous — a hand holding a
             notebook in the lap (no desk in frame, e.g. a lab with chairs
             only) looks geometrically identical to a hand holding a phone
             in the lap. Because of that ambiguity, Path 2 REQUIRES the
             pose to persist for a sustained number of frames before it's
             trusted, and leans harder on the book/notebook veto.
    """

    PHONE_LAP_HEIGHT_FRACTION  = 0.65   # wrist must be genuinely low (lap), not mid-torso
    PHONE_CUPPED_SPREAD_MAX    = 0.15   # tight cupped grip only — open notebook has wider spread
    PHONE_SINGLE_HAND_Y_MIN    = 0.45   # wrist-relative-y above this = torso zone
    PHONE_SINGLE_HAND_Y_MAX    = 0.85
    WRITING_DESK_Y_MIN         = 0.70   # wrist above this threshold = desk level (when a desk exists)

    # Path 2 (pose-heuristic) must hold for this many consecutive processed
    # frames before it's reported. At process_fps=10 this is ~0.8s — long
    # enough to filter a mid-stroke hand lift while writing, short enough
    # to still catch real, sustained phone use.
    HEURISTIC_CONSECUTIVE_FRAMES = 8

    # Path 1 (actual YOLO phone box) still needs a little smoothing to
    # avoid a single flickered misclassification, but far less than Path 2
    # since it's grounded in a real object detection, not a guess.
    YOLO_HIT_CONSECUTIVE_FRAMES = 2

    def __init__(self):
        self.head_pose_detector = HeadPoseDetector()
        self._heuristic_counters = {}
        self._yolo_counters = {}

    def detect_phone_usage(self, person: TrackedPerson, phone_detections: List[Tuple],
                           head_pose: str, book_detections: Optional[List[Tuple]] = None) -> Tuple[bool, float]:
        book_detections = book_detections or []
        x1, y1, x2, y2 = person.bbox
        bbox_h = y2 - y1
        bbox_w = x2 - x1
        tid = person.track_id
        if bbox_h <= 0 or bbox_w <= 0:
            return False, 0.0

        # Suppress phone detection when there is clear, high-variance writing motion
        is_writing, _ = SharedHelpers.calculate_wrist_motion_variance(person)
        if is_writing:
            self._heuristic_counters.pop(tid, None)
            self._yolo_counters.pop(tid, None)
            return False, 0.0

        head_is_down = (head_pose in ('head_down', 'looking_away')) or \
                       self.head_pose_detector.is_head_down_like(person, head_pose)

        def _book_near(pt) -> bool:
            return SharedHelpers.point_near_book(pt, bbox_h, book_detections)

        def _wrist_ok(w, min_conf=0.4) -> bool:
            return w is not None and len(w) >= 3 and w[2] >= min_conf and w[0] != 0.0

        # ── Path 1: YOLO detected a phone object ─────────────────────────────
        yolo_hit_conf = None
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
                            yolo_hit_conf = float(conf)
                            break

                if yolo_hit_conf is None and conf >= 0.55:
                    overlap_x = max(0, min(px2, x2) - max(px1, x1))
                    overlap_y = max(0, min(py2, y2) - max(py1, y1))
                    overlap_area = overlap_x * overlap_y
                    phone_area   = max((px2 - px1) * (py2 - py1), 1)
                    if overlap_area / phone_area > 0.5:
                        yolo_hit_conf = float(conf)

                if yolo_hit_conf is not None:
                    break

        if yolo_hit_conf is not None:
            self._heuristic_counters.pop(tid, None)
            self._yolo_counters[tid] = self._yolo_counters.get(tid, 0) + 1
            if self._yolo_counters[tid] >= self.YOLO_HIT_CONSECUTIVE_FRAMES:
                print(f'[PHONE] Person {tid}: YOLO phone confirmed, conf={yolo_hit_conf:.2f}')
                return True, yolo_hit_conf
            return False, 0.0
        else:
            self._yolo_counters.pop(tid, None)

        # ── Path 2: Heuristic (no YOLO phone, use pose only) ─────────────────
        if person.keypoints is None or person.keypoints.size == 0 or len(person.keypoints) <= 10:
            self._heuristic_counters.pop(tid, None)
            return False, 0.0

        try:
            left_wrist  = person.keypoints[9]
            right_wrist = person.keypoints[10]
            lap_thresh  = y1 + bbox_h * self.PHONE_LAP_HEIGHT_FRACTION
            desk_thresh = y1 + bbox_h * self.WRITING_DESK_Y_MIN

            wrist_pts = [w[:2] for w in (left_wrist, right_wrist) if _wrist_ok(w, 0.5)]
            both_wrists_for_veto = [w[:2] for w in (left_wrist, right_wrist) if _wrist_ok(w, 0.4)]

            if not wrist_pts:
                self._heuristic_counters.pop(tid, None)
                return False, 0.0

            heuristic_matched = False
            match_conf = 0.0

            # Single-hand in torso/lap zone with head down
            if head_is_down and len(wrist_pts) >= 1:
                for w in wrist_pts:
                    rel_y = (w[1] - y1) / bbox_h
                    rel_x_offset = abs(w[0] - (x1 + x2) / 2.0) / bbox_w

                    # Wrist in mid-torso to lap zone, centred (not at desk edge),
                    # and neither wrist is anywhere near a detected book/notebook.
                    if (self.PHONE_SINGLE_HAND_Y_MIN <= rel_y <= self.PHONE_SINGLE_HAND_Y_MAX
                            and rel_x_offset < 0.35
                            and w[1] < desk_thresh
                            and not any(_book_near(p) for p in both_wrists_for_veto)):
                        heuristic_matched = True
                        match_conf = 0.60
                        break

            # Two cupped hands at lap level with head down
            if not heuristic_matched and head_is_down and len(wrist_pts) == 2:
                both_low = all(w[1] > lap_thresh for w in wrist_pts)
                spread   = np.linalg.norm(wrist_pts[0] - wrist_pts[1]) / bbox_h
                if both_low and spread < self.PHONE_CUPPED_SPREAD_MAX:
                    if not any(_book_near(p) for p in both_wrists_for_veto):
                        heuristic_matched = True
                        match_conf = 0.55

            if not heuristic_matched:
                self._heuristic_counters.pop(tid, None)
                return False, 0.0

            self._heuristic_counters[tid] = self._heuristic_counters.get(tid, 0) + 1
            if self._heuristic_counters[tid] >= self.HEURISTIC_CONSECUTIVE_FRAMES:
                print(f'[PHONE] Person {tid}: sustained pose heuristic match, conf={match_conf:.2f}')
                return True, match_conf
            return False, 0.0

        except Exception as exc:
            print(f'[PHONE] Heuristic error: {exc}')

        return False, 0.0