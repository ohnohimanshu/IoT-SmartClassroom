import numpy as np
from typing import List, Tuple, Optional, Dict
from classroom_monitor.behavior_detection_core import TrackedPerson, SharedHelpers
from classroom_monitor.head_pose_detection import HeadPoseDetector


class PhoneDetector:
    PHONE_LAP_HEIGHT_FRACTION  = 0.55   # was 0.65 — catch higher lap positions
    PHONE_CUPPED_SPREAD_MAX    = 0.22   # was 0.15 — slightly wider cupped grip
    PHONE_SINGLE_HAND_Y_MIN    = 0.45   # wrist-relative-y above this = torso zone
    PHONE_SINGLE_HAND_Y_MAX    = 0.85
    WRITING_DESK_Y_MIN         = 0.70   # wrist above this threshold = desk level

    # ── Anti-false-positive gate ──────────────────────────────────────────────
    # Path 2 (pose heuristic, no object evidence) must repeat for this many
    # consecutive heavy-detection calls on the SAME track before we trust it.
    # Path 1 (real YOLO phone bbox) is evidence-backed and is never gated —
    # only the "guess from posture alone" path needs debouncing.
    #
    # At ~2 heavy-detection calls/sec (see views.py heavy_every), a value of 3
    # means ~1.5s of sustained "head down + hand in lap zone" before we call
    # it phone use. A student writing in a notebook glances down for a frame
    # or two and moves their pen constantly — that will not survive 3
    # consecutive re-confirmations. Actual phone use is sustained and will.
    HEURISTIC_CONFIRM_FRAMES = 3

    # Set True temporarily while diagnosing missed/false detections. Prints
    # WHY a candidate was rejected on both the YOLO-object path and the
    # pose-only heuristic, so thresholds get tuned against real numbers
    # instead of guesswork. Turn off once detection quality is confirmed —
    # this is chatty at scale.
    DEBUG = True

    def __init__(self):
        self.head_pose_detector = HeadPoseDetector()
        # track_id -> consecutive heuristic-hit count
        self._heuristic_streak: Dict[int, int] = {}

    def cleanup_stale(self, active_track_ids: set):
        """Call once per frame with the set of currently-active track IDs so
        streak counters for tracks that have disappeared don't linger forever
        and don't leak onto a reused/new track ID."""
        stale = set(self._heuristic_streak.keys()) - active_track_ids
        for tid in stale:
            self._heuristic_streak.pop(tid, None)

    def _clear_streak(self, track_id):
        self._heuristic_streak.pop(track_id, None)

    def _confirm_or_hold(self, track_id, base_conf: float) -> Tuple[bool, float]:
        streak = self._heuristic_streak.get(track_id, 0) + 1
        self._heuristic_streak[track_id] = streak
        if streak >= self.HEURISTIC_CONFIRM_FRAMES:
            return True, base_conf
        return False, 0.0

    def detect_phone_usage(self, person: TrackedPerson, phone_detections: List[Tuple],
                           head_pose: str, book_detections: Optional[List[Tuple]] = None) -> Tuple[bool, float]:
        book_detections = book_detections or []
        x1, y1, x2, y2 = person.bbox
        bbox_h = y2 - y1
        bbox_w = x2 - x1
        if bbox_h <= 0 or bbox_w <= 0:
            return False, 0.0

        # Suppress phone detection only when there is clear, high-variance writing motion
        is_writing, writing_conf = SharedHelpers.calculate_wrist_motion_variance(person)
        if is_writing:
            self._clear_streak(person.track_id)
            if self.DEBUG:
                print(f'[PHONE-DEBUG] {person.track_id}: suppressed as writing motion '
                      f'(writing_conf={writing_conf}) — check this is correct for this student')
            return False, 0.0

        head_is_down = (head_pose in ('head_down', 'looking_away')) or \
                       self.head_pose_detector.is_head_down_like(person, head_pose)

        def _book_near(pt) -> bool:
            return SharedHelpers.point_near_book(pt, bbox_h, book_detections)

        def _wrist_ok(w, min_conf=0.4) -> bool:
            return w is not None and len(w) >= 3 and w[2] >= min_conf and w[0] != 0.0

        # ── Path 1: YOLO detected a phone object ─────────────────────────────
        # Real object evidence — never gated by the persistence counter.
        if phone_detections:
            for (px1, py1, px2, py2, conf) in phone_detections:
                if conf < 0.25:
                    if self.DEBUG:
                        print(f'[PHONE-DEBUG] {person.track_id}: YOLO phone conf={conf:.2f} < 0.25, skipped')
                    continue

                pc  = np.array([(px1 + px2) / 2.0, (py1 + py2) / 2.0])
                ph  = py2 - py1
                pw  = px2 - px1

                # Sanity-check phone bbox size relative to person
                size_ratio_max = max(pw, ph) / bbox_h
                size_ratio_min = min(pw, ph) / bbox_h
                if size_ratio_max > 0.7 or size_ratio_min < 0.03:
                    if self.DEBUG:
                        print(f'[PHONE-DEBUG] {person.track_id}: phone bbox size ratio '
                              f'max={size_ratio_max:.2f} min={size_ratio_min:.2f} out of [0.03,0.7], rejected')
                    continue

                # Phone must be inside or near the person bbox
                in_person_x = px1 < x2 + 20 and px2 > x1 - 20
                in_person_y = py1 < y2 + 20 and py2 > y1 - 20
                if not (in_person_x and in_person_y):
                    if self.DEBUG:
                        print(f'[PHONE-DEBUG] {person.track_id}: phone bbox ({px1},{py1},{px2},{py2}) '
                              f'outside person bbox ({x1},{y1},{x2},{y2}), rejected')
                    continue

                # Check proximity to any wrist or elbow (indices 7,8=elbow 9,10=wrist)
                matched = False
                if person.keypoints is not None and len(person.keypoints) > 10:
                    for idx in (9, 10, 7, 8):
                        if idx >= len(person.keypoints):
                            continue
                        w = person.keypoints[idx]
                        if not _wrist_ok(w, 0.4):
                            continue
                        dist = np.linalg.norm(w[:2] - pc) / bbox_h
                        if dist < 0.35:
                            self._clear_streak(person.track_id)
                            print(f'[PHONE] Person {person.track_id}: YOLO phone near joint {idx}, conf={conf:.2f}')
                            return True, float(conf)
                        elif self.DEBUG:
                            print(f'[PHONE-DEBUG] {person.track_id}: joint {idx} dist={dist:.2f} '
                                  f'(need <0.35) from phone centre, not matched')

                # If keypoints are unreliable but phone bbox overlaps person strongly, trust YOLO
                if conf >= 0.55:
                    overlap_x = max(0, min(px2, x2) - max(px1, x1))
                    overlap_y = max(0, min(py2, y2) - max(py1, y1))
                    overlap_area = overlap_x * overlap_y
                    phone_area   = max((px2 - px1) * (py2 - py1), 1)
                    overlap_ratio = overlap_area / phone_area
                    if overlap_ratio > 0.5:
                        self._clear_streak(person.track_id)
                        print(f'[PHONE] Person {person.track_id}: high-conf YOLO phone inside bbox, conf={conf:.2f}')
                        return True, float(conf)
                    elif self.DEBUG:
                        print(f'[PHONE-DEBUG] {person.track_id}: high-conf phone (conf={conf:.2f}) '
                              f'but overlap_ratio={overlap_ratio:.2f} < 0.5, rejected')
                elif self.DEBUG:
                    print(f'[PHONE-DEBUG] {person.track_id}: phone conf={conf:.2f} < 0.55, '
                          f'no wrist match, no fallback overlap check')

        # ── Path 2: Heuristic (no YOLO phone, use pose only) ─────────────────
        # No hard object evidence here — this is an inference from posture
        # alone, which is exactly what was producing false positives on
        # students writing in notebooks (head down + one wrist in lap zone
        # looks identical to phone use from pose alone). Gated by
        # _confirm_or_hold so a single ambiguous frame can't trigger it.
        if person.keypoints is None or person.keypoints.size == 0 or len(person.keypoints) <= 10:
            self._clear_streak(person.track_id)
            if self.DEBUG:
                print(f'[PHONE-DEBUG] {person.track_id}: no usable keypoints for heuristic path')
            return False, 0.0

        try:
            left_wrist  = person.keypoints[9]
            right_wrist = person.keypoints[10]
            lap_thresh  = y1 + bbox_h * self.PHONE_LAP_HEIGHT_FRACTION
            desk_thresh = y1 + bbox_h * self.WRITING_DESK_Y_MIN

            wrist_pts = [w[:2] for w in (left_wrist, right_wrist) if _wrist_ok(w, 0.5)]

            if not wrist_pts:
                self._clear_streak(person.track_id)
                if self.DEBUG:
                    print(f'[PHONE-DEBUG] {person.track_id}: no confident wrist keypoints, head_is_down={head_is_down}')
                return False, 0.0

            heuristic_hit = False

            # Single-hand in torso/lap zone with head down
            if head_is_down and len(wrist_pts) >= 1:
                for w in wrist_pts:
                    rel_y = (w[1] - y1) / bbox_h
                    rel_x_offset = abs(w[0] - (x1 + x2) / 2.0) / bbox_w
                    in_y_zone   = self.PHONE_SINGLE_HAND_Y_MIN <= rel_y <= self.PHONE_SINGLE_HAND_Y_MAX
                    centred     = rel_x_offset < 0.35
                    above_desk  = w[1] < desk_thresh
                    book_block  = _book_near(w)

                    # Wrist in mid-torso to lap zone, centred (not at desk edge)
                    if in_y_zone and centred and above_desk and not book_block:
                        heuristic_hit = True
                        break
                    elif self.DEBUG:
                        print(f'[PHONE-DEBUG] {person.track_id}: wrist rel_y={rel_y:.2f} '
                              f'(need {self.PHONE_SINGLE_HAND_Y_MIN}-{self.PHONE_SINGLE_HAND_Y_MAX}), '
                              f'rel_x_offset={rel_x_offset:.2f} (need <0.35), '
                              f'above_desk={above_desk} (wrist_y={w[1]:.0f} vs desk_thresh={desk_thresh:.0f}), '
                              f'book_near={book_block} — single-hand check failed')

            # Two cupped hands at lap level with head down
            if not heuristic_hit and head_is_down and len(wrist_pts) == 2:
                both_low = all(w[1] > lap_thresh for w in wrist_pts)
                spread   = np.linalg.norm(wrist_pts[0] - wrist_pts[1]) / bbox_h
                cupped_ok = spread < self.PHONE_CUPPED_SPREAD_MAX
                book_block2 = any(_book_near(w) for w in wrist_pts)
                if both_low and cupped_ok and not book_block2:
                    heuristic_hit = True
                elif self.DEBUG:
                    print(f'[PHONE-DEBUG] {person.track_id}: both_low={both_low}, '
                          f'spread={spread:.2f} (need <{self.PHONE_CUPPED_SPREAD_MAX}), '
                          f'book_near={book_block2} — cupped-hands check failed')

            if not head_is_down and self.DEBUG:
                print(f'[PHONE-DEBUG] {person.track_id}: head_is_down=False, heuristic path skipped entirely')

            if heuristic_hit:
                confirmed, conf = self._confirm_or_hold(person.track_id, 0.58)
                if confirmed:
                    print(f'[PHONE] Person {person.track_id}: heuristic confirmed '
                          f'after {self._heuristic_streak[person.track_id]} consecutive frames')
                elif self.DEBUG:
                    print(f'[PHONE-DEBUG] {person.track_id}: heuristic hit but only '
                          f'{self._heuristic_streak.get(person.track_id, 0)}/{self.HEURISTIC_CONFIRM_FRAMES} frames so far')
                return confirmed, conf
            else:
                self._clear_streak(person.track_id)

        except Exception as exc:
            print(f'[PHONE] Heuristic error: {exc}')

        return False, 0.0