import os
import time
import numpy as np
from typing import List, Tuple, Optional
from classroom_monitor.behavior_detection_core import TrackedPerson, SharedHelpers
from classroom_monitor.head_pose_detection import HeadPoseDetector


class PhoneDetector:
    """
    Two independent detection paths:

    Path 1 - YOLO actually detected a phone-shaped object near a hand.
             Real visual evidence, needs only light smoothing.

    Path 2 - No phone object seen; guess from hand position + head pose
             alone. Inherently ambiguous — a hand holding a notebook in
             the lap (no desk in frame) looks geometrically identical to a
             hand holding a phone in the lap. Persistence is required
             before this is trusted, and the book/notebook veto matters.

    Persistence on both paths uses LEAKY counters (decay by 1 on a miss,
    not reset to 0) rather than strict consecutive-frame counters. Pose
    estimation is noisy frame to frame, and a hard reset means one bad
    frame in the middle of five seconds of real phone use can wipe out
    the whole streak and the behavior never confirms.
    """

    PHONE_LAP_HEIGHT_FRACTION  = 0.65   # wrist must be genuinely low (lap), not mid-torso
    PHONE_CUPPED_SPREAD_MAX    = 0.15   # tight cupped grip only — open notebook has wider spread
    PHONE_SINGLE_HAND_Y_MIN    = 0.45   # wrist-relative-y above this = torso zone
    PHONE_SINGLE_HAND_Y_MAX    = 0.85
    WRITING_DESK_Y_MIN         = 0.70   # wrist above this threshold = desk level (when a desk exists)

    # Stillness ceiling for the single-hand torso-zone heuristic, specifically.
    # "Wrist in torso area, head down" alone is ambiguous with writing — the
    # is_writing check further down only catches BIG motion (threshold 300,
    # deliberately high so it doesn't block real phone-object evidence), so
    # calm, controlled handwriting strokes were sailing through this branch
    # and slowly confirming via the leaky counter over sustained writing.
    # This requires the wrist to be genuinely still, not just "not obviously
    # writing", before this specific branch can match at all. NOT YET
    # CALIBRATED against your camera — watch for false negatives (missed
    # real phone use) vs false positives (writing students) and adjust.
    PHONE_STILLNESS_MAX_VARIANCE = 60

    # Leaky-counter thresholds. At process_fps=10, ~6 net frames of
    # "credit" is roughly ~0.6s of real sustained heuristic match.
    HEURISTIC_THRESHOLD = 6
    YOLO_HIT_THRESHOLD = 2
    COUNTER_CAP = 20

    def __init__(self):
        self.head_pose_detector = HeadPoseDetector()
        self._heuristic_counters = {}
        self._yolo_counters = {}
        # Per-check diagnostic logging — off by default (production would
        # get flooded with per-frame prints), enable with PHONE_DEBUG=1 to
        # see exactly why a candidate did or didn't match on any given
        # frame, not just the confirmed-match cases.
        self.DEBUG = os.environ.get('PHONE_DEBUG', '0') == '1'

    def _debug(self, msg: str):
        if self.DEBUG:
            print(f'[{time.strftime("%H:%M:%S")}] {msg}')

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
        book_detections = book_detections or []
        x1, y1, x2, y2 = person.bbox
        bbox_h = y2 - y1
        bbox_w = x2 - x1
        tid = person.track_id
        if bbox_h <= 0 or bbox_w <= 0:
            return False, 0.0

        head_is_down = (head_pose in ('head_down', 'looking_away')) or \
                       self.head_pose_detector.is_head_down_like(person, head_pose)

        def _book_near(pt) -> bool:
            return SharedHelpers.point_near_book(pt, bbox_h, book_detections)

        def _wrist_ok(w, min_conf=0.4) -> bool:
            return w is not None and len(w) >= 3 and w[2] >= min_conf and w[0] != 0.0

        # ── Path 1: YOLO detected a phone object ─────────────────────────────
        # Real object evidence — runs regardless of wrist motion. An earlier
        # version of this file checked writing-motion suppression BEFORE
        # this path, which meant a real, object-detected phone could be
        # silently discarded just because the wrist was moving a bit.
        # Suppression below only ever gates Path 2 (the no-object
        # heuristic), which is the only path that actually needs it.
        yolo_hit_conf = None
        if phone_detections:
            self._debug(f'[PHONE] Person {tid}: {len(phone_detections)} raw phone det(s) this frame')
            for (px1, py1, px2, py2, conf) in phone_detections:
                if conf < 0.25:
                    self._debug(f'[PHONE] Person {tid}: det conf={conf:.2f} < 0.25 floor, skipped')
                    continue

                pc  = np.array([(px1 + px2) / 2.0, (py1 + py2) / 2.0])
                ph  = py2 - py1
                pw  = px2 - px1

                if max(pw, ph) / bbox_h > 0.7 or min(pw, ph) / bbox_h < 0.03:
                    self._debug(f'[PHONE] Person {tid}: det size ratio out of bounds '
                                f'(max={max(pw,ph)/bbox_h:.2f}, min={min(pw,ph)/bbox_h:.2f})')
                    continue

                in_person_x = px1 < x2 + 20 and px2 > x1 - 20
                in_person_y = py1 < y2 + 20 and py2 > y1 - 20
                if not (in_person_x and in_person_y):
                    self._debug(f'[PHONE] Person {tid}: det outside person bbox '
                                f'(det=({px1},{py1},{px2},{py2}) person=({x1},{y1},{x2},{y2}))')
                    continue

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

                if yolo_hit_conf is None:
                    self._debug(f'[PHONE] Person {tid}: det conf={conf:.2f} in bbox but no wrist/'
                                f'overlap match')

                if yolo_hit_conf is not None:
                    break

        yolo_confirmed = self._leaky_update(self._yolo_counters, tid, yolo_hit_conf is not None, self.YOLO_HIT_THRESHOLD)
        if yolo_hit_conf is not None:
            self._leaky_update(self._heuristic_counters, tid, False, self.HEURISTIC_THRESHOLD)
            if yolo_confirmed:
                print(f'[{time.strftime("%H:%M:%S")}] [PHONE] Person {tid}: YOLO phone confirmed, conf={yolo_hit_conf:.2f}')
                return True, yolo_hit_conf
            return False, 0.0

        # ── Writing-motion suppression — only reachable here, i.e. only
        # gates Path 2 below. Path 1 already ran and returned above on any
        # real object evidence.
        is_writing, _ = SharedHelpers.calculate_wrist_motion_variance(person)
        if is_writing:
            self._debug(f'[PHONE] Person {tid}: suppressed as writing motion (>300 variance)')
            self._leaky_update(self._heuristic_counters, tid, False, self.HEURISTIC_THRESHOLD)
            return False, 0.0

        # ── Path 2: Heuristic (no YOLO phone, use pose only) ─────────────────
        if person.keypoints is None or person.keypoints.size == 0 or len(person.keypoints) <= 10:
            self._debug(f'[PHONE] Person {tid}: no usable keypoints for heuristic path')
            self._leaky_update(self._heuristic_counters, tid, False, self.HEURISTIC_THRESHOLD)
            return False, 0.0

        try:
            left_wrist  = person.keypoints[9]
            right_wrist = person.keypoints[10]
            lap_thresh  = y1 + bbox_h * self.PHONE_LAP_HEIGHT_FRACTION
            desk_thresh = y1 + bbox_h * self.WRITING_DESK_Y_MIN

            wrist_pts = [w[:2] for w in (left_wrist, right_wrist) if _wrist_ok(w, 0.5)]
            both_wrists_for_veto = [w[:2] for w in (left_wrist, right_wrist) if _wrist_ok(w, 0.4)]

            if not wrist_pts:
                self._debug(f'[PHONE] Person {tid}: no confident wrist keypoints '
                            f'(head_is_down={head_is_down})')
                self._leaky_update(self._heuristic_counters, tid, False, self.HEURISTIC_THRESHOLD)
                return False, 0.0

            heuristic_matched = False
            match_conf = 0.0

            # Single-hand torso-zone branch additionally requires genuine
            # stillness (see PHONE_STILLNESS_MAX_VARIANCE) — this is the
            # branch that was matching sustained, calm handwriting.
            has_variance, variance = SharedHelpers.wrist_motion_variance_raw(person)
            wrist_is_still = has_variance and variance <= self.PHONE_STILLNESS_MAX_VARIANCE
            self._debug(f'[PHONE] Person {tid}: stillness check — has_variance={has_variance} '
                        f'variance={variance:.1f} (need <= {self.PHONE_STILLNESS_MAX_VARIANCE}) '
                        f'still={wrist_is_still}')

            if head_is_down and len(wrist_pts) >= 1 and wrist_is_still:
                for w in wrist_pts:
                    rel_y = (w[1] - y1) / bbox_h
                    rel_x_offset = abs(w[0] - (x1 + x2) / 2.0) / bbox_w
                    above_desk = w[1] < desk_thresh
                    book_block = any(_book_near(p) for p in both_wrists_for_veto)

                    if (self.PHONE_SINGLE_HAND_Y_MIN <= rel_y <= self.PHONE_SINGLE_HAND_Y_MAX
                            and rel_x_offset < 0.35 and above_desk and not book_block):
                        heuristic_matched = True
                        match_conf = 0.60
                        break
                    else:
                        self._debug(f'[PHONE] Person {tid}: single-hand check failed — '
                                    f'rel_y={rel_y:.2f} (need {self.PHONE_SINGLE_HAND_Y_MIN}-'
                                    f'{self.PHONE_SINGLE_HAND_Y_MAX}) rel_x_offset={rel_x_offset:.2f} '
                                    f'(need <0.35) above_desk={above_desk} book_block={book_block}')
            elif head_is_down and len(wrist_pts) >= 1 and not wrist_is_still:
                self._debug(f'[PHONE] Person {tid}: single-hand zone check skipped — wrist not still')

            if not heuristic_matched and head_is_down and len(wrist_pts) == 2:
                both_low = all(w[1] > lap_thresh for w in wrist_pts)
                spread   = np.linalg.norm(wrist_pts[0] - wrist_pts[1]) / bbox_h
                book_block2 = any(_book_near(p) for p in both_wrists_for_veto)
                if both_low and spread < self.PHONE_CUPPED_SPREAD_MAX and not book_block2:
                    heuristic_matched = True
                    match_conf = 0.55
                else:
                    self._debug(f'[PHONE] Person {tid}: cupped-hands check failed — '
                                f'both_low={both_low} spread={spread:.2f} '
                                f'(need <{self.PHONE_CUPPED_SPREAD_MAX}) book_block={book_block2}')

            confirmed = self._leaky_update(self._heuristic_counters, tid, heuristic_matched, self.HEURISTIC_THRESHOLD)
            if heuristic_matched:
                self._debug(f'[PHONE] Person {tid}: heuristic matched this frame, '
                            f'counter={self._heuristic_counters.get(tid, 0)}/{self.HEURISTIC_THRESHOLD}')
            if heuristic_matched and confirmed:
                print(f'[{time.strftime("%H:%M:%S")}] [PHONE] Person {tid}: sustained pose heuristic match, conf={match_conf:.2f}')
                return True, match_conf
            return False, 0.0

        except Exception as exc:
            print(f'[PHONE] Heuristic error: {exc}')

        return False, 0.0