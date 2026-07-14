import os
import numpy as np
from typing import List, Tuple, Optional
from classroom_monitor.behavior_detection_core import TrackedPerson, SharedHelpers
from classroom_monitor.head_pose_detection import HeadPoseDetector


class PhoneDetector:
    PHONE_LAP_HEIGHT_FRACTION   = 0.55   # was 0.65 — catch higher lap positions
    PHONE_CUPPED_SPREAD_MAX     = 0.22   # was 0.15 — slightly wider cupped grip
    PHONE_SINGLE_HAND_Y_MIN     = 0.45   # wrist-relative-y above this = torso zone
    # Effective upper bound for the single-hand zone is enforced via
    # `w[1] < desk_thresh` (WRITING_DESK_Y_MIN = 0.70) below, not a
    # separate constant — a rel_y upper bound at or above 0.70 would
    # never be reachable since desk_thresh already excludes it first.
    WRITING_DESK_Y_MIN          = 0.70   # wrist above this threshold = desk level

    # Minimum COCO "cell phone" class confidence to trust a wrist/elbow
    # match in Path 1. Raised from an inline 0.25 — that floor was too
    # permissive and let calculators, remotes, and glasses cases through.
    PHONE_WRIST_MATCH_MIN_CONF  = 0.35

    # Ceiling on wrist-motion variance for the single-hand posture heuristic
    # (Path 2 below) to fire. Writing involves continuous small wrist
    # strokes; holding a phone keeps the wrist comparatively still. Without
    # this, "head down + one hand near torso" matches writing-in-a-notebook
    # just as well as it matches phone use — that ambiguity was the direct
    # cause of students writing getting tagged "Using Phone." NOT YET
    # CALIBRATED against your real camera footage — set PHONE_DEBUG=1,
    # watch the printed variance values for a few known-writing vs
    # known-phone-holding students, and adjust this if it's rejecting or
    # accepting too much.
    PHONE_STILLNESS_MAX_VARIANCE = 60

    # ── Path 3 constants: phone held up near the face/ear ────────────────
    # Path 2 above only recognizes "phone in lap, head down" — it has no
    # coverage for a phone held up in front of the face or to the ear,
    # where the head stays level or even tilts back rather than down, so
    # it never reaches Path 2's `head_is_down` gate at all. This path is
    # deliberately independent of head pose and instead keys off wrist
    # position relative to the face plus a raised elbow, which together
    # are specific enough not to be confused with hand-raising (wrist
    # goes well above head, straight arm) or writing (wrist stays low,
    # near desk level). NOT YET CALIBRATED against your real footage —
    # set PHONE_DEBUG=1 and check these against known phone-to-face clips.
    FACE_PHONE_X_MAX_FRACTION   = 0.30   # max |wrist_x - nose_x| / bbox_w
    # Vertical band is asymmetric, not a simple +/- around the nose: a
    # phone held to the face sits at forehead level or lower, never
    # raised high above the head. Symmetric bounds let a straight-arm
    # hand-raise (wrist far above the nose) false-positive as phone-use,
    # since both have "elbow at/above shoulder" — this asymmetry is what
    # actually tells the two postures apart.
    FACE_PHONE_Y_ABOVE_MAX_FRACTION = 0.12   # wrist at most this far ABOVE nose
    # Widened from 0.30 — production logs (2026-07-14, C Lab camera) showed
    # a visibly real phone-holder sitting at y_signed=0.30-0.33, i.e. just
    # outside the old band, consistent with holding the phone at chest/
    # lower-face height rather than right up against the nose.
    FACE_PHONE_Y_BELOW_MAX_FRACTION = 0.42   # wrist at most this far BELOW nose
    # Confirms the forearm is actually bent upward toward the face, rather
    # than the wrist merely landing near the face by keypoint noise. NOTE:
    # this is deliberately a wrist-vs-ELBOW check, not wrist-vs-shoulder or
    # elbow-vs-shoulder — when someone holds a phone up while seated, the
    # upper arm typically stays down near the ribs and only the forearm
    # swings up, so the elbow itself rarely reaches shoulder height. An
    # earlier version of this check required elbow[1] <= shoulder[1], which
    # production logs showed was false on literally every sample (including
    # frames with a phone visibly held to the face) — it was checking for a
    # raised-upper-arm pose that this posture doesn't produce.
    # Margin also narrowed from 0.06 — same production evidence showed
    # every candidate (including the real phone-holder) failing this gate,
    # consistent with a lower/less-sharply-bent holding posture than the
    # original margin assumed.
    FACE_PHONE_FOREARM_RAISE_MARGIN = 0.02   # wrist must be this far ABOVE elbow, as a fraction of bbox_h

    def __init__(self):
        self.head_pose_detector = HeadPoseDetector()
        # Per-frame debug prints are opt-in via env var so production
        # doesn't get flooded with per-frame logging.
        self.DEBUG = os.environ.get('PHONE_DEBUG', '0') == '1'

    def _debug(self, msg: str):
        if self.DEBUG:
            print(msg)

    @staticmethod
    def _wrist_ok(w, min_conf=0.4) -> bool:
        return w is not None and len(w) >= 3 and w[2] >= min_conf and w[0] != 0.0


    def detect_phone_usage(self, person: TrackedPerson, phone_detections: List[Tuple],
                           head_pose: str, book_detections: Optional[List[Tuple]] = None) -> Tuple[bool, float]:
        book_detections = book_detections or []
        x1, y1, x2, y2 = person.bbox
        bbox_h = y2 - y1
        bbox_w = x2 - x1
        if bbox_h <= 0 or bbox_w <= 0:
            return False, 0.0

        head_is_down = (head_pose in ('head_down', 'looking_away')) or \
                       self.head_pose_detector.is_head_down_like(person, head_pose)

        def _book_near(pt) -> bool:
            return SharedHelpers.point_near_book(pt, bbox_h, book_detections)

        # ── Path 1: YOLO detected a phone object ─────────────────────────────
        # Real object evidence is never discarded by the writing-motion
        # heuristic — that suppression only applies to Path 2 below, where
        # there's no object detection to fall back on.
        if phone_detections:
            self._debug(f'[PHONE] Person {person.track_id}: {len(phone_detections)} raw phone '
                        f'det(s) this frame: {[round(d[4], 3) for d in phone_detections]} '
                        f'(match floor={self.PHONE_WRIST_MATCH_MIN_CONF})')
            for (px1, py1, px2, py2, conf) in phone_detections:
                if conf < self.PHONE_WRIST_MATCH_MIN_CONF:
                    self._debug(f'[PHONE] Person {person.track_id}: rejected det conf={conf:.3f} '
                                f'< floor {self.PHONE_WRIST_MATCH_MIN_CONF} — this is the case to '
                                f'check if a real phone is being filtered out at distance')
                    continue

                pc  = np.array([(px1 + px2) / 2.0, (py1 + py2) / 2.0])
                ph  = py2 - py1
                pw  = px2 - px1

                # Sanity-check phone bbox size relative to person
                if max(pw, ph) / bbox_h > 0.7 or min(pw, ph) / bbox_h < 0.03:
                    self._debug(f'[PHONE] Person {person.track_id}: rejected det conf={conf:.3f} '
                                f'on size ratio (pw={pw:.0f} ph={ph:.0f} bbox_h={bbox_h:.0f})')
                    continue

                # Phone must be inside or near the person bbox
                in_person_x = px1 < x2 + 20 and px2 > x1 - 20
                in_person_y = py1 < y2 + 20 and py2 > y1 - 20
                if not (in_person_x and in_person_y):
                    self._debug(f'[PHONE] Person {person.track_id}: rejected det conf={conf:.3f} '
                                f'— outside person bbox')
                    continue

                # Check proximity to any wrist or elbow (indices 7,8=elbow 9,10=wrist)
                if person.keypoints is not None and len(person.keypoints) > 10:
                    for idx in (9, 10, 7, 8):
                        if idx >= len(person.keypoints):
                            continue
                        w = person.keypoints[idx]
                        if self._wrist_ok(w, 0.4) and np.linalg.norm(w[:2] - pc) / bbox_h < 0.35:
                            self._debug(f'[PHONE] Person {person.track_id}: YOLO phone near joint {idx}, conf={conf:.2f}')
                            return True, float(conf)

                # If keypoints are unreliable but phone bbox overlaps person strongly, trust YOLO
                if conf >= 0.55:
                    overlap_x = max(0, min(px2, x2) - max(px1, x1))
                    overlap_y = max(0, min(py2, y2) - max(py1, y1))
                    overlap_area = overlap_x * overlap_y
                    phone_area   = max((px2 - px1) * (py2 - py1), 1)
                    if overlap_area / phone_area > 0.5:
                        self._debug(f'[PHONE] Person {person.track_id}: high-conf YOLO phone inside bbox, conf={conf:.2f}')
                        return True, float(conf)

                self._debug(f'[PHONE] Person {person.track_id}: det conf={conf:.3f} passed size/position '
                            f'but matched no wrist/elbow and did not clear the high-conf overlap fallback')

        # ── Path 2 / Path 3: pose-only heuristics (no YOLO phone match) ──────
        if person.keypoints is None or person.keypoints.size == 0 or len(person.keypoints) <= 10:
            return False, 0.0

        # ── Path 3: phone held up near the face/ear ───────────────────────
        # Runs before the writing-motion suppression below on purpose: that
        # suppression exists to disambiguate Path 2's lap-level posture from
        # writing, which is irrelevant here — nobody writes with a hand up
        # at their nose. Gating this on `is_writing` would just make Path 3
        # miss real phone-to-face use when the *other* hand happens to be
        # writing at the time. Deliberately does NOT gate on head_is_down
        # either — see the FACE_PHONE_* constants comment above. Checks
        # each arm (wrist+elbow) independently since only one hand is
        # normally raised to the face.
        try:
            nose = person.keypoints[0] if len(person.keypoints) > 0 else None
            if self._wrist_ok(nose, 0.4):
                nose_pt = nose[:2]
                for wrist_idx, elbow_idx in ((9, 7), (10, 8)):
                    if elbow_idx >= len(person.keypoints):
                        continue
                    wrist = person.keypoints[wrist_idx]
                    elbow = person.keypoints[elbow_idx]
                    if not (self._wrist_ok(wrist, 0.4) and self._wrist_ok(elbow, 0.4)):
                        continue

                    # Signed, not absolute: negative = wrist above nose,
                    # positive = wrist below nose. The two directions get
                    # different tolerances (see constants comment above).
                    y_signed = (wrist[1] - nose_pt[1]) / bbox_h
                    x_frac   = abs(wrist[0] - nose_pt[0]) / bbox_w
                    forearm_gap = (elbow[1] - wrist[1]) / bbox_h  # positive = wrist above elbow
                    forearm_raised = forearm_gap >= self.FACE_PHONE_FOREARM_RAISE_MARGIN
                    y_in_band = (-self.FACE_PHONE_Y_ABOVE_MAX_FRACTION <= y_signed
                                 <= self.FACE_PHONE_Y_BELOW_MAX_FRACTION)

                    self._debug(f'[PHONE] Person {person.track_id}: face-phone check joint {wrist_idx} — '
                                f'y_signed={y_signed:.2f} (band [-{self.FACE_PHONE_Y_ABOVE_MAX_FRACTION}, '
                                f'{self.FACE_PHONE_Y_BELOW_MAX_FRACTION}]) '
                                f'x_frac={x_frac:.2f} (need <= {self.FACE_PHONE_X_MAX_FRACTION}) '
                                f'forearm_gap={forearm_gap:.3f} (need >= {self.FACE_PHONE_FOREARM_RAISE_MARGIN}) '
                                f'forearm_raised={forearm_raised}')

                    if (y_in_band
                            and x_frac <= self.FACE_PHONE_X_MAX_FRACTION
                            and forearm_raised
                            and not _book_near(wrist[:2])):
                        self._debug(f'[PHONE] Person {person.track_id}: phone-to-face heuristic fired '
                                    f'on joint {wrist_idx}')
                        return True, 0.58
        except Exception as exc:
            self._debug(f'[PHONE] Face-phone heuristic error: {exc}')

        # ── Path 2: Heuristic (lap-level posture only) ────────────────────
        # Suppress this posture-only path when there's clear, high-variance
        # writing motion — but only here. Path 1 above already returned on
        # any genuine object match, and Path 3 above is intentionally not
        # gated on this, so this never throws away real evidence.
        is_writing, _ = SharedHelpers.calculate_wrist_motion_variance(person)
        if is_writing:
            return False, 0.0

        try:
            left_wrist  = person.keypoints[9]
            right_wrist = person.keypoints[10]
            lap_thresh  = y1 + bbox_h * self.PHONE_LAP_HEIGHT_FRACTION
            desk_thresh = y1 + bbox_h * self.WRITING_DESK_Y_MIN

            wrist_pts = [w[:2] for w in (left_wrist, right_wrist) if self._wrist_ok(w, 0.5)]

            if not wrist_pts:
                return False, 0.0

            # Single-hand in torso/lap zone with head down. This posture
            # alone is ambiguous with writing (see PHONE_STILLNESS_MAX_VARIANCE
            # comment above), so it additionally requires the wrist to be
            # relatively still — enough keypoint history must exist to make
            # that judgement, and if it doesn't exist yet, we do NOT fall
            # back to firing anyway, since that would just re-introduce the
            # original ambiguity.
            has_variance, variance = SharedHelpers.wrist_motion_available_and_variance(person)
            wrist_is_still = has_variance and variance <= self.PHONE_STILLNESS_MAX_VARIANCE
            self._debug(f'[PHONE] Person {person.track_id}: stillness check — '
                        f'has_variance={has_variance} variance={variance:.1f} '
                        f'(need <= {self.PHONE_STILLNESS_MAX_VARIANCE}) still={wrist_is_still}')

            if head_is_down and len(wrist_pts) >= 1 and wrist_is_still:
                for w in wrist_pts:
                    rel_y = (w[1] - y1) / bbox_h
                    rel_x_offset = abs(w[0] - (x1 + x2) / 2.0) / bbox_w

                    # Wrist in mid-torso to lap zone, centred (not at desk edge).
                    # Upper bound of the zone is enforced by w[1] < desk_thresh
                    # (rel_y < WRITING_DESK_Y_MIN), not a separate constant.
                    if (rel_y >= self.PHONE_SINGLE_HAND_Y_MIN
                            and rel_x_offset < 0.35
                            and w[1] < desk_thresh   # not at desk level
                            and not _book_near(w)):
                        self._debug(f'[PHONE] Person {person.track_id}: single-hand torso heuristic '
                                    f'(still, variance={variance:.1f})')
                        return True, 0.60

            # Two cupped hands at lap level with head down
            if head_is_down and len(wrist_pts) == 2:
                both_low = all(w[1] > lap_thresh for w in wrist_pts)
                spread   = np.linalg.norm(wrist_pts[0] - wrist_pts[1]) / bbox_h
                if both_low and spread < self.PHONE_CUPPED_SPREAD_MAX:
                    if not any(_book_near(w) for w in wrist_pts):
                        self._debug(f'[PHONE] Person {person.track_id}: cupped hands at lap')
                        return True, 0.55

        except Exception as exc:
            self._debug(f'[PHONE] Heuristic error: {exc}')

        return False, 0.0