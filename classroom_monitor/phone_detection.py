import numpy as np
from typing import List, Tuple, Optional
from classroom_monitor.behavior_detection_core import TrackedPerson, SharedHelpers
from classroom_monitor.head_pose_detection import HeadPoseDetector


class PhoneDetector:
    """
    Three independent detection paths:

    Path 1 - YOLO actually detected a phone-shaped object near a hand.
             Real visual evidence, needs only light smoothing.

    Path 2 - No phone object seen; guess from hand position + head pose
             alone. Inherently ambiguous — a hand holding a notebook in
             the lap (no desk in frame) looks geometrically identical to a
             hand holding a phone in the lap. Persistence is required
             before this is trusted, and the book/notebook veto matters.

    Path 3 - No phone object seen; phone held up near the face/ear. Head
             pose is often still "focused"-looking here (chin isn't down),
             so this can't be gated on head_is_down like Path 2 — it's
             keyed on wrist-near-nose distance plus a bent-forearm check
             (wrist above elbow) to distinguish "hand at face" from "hand
             at chest/shoulder".

    Persistence on all heuristic paths uses LEAKY counters (decay by 1 on a miss,
    not reset to 0) rather than strict consecutive-frame counters. Pose
    estimation is noisy frame to frame, and a hard reset means one bad
    frame in the middle of five seconds of real phone use can wipe out
    the whole streak and the behavior never confirms.
    """

    PHONE_LAP_HEIGHT_FRACTION  = 0.65   # wrist must be genuinely low (lap), not mid-torso
    PHONE_CUPPED_SPREAD_MAX    = 0.20   # tight-ish cupped grip — open notebook has notably wider spread
    PHONE_SINGLE_HAND_Y_MIN    = 0.45   # wrist-relative-y above this = torso zone
    PHONE_SINGLE_HAND_Y_MAX    = 0.85
    PHONE_NEAR_FACE_DIST_FRAC  = 0.35   # wrist-to-nose distance / bbox_h — "hand at face" zone
    PHONE_FOREARM_BEND_FRAC    = 0.03   # wrist must sit at least this much above the elbow (/bbox_h)

    # Leaky-counter thresholds. At process_fps=10, ~6 net frames of
    # "credit" is roughly ~0.6s of real sustained heuristic match.
    HEURISTIC_THRESHOLD = 6
    YOLO_HIT_THRESHOLD = 1
    COUNTER_CAP = 20

    def __init__(self):
        self.head_pose_detector = HeadPoseDetector()
        self._heuristic_counters = {}
        self._yolo_counters = {}

    @staticmethod
    def _face_anchor(kp) -> Optional[np.ndarray]:
        """
        Best-available anchor point for the "phone near face" distance
        check. Prefers the nose, but falls back to eye-center, then a
        visible ear, if the nose keypoint isn't trustworthy.
        """
        def ok(pt, min_conf):
            return pt is not None and len(pt) >= 3 and pt[2] >= min_conf and pt[0] != 0.0

        if kp is None:
            return None

        nose = kp[0] if len(kp) > 0 else None
        if ok(nose, 0.3):
            return nose[:2]

        left_eye  = kp[1] if len(kp) > 1 else None
        right_eye = kp[2] if len(kp) > 2 else None
        eyes = [e[:2] for e in (left_eye, right_eye) if ok(e, 0.3)]
        if eyes:
            return np.mean(eyes, axis=0)

        left_ear  = kp[3] if len(kp) > 3 else None
        right_ear = kp[4] if len(kp) > 4 else None
        for ear in (left_ear, right_ear):
            if ok(ear, 0.3):
                return ear[:2]

        return None

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

        # Suppress phone detection when there is clear, high-variance writing motion
        is_writing, _ = SharedHelpers.calculate_wrist_motion_variance(person)
        if is_writing:
            self._leaky_update(self._heuristic_counters, tid, False, self.HEURISTIC_THRESHOLD)
            self._leaky_update(self._yolo_counters, tid, False, self.YOLO_HIT_THRESHOLD)
            return False, 0.0

        head_is_down = (head_pose in ('head_down', 'looking_away')) or \
                       self.head_pose_detector.is_head_down_like(person, head_pose)

        def _book_near(pt) -> bool:
            return SharedHelpers.point_near_book(pt, bbox_h, book_detections)

        def _wrist_ok(w, min_conf=0.25) -> bool:
            return w is not None and len(w) >= 3 and w[2] >= min_conf and w[0] != 0.0

        # ── Path 1: YOLO detected a phone object ─────────────────────────────
        yolo_hit_conf = None
        if phone_detections:
            for (px1, py1, px2, py2, conf) in phone_detections:
                if conf < 0.15:
                    print(f'[PHONE-DEBUG] tid={tid} rejected: below confidence floor '
                          f'conf={conf:.2f} < 0.15')
                    continue

                pc  = np.array([(px1 + px2) / 2.0, (py1 + py2) / 2.0])
                ph  = py2 - py1
                pw  = px2 - px1

                if max(pw, ph) / bbox_h > 0.7 or min(pw, ph) / bbox_h < 0.03:
                    print(f'[PHONE-DEBUG] tid={tid} rejected: size ratio '
                          f'max={max(pw, ph)/bbox_h:.2f} min={min(pw, ph)/bbox_h:.2f}')
                    continue

                pad_x = bbox_w * 0.20
                pad_y = bbox_h * 0.20
                in_person_x = px1 < x2 + pad_x and px2 > x1 - pad_x
                in_person_y = py1 < y2 + pad_y and py2 > y1 - pad_y
                if not (in_person_x and in_person_y):
                    print(f'[PHONE-DEBUG] tid={tid} rejected: outside padded bbox '
                          f'person=({x1},{y1},{x2},{y2}) phone=({px1},{py1},{px2},{py2})')
                    continue

                # Guard against false positives from hand resting under chin/face
                face_anchor = self._face_anchor(person.keypoints)
                if face_anchor is not None:
                    dist_to_face = np.linalg.norm(pc - face_anchor) / max(bbox_h, 1.0)
                    # If phone box is at face/chin area, require higher confidence (conf >= 0.45)
                    # to distinguish a real phone call from a hand resting on chin/cheek.
                    if dist_to_face < 0.25 and conf < 0.45:
                        print(f'[PHONE-DEBUG] tid={tid} rejected: hand at face/chin false positive conf={conf:.2f} < 0.45')
                        continue

                matched_kp = False
                if person.keypoints is not None and len(person.keypoints) > 10:
                    for idx in (9, 10, 7, 8):
                        if idx >= len(person.keypoints):
                            continue
                        w = person.keypoints[idx]
                        if _wrist_ok(w, 0.20):
                            dist_px = np.linalg.norm(w[:2] - pc)
                            dist_norm = dist_px / max(bbox_h, bbox_w, 1.0)
                            if dist_norm < 0.65 or dist_px < 140 or (x1 <= pc[0] <= x2 and y1 <= pc[1] <= y2):
                                yolo_hit_conf = float(conf)
                                matched_kp = True
                                break

                # Overlap match fallback: phone center must lie within person bounding box
                if yolo_hit_conf is None and conf >= 0.22:
                    in_person_inner = (x1 - 10) <= pc[0] <= (x2 + 10) and (y1 - 10) <= pc[1] <= (y2 + 10)
                    if in_person_inner:
                        overlap_x = max(0, min(px2, x2) - max(px1, x1))
                        overlap_y = max(0, min(py2, y2) - max(py1, y1))
                        overlap_area = overlap_x * overlap_y
                        phone_area   = max((px2 - px1) * (py2 - py1), 1)
                        if overlap_area / phone_area > 0.4:
                            yolo_hit_conf = float(conf)

                if yolo_hit_conf is not None:
                    break

        # YOLO hits >= 0.28 confirm on 1 hit
        yolo_confirmed = self._leaky_update(self._yolo_counters, tid, yolo_hit_conf is not None, 1)

        if yolo_hit_conf is not None:
            print(f'[PHONE-DEBUG] tid={tid} yolo hit conf={yolo_hit_conf:.2f} '
                  f'counter={self._yolo_counters.get(tid, 0)}/1')
            if yolo_confirmed:
                print(f'[PHONE] Person {tid}: YOLO phone confirmed, conf={yolo_hit_conf:.2f}')
                return True, yolo_hit_conf

        # ── Path 2: Heuristic (no YOLO phone, use pose only) ─────────────────
        if person.keypoints is None or person.keypoints.size == 0 or len(person.keypoints) <= 10:
            self._leaky_update(self._heuristic_counters, tid, False, self.HEURISTIC_THRESHOLD)
            return False, 0.0

        try:
            left_wrist  = person.keypoints[9]
            right_wrist = person.keypoints[10]
            lap_thresh  = y1 + bbox_h * self.PHONE_LAP_HEIGHT_FRACTION

            wrist_pts = [w[:2] for w in (left_wrist, right_wrist) if _wrist_ok(w, 0.25)]
            both_wrists_for_veto = [w[:2] for w in (left_wrist, right_wrist) if _wrist_ok(w, 0.20)]

            heuristic_matched = False
            match_conf = 0.0

            # ── Path 3: phone held near face/ear ─────────────────────────
            face_anchor = self._face_anchor(person.keypoints)
            if face_anchor is not None:
                for w_idx, e_idx in ((9, 7), (10, 8)):
                    w = person.keypoints[w_idx] if w_idx < len(person.keypoints) else None
                    e = person.keypoints[e_idx] if e_idx < len(person.keypoints) else None
                    if not (_wrist_ok(w, 0.25) and e is not None and len(e) >= 3 and e[2] >= 0.25):
                        continue
                    dist_to_face = np.linalg.norm(w[:2] - face_anchor) / bbox_h
                    forearm_bent = w[1] <= e[1] - bbox_h * self.PHONE_FOREARM_BEND_FRAC
                    if (dist_to_face < self.PHONE_NEAR_FACE_DIST_FRAC and forearm_bent
                            and not _book_near(w[:2])):
                        heuristic_matched = True
                        match_conf = 0.62
                        break

            if not heuristic_matched and not wrist_pts:
                self._leaky_update(self._heuristic_counters, tid, False, self.HEURISTIC_THRESHOLD)
                return False, 0.0

            head_down_or_tilted = head_is_down or (head_pose in ('focused', 'not_visible'))

            if not heuristic_matched and head_down_or_tilted and len(wrist_pts) >= 1:
                for w in wrist_pts:
                    rel_y = (w[1] - y1) / bbox_h
                    rel_x_offset = abs(w[0] - (x1 + x2) / 2.0) / bbox_w

                    if (0.35 <= rel_y <= 0.88
                            and rel_x_offset < 0.38
                            and not any(_book_near(p) for p in both_wrists_for_veto)):
                        heuristic_matched = True
                        match_conf = 0.60
                        break

            if not heuristic_matched and head_down_or_tilted and len(wrist_pts) == 2:
                both_low = all(w[1] > lap_thresh for w in wrist_pts)
                spread   = np.linalg.norm(wrist_pts[0] - wrist_pts[1]) / bbox_h
                if both_low and spread < self.PHONE_CUPPED_SPREAD_MAX:
                    if not any(_book_near(p) for p in both_wrists_for_veto):
                        heuristic_matched = True
                        match_conf = 0.55

            confirmed = self._leaky_update(self._heuristic_counters, tid, heuristic_matched, self.HEURISTIC_THRESHOLD)
            if heuristic_matched and confirmed:
                print(f'[PHONE] Person {tid}: sustained pose heuristic match, conf={match_conf:.2f}')
                return True, match_conf
            return False, 0.0

        except Exception as exc:
            print(f'[PHONE] Heuristic error: {exc}')

        return False, 0.0