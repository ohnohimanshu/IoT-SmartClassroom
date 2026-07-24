import cv2
import json
import time
import base64
import numpy as np
import threading
import os
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple

# Import modular detectors
from classroom_monitor.constants import (
    COLOR_MAP, LABEL_MAP, ALERT_POSES, DISTRACTED_POSES, FACE_MATCH_TOLERANCE,
)
from classroom_monitor.behavior_detection_core import (
    TrackedPerson, DetectionResult, TemporalBehaviorEngine, SharedHelpers
)
from classroom_monitor.head_pose_detection import HeadPoseDetector
from classroom_monitor.phone_detection import PhoneDetector
from classroom_monitor.hand_raise_detection import HandRaiseDetector
from classroom_monitor.food_detection import FoodDetector


def _http_verify_ssl() -> bool:
    return os.environ.get('HTTP_VERIFY_SSL', 'true').strip().lower() not in (
        'false', '0', 'no',
    )


def _load_env_file():
    for candidate in [
        os.path.join(os.path.dirname(__file__), '..', '.env'),
        os.path.join(os.path.dirname(__file__), '..', '..', '.env'),
        '.env',
    ]:
        path = os.path.abspath(candidate)
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, _, v = line.partition('=')
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            print(f'[ENV] Loaded {path}')
            return
    print('[ENV] No .env file found')


_load_env_file()

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')


# ── Shared YOLO models (one load per process) ────────────────────────────────
class _SharedYOLOModels:
    _lock = threading.Lock()
    _pose_model = None
    _object_model = None
    _phone_model = None
    _phone_model_attempted = False

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._pose_model is None:
                try:
                    from ultralytics import YOLO
                    cls._pose_model = YOLO('yolo11s-pose.pt')
                    cls._object_model = YOLO('yolo11s.pt')
                    print('[OK] Shared YOLO pose + object models loaded')
                except Exception as e:
                    print(f'[WARN] Shared YOLO model load failed: {e}')
            if cls._phone_model is None and not cls._phone_model_attempted:
                # Only attempt this once per process — previously this
                # retried (and re-printed a warning) on every single
                # detector construction, which is why the log was full of
                # repeated "Custom phone model load failed" lines.
                cls._phone_model_attempted = True
                try:
                    from ultralytics import YOLO
                    # CLASSROOM_PHONE_MODEL_PATH must actually be read here
                    # — a bare 'classroom_phone_yolo.pt' filename (no env
                    # var lookup) was silently ignoring any path set in
                    # .env/docker-compose, which is why that setting had no
                    # effect no matter what it was set to.
                    path = os.environ.get(
                        'CLASSROOM_PHONE_MODEL_PATH',
                        os.path.join(os.path.dirname(__file__), 'model_weights', 'classroom_phone_yolo.pt'),
                    )
                    if os.path.exists(path):
                        cls._phone_model = YOLO(path)
                        print(f'[OK] Custom classroom phone model loaded from {path}')
                    else:
                        print(f'[WARN] Custom phone model not found at {path} — '
                              f'set CLASSROOM_PHONE_MODEL_PATH or place the file there')
                except Exception as e:
                    print(f'[WARN] Custom phone model load failed: {e}')
            return cls._pose_model, cls._object_model, cls._phone_model


# ── Production Stream Processor ──────────────────────────────────────────────
class ProductionStreamProcessor:
    _PHONE_CLS = {67}
    _FOOD_CLS  = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55}
    _BOOK_CLS  = {73}

    def __init__(self, process_fps: int = 10, buffer_size: int = 5):
        self.process_fps      = process_fps
        self.frame_interval   = 1.0 / process_fps
        self.frame_buffer: deque = deque(maxlen=buffer_size)
        self.result_buffer: deque = deque(maxlen=2)
        self.yolo_model       = None
        self.object_model     = None
        self.phone_model      = None
        self.running          = False
        self.lock             = threading.Lock()
        self.stop_event       = threading.Event()
        self.phone_detections: List[Tuple] = []
        self.food_detections:  List[Tuple] = []
        self.person_tracks: List[Tuple] = []
        self.behavior_engine  = TemporalBehaviorEngine()
        self.fight_detector   = None
        # Static-false-positive suppression (see _update_static_fp_map /
        # _filter_static_fp below) — tracks phone-shaped detections that
        # keep appearing at the same screen location with no person ever
        # nearby (a poster, light switch, reflective surface, etc).
        self._static_fp_counters: Dict[Tuple[int, int], int] = {}
        self._STATIC_FP_CELL = 40        # px grid cell size
        self._STATIC_FP_THRESHOLD = 40   # ~4s of net unmatched hits at process_fps=10
        self._STATIC_FP_CAP = 80
        track_buffer_frames = 60          # must match bytetrack_classroom.yaml's track_buffer
        self.behavior_engine.cleanup_threshold = (track_buffer_frames / process_fps) + 1.0

        # Initialize modular detectors
        self.head_pose_detector = HeadPoseDetector()
        self.phone_detector     = PhoneDetector()
        self.hand_raise_detector = HandRaiseDetector()
        self.food_detector      = FoodDetector()

        self._ensure_models()
        self._init_fight_detector()

    def _ensure_models(self):
        self.yolo_model, self.object_model, self.phone_model = _SharedYOLOModels.get()

    def _init_fight_detector(self):
        try:
            from classroom_monitor.fight_detection_3dcnn import FightDetector3DCNN
            self.fight_detector = FightDetector3DCNN()
            print('[OK] Fight detector initialized')
        except Exception as e:
            print(f'[WARN] Fight detector initialization failed: {e}')
            self.fight_detector = None

    def _load_models(self):
        self._ensure_models()

    def _capture_frames(self, camera_url: str):
        cap, reconnect_delay = None, 1.0
        while not self.stop_event.is_set():
            try:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(camera_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    if cap.isOpened():
                        print('[OK] Camera connected')
                        reconnect_delay = 1.0
                    else:
                        time.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 2, 10.0)
                        continue
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    cap = None
                    continue
                with self.lock:
                    self.frame_buffer.append((time.time(), frame.copy()))
            except Exception as e:
                print(f'[ERROR] Capture: {e}')
                if cap:
                    cap.release()
                    cap = None
                time.sleep(1.0)
        if cap:
            cap.release()

    def _process_frames(self):
        last_ts = 0.0
        while not self.stop_event.is_set():
            now = time.time()
            if now - last_ts >= self.frame_interval:
                frame, ts = None, now
                with self.lock:
                    if self.frame_buffer:
                        ts, frame = self.frame_buffer[-1]
                        self.frame_buffer.clear()
                if frame is not None:
                    self._process_single_frame(frame, ts)
                    last_ts = now
            time.sleep(0.01)

    @staticmethod
    def _is_dup_box(box: Tuple, existing: List[Tuple], iou_thresh: float = 0.3) -> bool:
        """IoU + center-distance dedup. Replaces the old flat '25px' check,
        which was calibrated for full-frame-scale boxes only — it under-
        merges for tiny ROI-crop detections (a few px of jitter there is
        proportionally huge) and over-merges for large boxes far apart in
        a big frame. Center distance is checked in addition to IoU because
        two small, barely-overlapping boxes for the same real phone can
        have ~0 IoU while clearly being the same object.
        """
        x1, y1, x2, y2, _ = box
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        for (ex1, ey1, ex2, ey2, _) in existing:
            ix1, iy1 = max(x1, ex1), max(y1, ey1)
            ix2, iy2 = min(x2, ex2), min(y2, ey2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = bw * bh + max(ex2 - ex1, 1) * max(ey2 - ey1, 1) - inter
                if union > 0 and inter / union > iou_thresh:
                    return True
            ecx, ecy = (ex1 + ex2) / 2.0, (ey1 + ey2) / 2.0
            if abs(cx - ecx) < bw * 0.5 and abs(cy - ecy) < bh * 0.5:
                return True
        return False

    def _hand_roi_phone_boost(self, frame: np.ndarray, person_tracks: List[Tuple],
                               existing_phone_dets: List[Tuple]) -> List[Tuple]:
        """
        Small-object accuracy boost: crop a tight, upscaled region around
        each wrist keypoint and re-run the phone model on just that patch.

        Why: in a wide classroom shot a held phone can be a ~15-25px
        object. Full-frame detectors (generic and custom-trained alike)
        systematically under-detect objects that small — this is the same
        problem tiled/"sliced" inference (SAHI) exists to solve. We
        already have wrist keypoints from the pose model for free, so we
        can target exactly the region a phone would be in and upscale it
        2-2.5x before inference, which meaningfully increases recall on
        small/occluded phones without having to slice the whole frame.

        Gated to keep cost down:
        - Only for tracks whose bbox is small relative to the frame (i.e.
          camera is wide/far — the regime this actually helps). Close-up
          shots already give full-frame detection enough pixels to work with.
        - Skipped for a person who already has a confident phone hit inside
          their padded bbox from the full-frame pass — no need to re-check.
        """
        boosted: List[Tuple] = []
        if (self.phone_model is None and self.object_model is None) or not person_tracks:
            return boosted
        model = self.phone_model or self.object_model
        use_generic = model is self.object_model
        frame_h, frame_w = frame.shape[:2]

        for tid, x1, y1, x2, y2, conf, kp in person_tracks:
            bbox_h, bbox_w = y2 - y1, x2 - x1
            if bbox_h <= 0 or bbox_w <= 0 or kp is None or len(kp) <= 10:
                continue
            if bbox_h > frame_h * 0.42:
                continue  # person already large enough in-frame; full-frame pass suffices

            pad_x, pad_y = bbox_w * 0.25, bbox_h * 0.20
            already_found = any(
                px1 < x2 + pad_x and px2 > x1 - pad_x and py1 < y2 + pad_y and py2 > y1 - pad_y and pconf >= 0.40
                for (px1, py1, px2, py2, pconf) in existing_phone_dets
            )
            if already_found:
                continue

            wrist_pts = []
            for idx in (9, 10):
                if idx < len(kp):
                    w = kp[idx]
                    if len(w) >= 3 and w[2] >= 0.3 and w[0] != 0.0:
                        wrist_pts.append((float(w[0]), float(w[1])))
            if not wrist_pts:
                continue

            for (wx, wy) in wrist_pts:
                half = bbox_h * 0.22
                cx1, cy1 = int(max(0, wx - half)), int(max(0, wy - half))
                cx2, cy2 = int(min(frame_w, wx + half)), int(min(frame_h, wy + half))
                if cx2 - cx1 < 20 or cy2 - cy1 < 20:
                    continue
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                scale = 2.5 if max(crop.shape[:2]) < 200 else 1.5
                crop_rs = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

                try:
                    # Lower conf floor than the full-frame pass (0.20 vs 0.25)
                    # is deliberate: this crop is small and phone-centric by
                    # construction, so the base rate of noise here is much
                    # lower than scanning the whole frame at that threshold.
                    for result in model(crop_rs, verbose=False, conf=0.20, iou=0.45):
                        if result.boxes is None:
                            continue
                        for i in range(len(result.boxes)):
                            if use_generic and int(result.boxes.cls[i]) not in self._PHONE_CLS:
                                continue
                            pconf = float(result.boxes.conf[i])
                            bx1, by1, bx2, by2 = result.boxes.xyxy[i].tolist()
                            fx1, fy1 = cx1 + bx1 / scale, cy1 + by1 / scale
                            fx2, fy2 = cx1 + bx2 / scale, cy1 + by2 / scale
                            box = (int(fx1), int(fy1), int(fx2), int(fy2), pconf)
                            if not self._is_dup_box(box, existing_phone_dets) and not self._is_dup_box(box, boosted):
                                boosted.append(box)
                except Exception as e:
                    print(f'[WARN] ROI phone boost failed tid={tid}: {e}')

        if boosted:
            print(f'[PHONE-ROI] Found {len(boosted)} additional phone(s) via hand-ROI boost')
        return boosted

    def _update_static_fp_map(self, phone_dets: List[Tuple], person_tracks: List[Tuple]):
        """
        Tracks phone-shaped detections that recur at the same screen
        location without ever being near a person — almost certainly a
        fixed environmental false positive (poster, light switch,
        reflective surface) rather than an actual phone, which moves.

        Observed in production logs: the same ~(0,280)-(50,350) box got
        rejected over and over for a dozen different track IDs across
        multiple sessions. It was never causing a false alert (the
        per-person padded-bbox check already rejects it correctly), but
        it was pure wasted matching work and log spam every single frame.

        Uses a leaky per-grid-cell counter (accumulates on unmatched
        hits, decays otherwise) rather than a hard blacklist, so a real
        phone that happens to pass through that exact spot briefly won't
        get stuck suppressed.
        """
        seen_cells = set()
        for (px1, py1, px2, py2, conf) in phone_dets:
            cx, cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
            cell = (int(cx) // self._STATIC_FP_CELL, int(cy) // self._STATIC_FP_CELL)
            seen_cells.add(cell)
            near_person = any(
                px1 < x2 + 40 and px2 > x1 - 40 and py1 < y2 + 40 and py2 > y1 - 40
                for (_tid, x1, y1, x2, y2, *_rest) in person_tracks
            )
            val = self._static_fp_counters.get(cell, 0)
            val = min(val + 1, self._STATIC_FP_CAP) if not near_person else max(val - 2, 0)
            if val == 0:
                self._static_fp_counters.pop(cell, None)
            else:
                self._static_fp_counters[cell] = val

        for cell in list(self._static_fp_counters):
            if cell not in seen_cells:
                v = max(self._static_fp_counters[cell] - 1, 0)
                if v == 0:
                    del self._static_fp_counters[cell]
                else:
                    self._static_fp_counters[cell] = v

    def _filter_static_fp(self, phone_dets: List[Tuple]) -> List[Tuple]:
        if not self._static_fp_counters:
            return phone_dets
        kept = []
        for box in phone_dets:
            px1, py1, px2, py2, conf = box
            cx, cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
            cell = (int(cx) // self._STATIC_FP_CELL, int(cy) // self._STATIC_FP_CELL)
            if self._static_fp_counters.get(cell, 0) >= self._STATIC_FP_THRESHOLD:
                continue
            kept.append(box)
        return kept

    def _parse_object_detections(self, frame: np.ndarray, person_tracks: Optional[List[Tuple]] = None):
        phone_dets, food_dets, book_dets = [], [], []
        if self.object_model is not None:
            try:
                for result in self.object_model(frame, verbose=False, conf=0.25, iou=0.45):
                    if result.boxes is None:
                        continue
                    for i in range(len(result.boxes)):
                        cls  = int(result.boxes.cls[i])
                        conf = float(result.boxes.conf[i])
                        x1, y1, x2, y2 = map(int, result.boxes.xyxy[i])
                        det = (x1, y1, x2, y2, conf)
                        if cls in self._PHONE_CLS:
                            phone_dets.append(det)
                        elif cls in self._FOOD_CLS:
                            food_dets.append(det)
                        elif cls in self._BOOK_CLS:
                            book_dets.append(det)
            except Exception as e:
                print(f'[WARN] YOLO object detection failed: {e}')

        # Custom classroom-trained phone model. More reliable than the
        # generic model's COCO 'cell phone' class for this footage (small,
        # hand-occluded, off-angle phones). Always runs and merges with
        # the generic model's results, skipping near-duplicate boxes,
        # rather than only running when the generic model found nothing.
        if self.phone_model is not None:
            try:
                for result in self.phone_model(frame, verbose=False, conf=0.25, iou=0.45):
                    if result.boxes is None:
                        continue
                    new_count = 0
                    for i in range(len(result.boxes)):
                        conf = float(result.boxes.conf[i])
                        x1, y1, x2, y2 = map(int, result.boxes.xyxy[i])
                        box = (x1, y1, x2, y2, conf)
                        if not self._is_dup_box(box, phone_dets):
                            phone_dets.append(box)
                            new_count += 1
                    if new_count:
                        print(f'[PHONE-MODEL] Found {new_count} additional phone(s)')
            except Exception as e:
                print(f'[WARN] Custom phone model failed: {e}')

        # Static-false-positive suppression — filter out phone-shaped
        # detections recurring at a fixed, person-less screen location
        # before spending any more work on them (heuristic matching,
        # ROI boost, logging).
        self._update_static_fp_map(phone_dets, person_tracks or [])
        phone_dets = self._filter_static_fp(phone_dets)

        # Hand-ROI boost pass (see _hand_roi_phone_boost docstring) — only
        # meaningfully useful once we know where people/wrists are, so it
        # runs last and merges into the same list.
        if person_tracks:
            phone_dets.extend(self._hand_roi_phone_boost(frame, person_tracks, phone_dets))

        return phone_dets, food_dets, book_dets

    def _parse_pose_detections(self, frame: np.ndarray):
        tracks = []
        if self.yolo_model is None:
            return tracks
        try:
            for result in self.yolo_model.track(frame, persist=True, verbose=False, conf=0.1, iou=0.5, tracker='bytetrack_classroom.yaml'):
                if result.boxes is None:
                    continue
                kp_list = result.keypoints if hasattr(result, 'keypoints') else None
                for i in range(len(result.boxes)):
                    if int(result.boxes.cls[i]) != 0:
                        continue
                    x1, y1, x2, y2 = map(int, result.boxes.xyxy[i])
                    conf     = float(result.boxes.conf[i])
                    track_id = int(result.boxes.id[i]) if result.boxes.id is not None else i
                    kp = None
                    if kp_list is not None and i < len(kp_list):
                        try:
                            kp = kp_list[i].data.cpu().numpy()[0]
                        except Exception:
                            pass
                    tracks.append((track_id, x1, y1, x2, y2, conf, kp))
        except Exception as e:
            print(f'[ERROR] Pose detection: {e}')
        return tracks

    def _run_behavior_evaluation(self, frame, person_tracks, phone_dets, food_dets, book_dets, timestamp, fight_detected=False):
        active_tids = set()
        for tid, x1, y1, x2, y2, conf, kp in person_tracks:
            active_tids.add(tid)
            self.behavior_engine.update_person(tid, (x1, y1, x2, y2), kp, timestamp)
        self.behavior_engine.cleanup_stale(timestamp)
        results = []
        for tid in active_tids:
            with self.behavior_engine.lock:
                if tid not in self.behavior_engine.tracked_people:
                    continue
                person = self.behavior_engine.tracked_people[tid]
            if fight_detected:
                result = DetectionResult(
                    type='fighting', bbox=person.bbox, confidence=0.9,
                    color=(0,0,255), label='Fighting!', is_alert=True, is_distracted=False,
                    track_id=tid
                )
                results.append(result)
            else:
                head_pose = self.head_pose_detector.calculate_head_pose(person)
                is_hand_raised, hand_conf = self.hand_raise_detector.detect_hand_raise(person)
                raw_behavior = ""
                raw_confidence = 0.0
                if is_hand_raised:
                    raw_behavior = "hand_raised"
                    raw_confidence = hand_conf
                else:
                    is_phone, phone_conf = self.phone_detector.detect_phone_usage(person, phone_dets, head_pose, book_dets)
                    if is_phone:
                        raw_behavior = "using_phone"
                        raw_confidence = phone_conf
                    else:
                        is_eating, eating_conf = self.food_detector.detect_eating(person, food_dets)
                        if is_eating:
                            raw_behavior = "eating_food"
                            raw_confidence = eating_conf
                        else:
                            if head_pose == "focused":
                                raw_behavior   = "focused"
                                raw_confidence = 0.75
                            elif head_pose == "head_down":
                                # hands_near_book relies on COCO's generic "book" class,
                                # which essentially never fires on an open notebook/notepad
                                # at a normal writing angle -- so this veto almost never
                                # actually confirms real writing, and writers with head
                                # down get mislabeled "distracted" by default. Wrist-motion
                                # variance (already computed for phone-vs-writing
                                # disambiguation elsewhere) is a much more reliable signal
                                # here: real handwriting produces small, repetitive wrist
                                # motion that phone-holding or idle hands don't.
                                #
                                # This veto only applies to head_down (writing posture).
                                # looking_away means the head is turned aside, not down at
                                # a desk -- there's no legitimate "writing" interpretation
                                # for that, so it no longer shares this branch. Previously
                                # bundling both meant a student looking away with any
                                # incidental hand movement (adjusting a phone, gesturing)
                                # got relabeled "focused" instead of "distracted".
                                is_writing_motion, _ = SharedHelpers.calculate_wrist_motion_variance(person)
                                if SharedHelpers.hands_near_book(person, book_dets) or is_writing_motion:
                                    raw_behavior   = "focused"
                                    raw_confidence = 0.70
                                else:
                                    raw_behavior   = "distracted"
                                    raw_confidence = 0.70
                            elif head_pose == "looking_away":
                                raw_behavior   = "distracted"
                                raw_confidence = 0.70
                            elif head_pose == "not_visible":
                                raw_behavior   = "not_visible"
                                raw_confidence = 0.60
                            else:
                                raw_behavior   = "distracted"
                                raw_confidence = 0.65
                final_behavior, confidence = self.behavior_engine.evaluate_final_behavior(person, raw_behavior, raw_confidence)
                is_alert = final_behavior in ALERT_POSES
                is_distracted = final_behavior in DISTRACTED_POSES
                results.append(DetectionResult(
                    type=final_behavior, bbox=person.bbox, confidence=confidence,
                    color=COLOR_MAP.get(final_behavior, COLOR_MAP["not_visible"]),
                    label=LABEL_MAP.get(final_behavior, final_behavior),
                    is_alert=is_alert, is_distracted=is_distracted, track_id=tid
                ))
        return results

    def _process_single_frame(self, frame: np.ndarray, timestamp: float):
        t0 = time.time()
        try:
            person_tracks = self._parse_pose_detections(frame)
            phone_dets, food_dets, book_dets = self._parse_object_detections(frame, person_tracks)
            fight_detected = False
            if self.fight_detector is not None:
                self.fight_detector.add_frame(frame)   # buffer frames for 3D CNN
                fight_result = self.fight_detector.predict()
                fight_detected = fight_result[0] if isinstance(fight_result, tuple) else bool(fight_result)
                if fight_detected:
                    print('[ALERT] Fight detected!')
            final_results = self._run_behavior_evaluation(frame, person_tracks, phone_dets, food_dets, book_dets, timestamp, fight_detected)
            with self.lock:
                self.phone_detections = phone_dets
                self.food_detections  = food_dets
                self.person_tracks    = [(t[0], t[1], t[2], t[3], t[4], t[5]) for t in person_tracks]
                self.result_buffer.append((timestamp, frame.copy(), final_results))
        except Exception as e:
            print(f'[ERROR] Frame processing: {e}')
        finally:
            elapsed = time.time() - t0
            if elapsed > self.frame_interval * 1.5:
                print(f'[PERF] Frame processing took {elapsed:.2f}s (target {self.frame_interval:.2f}s) — falling behind')

    def start(self, camera_url: str):
        if self.running:
            return
        self.running = True
        self.stop_event.clear()
        threading.Thread(target=self._capture_frames, args=(camera_url,), daemon=True).start()
        threading.Thread(target=self._process_frames, daemon=True).start()
        print('[OK] Stream processor started')

    def stop(self):
        self.running = False
        self.stop_event.set()
        print('[OK] Stream processor stopped')

    def get_latest_result(self) -> Tuple[Optional[float], Optional[np.ndarray], List[DetectionResult]]:
        with self.lock:
            return self.result_buffer[-1] if self.result_buffer else (None, None, [])


# ── ClassroomBehaviorDetector (backward-compatible public API) ───────────────
class ClassroomBehaviorDetector:
    def __init__(self, camera_url, camera_id, server_url='http://localhost:8000', alert_cooldown=120, whatsapp_admin=None):
        self.camera_url      = camera_url
        self.camera_id       = camera_id
        self.server_url      = server_url.rstrip('/')
        self.alert_cooldown  = alert_cooldown
        self.whatsapp_admin  = whatsapp_admin or os.environ.get('ADMIN_WHATSAPP', '')
        self._face_recognizer = None
        self.known_students  = []
        self.last_alert_time: dict = defaultdict(float)
        self.running         = False
        self.thread          = None
        self.processor       = ProductionStreamProcessor(process_fps=10)
        self._api_key        = os.environ.get('DETECTION_API_KEY', '').strip()

    @property
    def behavior_engine(self):
        return self.processor.behavior_engine

    @property
    def yolo_model(self):
        return self.processor.yolo_model

    @property
    def object_model(self):
        return self.processor.object_model

    def _init_face_recognition(self):
        if self._face_recognizer is not None:
            return
        try:
            from classroom_monitor.face_recognition_helper import StudentFaceRecognizer
            rec = StudentFaceRecognizer()
            rec.load_from_db()
            if rec._known_encodings:
                self._face_recognizer = rec
                print(f'[OK] Face recognition: {len(rec._known_encodings)} encodings from DB')
                return
        except Exception as e:
            print(f'[WARN] DB face encodings unavailable ({e}), trying HTTP')
        self._load_known_students_http()

    def _load_known_students_http(self):
        import requests as _req
        url = f'{self.server_url}/camera-attendance/api/students/encodings/'
        headers = {}
        if self._api_key:
            headers['X-Detection-API-Key'] = self._api_key
        try:
            r = _req.get(url, timeout=5, verify=_http_verify_ssl(), headers=headers)
            r.raise_for_status()
            self.known_students = r.json()
            print(f'[OK] {len(self.known_students)} student encodings loaded via HTTP')
        except Exception as e:
            print(f'[WARN] Could not load students: {e}')
            self.known_students = []

    def evaluate_tracks(self, person_tracks, phone_dets=None, food_dets=None,
                        book_dets=None, fight_detected=False) -> List[Dict]:
        """
        Lightweight companion to `detect()` — runs ONLY behavior evaluation
        (head pose, hand raise, phone, distraction, fight) against tracks a
        caller already has, using whatever phone/food/book detections were
        most recently produced by a separate (slower) object-detection
        worker. No YOLO model is called here.

        Why this exists: head_pose_detection.py / phone_detection.py use
        LEAKY COUNTERS with thresholds (e.g. ~6-10 matched calls) tuned
        assuming evaluation happens close to pose-tracking frequency
        (~8fps). If behavior evaluation only ever runs at the much slower
        heavy object-detection cadence (~2fps, since that path also calls
        the phone/food/book YOLO models), those same thresholds take
        several REAL SECONDS to confirm a state change, and a brief
        gesture (a quick hand-raise) can start and end entirely between
        two heavy-worker ticks and never get evaluated at all.

        Call this once per pose-tracking cycle (e.g. from an ~8fps pose
        worker) with the latest pose tracks and the latest cached
        phone/food/book detections — this makes hand-raise/phone/
        distracted/focused state changes show up within roughly one pose
        cycle instead of lagging behind the heavy-detection cadence.
        """
        if self.processor.yolo_model is None:
            return []
        try:
            timestamp = time.time()
            det_objs = self.processor._run_behavior_evaluation(
                None, person_tracks, phone_dets or [], food_dets or [],
                book_dets or [], timestamp, fight_detected,
            )
            detections = []
            for d in det_objs:
                detections.append({
                    'type': d.type, 'bbox': d.bbox, 'confidence': d.confidence,
                    'color': d.color, 'label': d.label, 'is_alert': d.is_alert,
                    'is_distracted': d.is_distracted, 'track_id': d.track_id,
                })
            return detections
        except Exception as e:
            print(f'[ERROR] evaluate_tracks(): {e}')
            return []

    def detect(self, frame, person_tracks=None) -> List[Dict]:
        """
        `person_tracks` lets a caller pass in tracks already computed by a
        higher-frequency pose worker instead of having this method call
        `.track()` again itself. Two independent callers each calling
        `.track(persist=True)` on the same shared ByteTrack tracker — one
        from a pose loop, one from here — can feed it frames out of
        chronological order (this path is much slower per-call, since it
        also runs the phone/food/book models), which corrupts ByteTrack's
        motion prediction and causes constant track ID reassignment. Pass
        `person_tracks=None` only for callers that don't also run a
        separate pose-tracking loop against the same detector.
        """
        if self.processor.yolo_model is None:
            return []
        try:
            timestamp     = time.time()
            if person_tracks is None:
                person_tracks = self.processor._parse_pose_detections(frame)
            phone_dets, food_dets, book_dets = self.processor._parse_object_detections(frame, person_tracks)
            det_objs = self.processor._run_behavior_evaluation(frame, person_tracks, phone_dets, food_dets, book_dets, timestamp)
            detections = []
            for d in det_objs:
                entry = {
                    'type': d.type, 'bbox': d.bbox, 'confidence': d.confidence,
                    'color': d.color, 'label': d.label, 'is_alert': d.is_alert,
                    'is_distracted': d.is_distracted, 'track_id': d.track_id
                }
                detections.append(entry)
            return detections
        except Exception as e:
            print(f'[ERROR] detect(): {e}')
            return []

    def _detect_behaviors(self, frame):
        return self.detect(frame)

    def _draw_detections(self, frame, detections):
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color, label, conf = det['color'], det['label'], det['confidence']
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"{label} ({conf:.2f})"
            tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            bg_y1 = max(0, y1 - th - 4)
            cv2.rectangle(out, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, text, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out

    def _report_incident(self, detection, frame, student_id, student_name, roll_no, all_detections=None):
        import requests as _req
        try:
            annotated = self._draw_detections(frame, all_detections) if all_detections else frame
            _, buf   = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            snap_b64 = base64.b64encode(buf).decode()
            tag = f"{student_name} ({roll_no})" if student_id else 'Unknown'
            resp = _req.post(
                f'{self.server_url}/classroom/api/incidents/report/',
                json={
                    'student_id': student_id, 'camera_id': self.camera_id, 'incident_type': detection['type'],
                    'confidence': round(detection['confidence'], 3), 'snapshot': snap_b64,
                    'student_name': student_name, 'roll_no': roll_no,
                    'description': f"ALERT {detection['label']} — {tag}"
                },
                headers={'X-Detection-API-Key': self._api_key} if self._api_key else {},
                timeout=10, verify=_http_verify_ssl()
            )
            print(f"[INCIDENT] {detection['label']} | {tag} | {resp.status_code}")
        except Exception as e:
            print(f'[ERROR] report_incident: {e}')

    def _recognize_face(self, frame, bbox):
        self._init_face_recognition()
        try:
            x1, y1, x2, y2 = bbox
            mid_y = y1 + int((y2 - y1) * 0.55)
            crop  = frame[y1:mid_y, x1:x2]
            if crop.size == 0:
                crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None, 'Unknown', ''

            if self._face_recognizer is not None:
                sid, name, roll, _dist = self._face_recognizer.match(crop)
                if sid:
                    return sid, name, roll
                return None, name, roll

            if not self.known_students:
                return None, 'Unknown', ''

            import face_recognition as fr
            rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            encs = fr.face_encodings(rgb, num_jitters=1, model='small')
            if not encs:
                return None, 'Unknown', ''
            det    = encs[0]
            best_d, best = 1.0, None
            for s in self.known_students:
                try:
                    d = fr.face_distance([np.array(json.loads(s['encoding']))], det)[0]
                    if d < best_d:
                        best_d, best = d, s
                except Exception:
                    continue
            if best_d < FACE_MATCH_TOLERANCE and best:
                return best['id'], best['name'], best.get('roll_no', '')
            return None, 'Unknown', ''
        except Exception as e:
            print(f'[ERROR] face_recog: {e}')
            return None, 'Unknown', ''

    def _detection_loop(self):
        self.processor.start(self.camera_url)
        frame_count, last_ts = 0, 0.0
        while self.running:
            ts, frame, det_objs = self.processor.get_latest_result()
            if frame is None or ts <= last_ts:
                time.sleep(0.01)
                continue
            last_ts = ts
            detections = []
            for d in det_objs:
                entry = {
                    'type': d.type, 'bbox': d.bbox, 'confidence': d.confidence,
                    'color': d.color, 'label': d.label, 'is_alert': d.is_alert,
                    'is_distracted': d.is_distracted, 'track_id': d.track_id
                }
                detections.append(entry)
            frame_count += 1
            now = time.time()
            for det in detections:
                if not (det['is_alert'] or det['is_distracted']):
                    continue
                tid = det.get('track_id')
                key = (det['type'], tid if tid is not None else tuple(det['bbox']))
                if now - self.last_alert_time.get(key, 0) < self.alert_cooldown:
                    continue
                sid, name, roll = self._recognize_face(frame, det['bbox'])
                self._report_incident(det, frame, sid, name, roll, all_detections=detections)
                self.last_alert_time[key] = now
            if frame_count % 300 == 0:
                print(f'[INFO] Frame {frame_count} | {len(detections)} detections')
        self.processor.stop()

    def start(self):
        if self.running:
            return
        self._init_face_recognition()
        self.running = True
        self.thread  = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print('[OK] Behavior detection started')

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.processor.stop()
        print('[OK] Behavior detection stopped')