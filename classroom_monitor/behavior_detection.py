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
            return cls._pose_model, cls._object_model


# ── Production Stream Processor ──────────────────────────────────────────────
class ProductionStreamProcessor:
    _PHONE_CLS = {67}
    _FOOD_CLS  = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55}
    _BOOK_CLS  = {73}

    # Fight-detection tuning. FIGHT_CONFIRM_FRAMES is the number of
    # consecutive positive frames from the 3D-CNN classifier required
    # before a fight becomes a confirmed, incident-reportable behavior.
    # At process_fps=10, 3 frames ~= 0.3s of sustained signal — tune
    # against the actual pipeline frame rate.
    FIGHT_CONFIRM_FRAMES    = 3
    # Fallback localization (used only if the classifier gives no
    # spatial info): two people are fight candidates only if their bbox
    # centers are within this multiple of their average bbox diagonal...
    FIGHT_PROXIMITY_RATIO   = 1.5
    # ...AND both show normalized keypoint motion (frame-over-frame
    # displacement / bbox diagonal) at or above this threshold.
    FIGHT_MOTION_THRESHOLD  = 0.12

    def __init__(self, process_fps: int = 10, buffer_size: int = 5):
        self.process_fps      = process_fps
        self.frame_interval   = 1.0 / process_fps
        self.frame_buffer: deque = deque(maxlen=buffer_size)
        self.result_buffer: deque = deque(maxlen=2)
        self.yolo_model       = None
        self.object_model     = None
        self.roboflow_model   = None
        self.running          = False
        self.lock             = threading.Lock()
        self.stop_event       = threading.Event()
        self.phone_detections: List[Tuple] = []
        self.food_detections:  List[Tuple] = []
        self.person_tracks: List[Tuple] = []
        self.behavior_engine  = TemporalBehaviorEngine()
        self.fight_detector   = None
        self._fight_streak    = 0          # consecutive positive fight_detected frames
        self._prev_kp_by_tid: Dict[int, np.ndarray] = {}  # last-frame keypoints, for motion heuristic

        # Initialize modular detectors
        self.head_pose_detector = HeadPoseDetector()
        self.phone_detector     = PhoneDetector()
        self.hand_raise_detector = HandRaiseDetector()
        self.food_detector      = FoodDetector()

        self._ensure_models()
        self._init_fight_detector()

    def _ensure_models(self):
        self.yolo_model, self.object_model = _SharedYOLOModels.get()

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
        """DEPRECATED / UNUSED IN THE LIVE PATH — only reachable via start(),
        which nothing in views.py calls. See start()'s docstring."""
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
        """DEPRECATED / UNUSED IN THE LIVE PATH — only reachable via start(),
        which nothing in views.py calls. See start()'s docstring."""
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
    def _iou(box_a: Tuple, box_b: Tuple) -> float:
        ax1, ay1, ax2, ay2 = box_a[:4]
        bx1, by1, bx2, by2 = box_b[:4]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _merge_phone_detections(self, coco_dets: List[Tuple], roboflow_dets: List[Tuple],
                                 iou_thresh: float = 0.4) -> List[Tuple]:
        """Merge COCO generic-object and Roboflow classroom-specific phone
        detections, deduping overlapping boxes by IoU rather than treating
        either source as authoritative. On overlap, keep whichever box has
        the higher confidence."""
        merged = list(coco_dets)
        for rdet in roboflow_dets:
            dup_idx = None
            for i, cdet in enumerate(merged):
                if self._iou(rdet, cdet) >= iou_thresh:
                    dup_idx = i
                    break
            if dup_idx is None:
                merged.append(rdet)
            elif rdet[4] > merged[dup_idx][4]:
                merged[dup_idx] = rdet
        return merged

    def _parse_object_detections(self, frame: np.ndarray):
        phone_dets, food_dets, book_dets = [], [], []
        if self.object_model is not None:
            try:
                for result in self.object_model(frame, verbose=False, conf=0.3, iou=0.45):
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

        # Roboflow's classroom-specific phone model always runs (when
        # available) rather than only as a fallback when COCO found
        # nothing — it's purpose-trained for this exact scenario and
        # shouldn't be gated behind a single low-confidence COCO miss.
        # Results are merged/deduped with the COCO phone detections by
        # IoU instead of either source overriding the other outright.
        roboflow_dets = []
        try:
            api_key = os.environ.get('ROBOFLOW_API_KEY', '')
            if api_key:
                if self.roboflow_model is None:
                    from roboflow import Roboflow
                    rf = Roboflow(api_key=api_key)
                    project = rf.workspace().project("classroom-cell-phone-detection")
                    self.roboflow_model = project.version(18).model
                    print('[OK] Roboflow model loaded')
                result = self.roboflow_model.predict(frame, confidence=30, overlap=30).json()
                for prediction in result.get('predictions', []):
                    x1 = int(prediction['x'] - prediction['width'] / 2)
                    y1 = int(prediction['y'] - prediction['height'] / 2)
                    x2 = int(prediction['x'] + prediction['width'] / 2)
                    y2 = int(prediction['y'] + prediction['height'] / 2)
                    conf = prediction['confidence']
                    roboflow_dets.append((x1, y1, x2, y2, conf))
                if roboflow_dets:
                    print(f'[ROBOFLOW] Found {len(roboflow_dets)} phone(s)')
        except ImportError:
            print('[WARN] roboflow not installed')
        except Exception as e:
            print(f'[WARN] Roboflow failed: {e}')

        phone_dets = self._merge_phone_detections(phone_dets, roboflow_dets)
        return phone_dets, food_dets, book_dets


    def _parse_pose_detections(self, frame: np.ndarray):
        tracks = []
        if self.yolo_model is None:
            return tracks
        try:
            for result in self.yolo_model.track(frame, persist=True, verbose=False, conf=0.3, iou=0.5, tracker='bytetrack.yaml'):
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

    @staticmethod
    def _bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _bbox_diag(bbox):
        x1, y1, x2, y2 = bbox
        return float(np.hypot(x2 - x1, y2 - y1))

    def _keypoint_motion_score(self, tid, kp):
        """Average frame-over-frame displacement of confident keypoints for a track.
        Returns 0.0 if there's no usable previous-frame keypoints to compare against."""
        if kp is None:
            return 0.0
        prev_kp = self._prev_kp_by_tid.get(tid)
        if prev_kp is None or prev_kp.shape != kp.shape:
            return 0.0
        diffs = []
        for i in range(len(kp)):
            conf_cur  = kp[i][2] if kp.shape[-1] > 2 else 1.0
            conf_prev = prev_kp[i][2] if prev_kp.shape[-1] > 2 else 1.0
            if conf_cur < 0.3 or conf_prev < 0.3:
                continue
            dx = kp[i][0] - prev_kp[i][0]
            dy = kp[i][1] - prev_kp[i][1]
            diffs.append((dx * dx + dy * dy) ** 0.5)
        if not diffs:
            return 0.0
        return float(np.mean(diffs))

    def _localize_fight_participants(self, person_tracks, fight_region):
        """
        Determine which track_ids are actually involved in a confirmed fight,
        instead of tagging every tracked person.

        Priority:
          1. fight_region gives explicit track_ids  -> use them directly.
          2. fight_region gives a spatial bbox/region -> flag tracks whose
             center falls inside it.
          3. No spatial info at all -> fallback heuristic: flag pairs of
             people who are both physically close AND both show high
             frame-over-frame keypoint motion (proxy for a scuffle rather
             than two people just standing near each other).
        """
        track_ids = {t[0] for t in person_tracks}

        if fight_region:
            if isinstance(fight_region, dict) and 'track_ids' in fight_region:
                return set(fight_region['track_ids']) & track_ids
            if isinstance(fight_region, (list, tuple, set)) and fight_region and \
                    all(isinstance(v, (int, np.integer)) for v in fight_region):
                return set(fight_region) & track_ids
            if isinstance(fight_region, (list, tuple)) and len(fight_region) == 4:
                rx1, ry1, rx2, ry2 = fight_region
                involved = set()
                for tid, x1, y1, x2, y2, conf, kp in person_tracks:
                    cx, cy = self._bbox_center((x1, y1, x2, y2))
                    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                        involved.add(tid)
                if involved:
                    return involved

        # Fallback: proximity + high relative motion, evaluated pairwise.
        involved = set()
        tracks = list(person_tracks)
        for i in range(len(tracks)):
            tid_a, ax1, ay1, ax2, ay2, aconf, akp = tracks[i]
            diag_a = self._bbox_diag((ax1, ay1, ax2, ay2))
            if diag_a <= 0:
                continue
            for j in range(i + 1, len(tracks)):
                tid_b, bx1, by1, bx2, by2, bconf, bkp = tracks[j]
                diag_b = self._bbox_diag((bx1, by1, bx2, by2))
                if diag_b <= 0:
                    continue
                cax, cay = self._bbox_center((ax1, ay1, ax2, ay2))
                cbx, cby = self._bbox_center((bx1, by1, bx2, by2))
                dist = float(np.hypot(cax - cbx, cay - cby))
                avg_diag = (diag_a + diag_b) / 2.0
                if dist > avg_diag * self.FIGHT_PROXIMITY_RATIO:
                    continue  # not near each other -> not a candidate pair
                norm_motion_a = self._keypoint_motion_score(tid_a, akp) / diag_a
                norm_motion_b = self._keypoint_motion_score(tid_b, bkp) / diag_b
                if norm_motion_a >= self.FIGHT_MOTION_THRESHOLD and \
                        norm_motion_b >= self.FIGHT_MOTION_THRESHOLD:
                    involved.add(tid_a)
                    involved.add(tid_b)
        return involved

    def _run_behavior_evaluation(self, frame, person_tracks, phone_dets, food_dets, book_dets, timestamp,
                                  fight_detected=False, fight_region=None):
        active_tids = set()
        for tid, x1, y1, x2, y2, conf, kp in person_tracks:
            active_tids.add(tid)
            self.behavior_engine.update_person(tid, (x1, y1, x2, y2), kp, timestamp)
        self.behavior_engine.cleanup_stale(timestamp)

        # --- Fight signal: multi-frame confirmation + spatial localization ---
        # Mirrors the hysteresis every other behavior type gets via
        # evaluate_final_behavior; a single noisy frame can no longer
        # promote straight to a reportable, emailed alert.
        self._fight_streak = self._fight_streak + 1 if fight_detected else 0
        fight_confirmed = self._fight_streak >= self.FIGHT_CONFIRM_FRAMES

        fighting_tids = set()
        if fight_confirmed:
            fighting_tids = self._localize_fight_participants(person_tracks, fight_region)

        # Cache this frame's keypoints for next frame's motion comparison.
        for tid, x1, y1, x2, y2, conf, kp in person_tracks:
            if kp is not None:
                self._prev_kp_by_tid[tid] = kp
        for stale_tid in set(self._prev_kp_by_tid) - active_tids:
            self._prev_kp_by_tid.pop(stale_tid, None)

        results = []
        for tid in active_tids:
            with self.behavior_engine.lock:
                if tid not in self.behavior_engine.tracked_people:
                    continue
                person = self.behavior_engine.tracked_people[tid]
            if tid in fighting_tids:
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
                            elif head_pose in ("head_down", "looking_away"):
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
        try:
            person_tracks = self._parse_pose_detections(frame)
            phone_dets, food_dets, book_dets = self._parse_object_detections(frame)
            fight_detected = False
            fight_region = None  # optional spatial localization from the detector, if it provides any
            if self.fight_detector is not None:
                self.fight_detector.add_frame(frame)   # buffer frames for 3D CNN
                fight_result = self.fight_detector.predict()
                if isinstance(fight_result, tuple):
                    fight_detected = bool(fight_result[0])
                    # Some detector variants may return (detected, confidence, region_or_track_ids)
                    if len(fight_result) > 2:
                        fight_region = fight_result[2]
                else:
                    fight_detected = bool(fight_result)
                if fight_detected:
                    print('[ALERT] Fight signal this frame (pending multi-frame confirmation)')
            final_results = self._run_behavior_evaluation(
                frame, person_tracks, phone_dets, food_dets, book_dets, timestamp,
                fight_detected, fight_region
            )
            with self.lock:
                self.phone_detections = phone_dets
                self.food_detections  = food_dets
                self.person_tracks    = [(t[0], t[1], t[2], t[3], t[4], t[5]) for t in person_tracks]
                self.result_buffer.append((timestamp, frame.copy(), final_results))
        except Exception as e:
            print(f'[ERROR] Frame processing: {e}')

    def start(self, camera_url: str):
        """
        DEPRECATED / UNUSED IN THE LIVE PATH.

        Nothing in views.py calls this. The actual live stream
        (_generate_video_stream in views.py) drives its own manual
        cv2.VideoCapture loop + its own worker threads and calls
        detector.detect(frame) -> ProductionStreamProcessor's parsing/
        evaluation methods directly, frame by frame, on the shared
        detector singleton (_get_yolo_detector()).

        Do NOT re-enable this (or wire it up in a new call site) against
        that same shared detector/processor instance: it spins up its
        own _capture_frames + _process_frames threads, which would open
        a second capture on the camera and contend with the manual
        pipeline for the same YOLO model. If you need this pipeline,
        give it its own ProductionStreamProcessor instance, not the
        shared one used by _generate_video_stream.
        """
        if self.running:
            return
        self.running = True
        self.stop_event.clear()
        threading.Thread(target=self._capture_frames, args=(camera_url,), daemon=True).start()
        threading.Thread(target=self._process_frames, daemon=True).start()
        print('[OK] Stream processor started')

    def stop(self):
        """DEPRECATED / UNUSED IN THE LIVE PATH — see start() docstring above."""
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

    def detect(self, frame) -> List[Dict]:
        if self.processor.yolo_model is None:
            return []
        try:
            timestamp     = time.time()
            person_tracks = self.processor._parse_pose_detections(frame)
            phone_dets, food_dets, book_dets = self.processor._parse_object_detections(frame)
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
        """
        DEPRECATED / UNUSED IN THE LIVE PATH.

        Only reachable via start() below, which nothing in views.py calls.
        The live stream (_generate_video_stream in views.py) calls
        self.detect(frame) directly per-frame from its own worker thread
        instead of running this loop. See ProductionStreamProcessor.start()
        for why this shouldn't be re-enabled against a shared detector.
        """
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
        """DEPRECATED / UNUSED IN THE LIVE PATH — see _detection_loop's docstring.
        Nothing in views.py calls this; the shared detector singleton
        (_get_yolo_detector) is driven entirely via .detect(frame) instead."""
        if self.running:
            return
        self._init_face_recognition()
        self.running = True
        self.thread  = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        print('[OK] Behavior detection started')

    def stop(self):
        """DEPRECATED / UNUSED IN THE LIVE PATH — see _detection_loop's docstring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.processor.stop()
        print('[OK] Behavior detection stopped')