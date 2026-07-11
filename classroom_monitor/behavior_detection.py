"""
Behavior Detection — Production Stream Processor
=================================================
Refactored pipeline:
  1. Pose model runs on the full frame to get person tracks.
  2. Per-person padded RoI crops are extracted and passed to the object detector
     (prevents small-object downscaling loss on wide classroom shots).
  3. Roboflow fallback runs on the crop only, never the full frame.
  4. Local crop detections are remapped to global frame coordinates.
  5. PhoneDetector cleanup_stale() is called each frame to free memory.
"""

import cv2
import json
import time
import base64
import numpy as np
import threading
import os
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple

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


# ── Shared YOLO models (one load per process) ─────────────────────────────────
class _SharedYOLOModels:
    _lock         = threading.Lock()
    _pose_model   = None
    _object_model = None

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._pose_model is None:
                try:
                    from ultralytics import YOLO
                    cls._pose_model   = YOLO('yolo11s-pose.pt')
                    cls._object_model = YOLO('yolo11s.pt')
                    print('[OK] Shared YOLO pose + object models loaded')
                except Exception as e:
                    print(f'[WARN] Shared YOLO model load failed: {e}')
            return cls._pose_model, cls._object_model


# ── Production Stream Processor ───────────────────────────────────────────────
class ProductionStreamProcessor:
    _PHONE_CLS = {67}
    _FOOD_CLS  = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55}
    _BOOK_CLS  = {73}

    # RoI crop config
    _ROI_PAD        = 0.15   # 15% padding around each person bbox
    _MIN_CROP_PX    = 32     # skip inference if crop is smaller than this

    def __init__(self, process_fps: int = 10, buffer_size: int = 5):
        self.process_fps    = process_fps
        self.frame_interval = 1.0 / process_fps
        self.frame_buffer:  deque = deque(maxlen=buffer_size)
        self.result_buffer: deque = deque(maxlen=2)
        self.yolo_model     = None
        self.object_model   = None
        self.roboflow_model = None
        self.running        = False
        self.lock           = threading.Lock()
        self.stop_event     = threading.Event()
        self.phone_detections: List[Tuple] = []
        self.food_detections:  List[Tuple] = []
        self.person_tracks:    List[Tuple] = []
        self.behavior_engine   = TemporalBehaviorEngine()
        self.fight_detector    = None

        # Modular detectors — PhoneDetector receives process_fps for window sizing
        self.head_pose_detector  = HeadPoseDetector()
        self.phone_detector      = PhoneDetector(process_fps=process_fps)
        self.hand_raise_detector = HandRaiseDetector()
        self.food_detector       = FoodDetector()

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

    # ── RoI helpers ───────────────────────────────────────────────────────────

    def _extract_person_crop(
        self, frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        pad: float = None,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """
        Extract a padded crop around a person bbox, clamped to frame bounds.
        Returns (crop, (origin_x, origin_y)) or (None, None) if too small.
        """
        pad = pad if pad is not None else self._ROI_PAD
        fh, fw = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1

        px1 = max(0, int(x1 - bw * pad))
        py1 = max(0, int(y1 - bh * pad))
        px2 = min(fw, int(x2 + bw * pad))
        py2 = min(fh, int(y2 + bh * pad))

        if (px2 - px1) < self._MIN_CROP_PX or (py2 - py1) < self._MIN_CROP_PX:
            return None, None

        return frame[py1:py2, px1:px2].copy(), (px1, py1)

    @staticmethod
    def _remap_to_global(
        dets: List[Tuple], origin: Tuple[int, int]
    ) -> List[Tuple]:
        """Map local crop detection coords back to global frame coordinates."""
        ox, oy = origin
        return [(x1 + ox, y1 + oy, x2 + ox, y2 + oy, conf)
                for (x1, y1, x2, y2, conf) in dets]

    # ── Per-person object detection (RoI-based) ───────────────────────────────

    def _parse_object_detections_for_person(
        self,
        crop:     np.ndarray,
        origin:   Tuple[int, int],
        track_id: int,
    ) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
        """
        Run object detection on a single person's padded crop.
        Returns (phone_dets, food_dets, book_dets) in GLOBAL frame coordinates.

        Roboflow is called only when YOLO finds zero phones in the crop.
        """
        phone_dets, food_dets, book_dets = [], [], []

        # ── Local YOLO object inference on crop ───────────────────────────────
        if self.object_model is not None:
            try:
                for result in self.object_model(
                    crop, verbose=False, conf=0.55, iou=0.45
                ):
                    if result.boxes is None:
                        continue
                    for i in range(len(result.boxes)):
                        cls  = int(result.boxes.cls[i])
                        conf = float(result.boxes.conf[i])
                        lx1, ly1, lx2, ly2 = map(int, result.boxes.xyxy[i])
                        det = (lx1, ly1, lx2, ly2, conf)
                        if cls in self._PHONE_CLS:
                            phone_dets.append(det)
                        elif cls in self._FOOD_CLS:
                            food_dets.append(det)
                        elif cls in self._BOOK_CLS:
                            book_dets.append(det)
            except Exception as e:
                print(f'[WARN] YOLO object detection failed (track {track_id}): {e}')

        # ── Roboflow fallback — disabled: too noisy on classroom crops ────────
        # Roboflow models trained on phone datasets fire heavily on books,
        # notebooks and papers when given close-cropped person images.
        # Re-enable only if you have a model fine-tuned on your classroom data.

        # Remap all local coords → global frame coords
        return (
            self._remap_to_global(phone_dets, origin),
            self._remap_to_global(food_dets,  origin),
            self._remap_to_global(book_dets,  origin),
        )

    def _parse_object_detections(self, frame: np.ndarray):
        """
        Deprecated full-frame wrapper kept for backward compatibility.
        New code should call _parse_object_detections_for_person() instead.
        """
        return self._parse_object_detections_for_person(
            crop=frame, origin=(0, 0), track_id=-1
        )

    # ── Pose detection ────────────────────────────────────────────────────────

    def _parse_pose_detections(self, frame: np.ndarray) -> List[Tuple]:
        tracks = []
        if self.yolo_model is None:
            return tracks
        try:
            for result in self.yolo_model.track(
                frame, persist=True, verbose=False,
                conf=0.3, iou=0.5, tracker='bytetrack.yaml'
            ):
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

    # ── Behavior evaluation ───────────────────────────────────────────────────

    def _run_behavior_evaluation(
        self, frame, person_tracks, phone_dets_by_tid,
        food_dets_by_tid, book_dets_by_tid,
        timestamp, fight_detected=False,
    ) -> List[DetectionResult]:
        """
        phone_dets_by_tid / food_dets_by_tid / book_dets_by_tid are dicts
        keyed by track_id, containing global-coord detections for that person.
        """
        active_tids = set()
        for tid, x1, y1, x2, y2, conf, kp in person_tracks:
            active_tids.add(tid)
            self.behavior_engine.update_person(tid, (x1, y1, x2, y2), kp, timestamp)

        self.behavior_engine.cleanup_stale(timestamp)
        # Cleanup phone detector and head pose state for departed tracks
        self.phone_detector.cleanup_stale(active_tids)
        # Cleanup head pose low_confidence_counters for departed tracks
        self.head_pose_detector.cleanup_stale(active_tids)

        results = []
        for tid in active_tids:
            with self.behavior_engine.lock:
                if tid not in self.behavior_engine.tracked_people:
                    continue
                person = self.behavior_engine.tracked_people[tid]

            if fight_detected:
                results.append(DetectionResult(
                    type='fighting', bbox=person.bbox, confidence=0.9,
                    color=(0, 0, 255), label='Fighting!',
                    is_alert=True, is_distracted=False, track_id=tid,
                ))
                continue

            phone_dets = phone_dets_by_tid.get(tid, [])
            food_dets  = food_dets_by_tid.get(tid, [])
            book_dets  = book_dets_by_tid.get(tid, [])

            head_pose = self.head_pose_detector.calculate_head_pose(person)
            is_hand_raised, hand_conf = self.hand_raise_detector.detect_hand_raise(person)

            raw_behavior   = ''
            raw_confidence = 0.0

            if is_hand_raised:
                raw_behavior   = 'hand_raised'
                raw_confidence = hand_conf
            else:
                is_phone, phone_conf = self.phone_detector.detect_phone_usage(
                    person=person,
                    phone_detections=phone_dets,
                    head_pose=head_pose,
                    book_detections=book_dets,
                    track_id=tid,
                    timestamp=timestamp,
                )
                if is_phone:
                    raw_behavior   = 'using_phone'
                    raw_confidence = phone_conf
                else:
                    is_eating, eating_conf = self.food_detector.detect_eating(
                        person, food_dets
                    )
                    if is_eating:
                        raw_behavior   = 'eating_food'
                        raw_confidence = eating_conf
                    else:
                        if head_pose == 'focused':
                            raw_behavior, raw_confidence = 'focused',     0.75
                        elif head_pose in ('head_down', 'looking_away'):
                            raw_behavior, raw_confidence = 'distracted',  0.70
                        elif head_pose == 'not_visible':
                            raw_behavior, raw_confidence = 'not_visible', 0.60
                        else:
                            raw_behavior, raw_confidence = 'distracted',  0.65

            final_behavior, confidence = self.behavior_engine.evaluate_final_behavior(
                person, raw_behavior, raw_confidence
            )
            is_alert      = final_behavior in ALERT_POSES
            is_distracted = final_behavior in DISTRACTED_POSES

            results.append(DetectionResult(
                type=final_behavior,
                bbox=person.bbox,
                confidence=confidence,
                color=COLOR_MAP.get(final_behavior, COLOR_MAP['not_visible']),
                label=LABEL_MAP.get(final_behavior, final_behavior),
                is_alert=is_alert,
                is_distracted=is_distracted,
                track_id=tid,
            ))

        return results

    # ── Core frame processing ─────────────────────────────────────────────────

    def _process_single_frame(self, frame: np.ndarray, timestamp: float):
        try:
            person_tracks = self._parse_pose_detections(frame)

            # Per-person RoI object detection — accumulate into per-tid dicts
            phone_dets_by_tid: Dict[int, List] = {}
            food_dets_by_tid:  Dict[int, List] = {}
            book_dets_by_tid:  Dict[int, List] = {}
            all_phone_dets: List[Tuple] = []
            all_food_dets:  List[Tuple] = []

            for tid, x1, y1, x2, y2, _conf, _kp in person_tracks:
                crop, origin = self._extract_person_crop(frame, x1, y1, x2, y2)
                if crop is None:
                    phone_dets_by_tid[tid] = []
                    food_dets_by_tid[tid]  = []
                    book_dets_by_tid[tid]  = []
                    continue

                p_dets, f_dets, b_dets = self._parse_object_detections_for_person(
                    crop, origin, tid
                )
                phone_dets_by_tid[tid] = p_dets
                food_dets_by_tid[tid]  = f_dets
                book_dets_by_tid[tid]  = b_dets
                all_phone_dets.extend(p_dets)
                all_food_dets.extend(f_dets)

            # Fight detection (runs on full frame)
            fight_detected = False
            if self.fight_detector is not None:
                self.fight_detector.add_frame(frame)
                fight_result   = self.fight_detector.predict()
                fight_detected = (
                    fight_result[0]
                    if isinstance(fight_result, tuple)
                    else bool(fight_result)
                )
                if fight_detected:
                    print('[ALERT] Fight detected!')

            final_results = self._run_behavior_evaluation(
                frame, person_tracks,
                phone_dets_by_tid, food_dets_by_tid, book_dets_by_tid,
                timestamp, fight_detected,
            )

            with self.lock:
                self.phone_detections = all_phone_dets
                self.food_detections  = all_food_dets
                self.person_tracks    = [
                    (t[0], t[1], t[2], t[3], t[4], t[5])
                    for t in person_tracks
                ]
                self.result_buffer.append((timestamp, frame.copy(), final_results))

        except Exception as e:
            print(f'[ERROR] Frame processing: {e}')

    # ── Stream lifecycle ──────────────────────────────────────────────────────

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

    def start(self, camera_url: str):
        if self.running:
            return
        self.running = True
        self.stop_event.clear()
        threading.Thread(
            target=self._capture_frames, args=(camera_url,), daemon=True
        ).start()
        threading.Thread(target=self._process_frames, daemon=True).start()
        print('[OK] Stream processor started')

    def stop(self):
        self.running = False
        self.stop_event.set()
        print('[OK] Stream processor stopped')

    def get_latest_result(
        self,
    ) -> Tuple[Optional[float], Optional[np.ndarray], List[DetectionResult]]:
        with self.lock:
            return self.result_buffer[-1] if self.result_buffer else (None, None, [])


# ── ClassroomBehaviorDetector (backward-compatible public API) ─────────────────
class ClassroomBehaviorDetector:
    def __init__(
        self,
        camera_url,
        camera_id,
        server_url='http://localhost:8000',
        alert_cooldown=120,
        whatsapp_admin=None,
    ):
        self.camera_url     = camera_url
        self.camera_id      = camera_id
        self.server_url     = server_url.rstrip('/')
        self.alert_cooldown = alert_cooldown
        self.whatsapp_admin = whatsapp_admin or os.environ.get('ADMIN_WHATSAPP', '')
        self._face_recognizer = None
        self.known_students   = []
        self.last_alert_time: dict = defaultdict(float)
        self.running          = False
        self.thread           = None
        self.processor        = ProductionStreamProcessor(process_fps=10)
        self._api_key         = os.environ.get('DETECTION_API_KEY', '').strip()

    # ── Convenience properties (used by views.py) ─────────────────────────────

    @property
    def behavior_engine(self):
        return self.processor.behavior_engine

    @property
    def yolo_model(self):
        return self.processor.yolo_model

    @property
    def object_model(self):
        return self.processor.object_model

    # ── Face recognition ──────────────────────────────────────────────────────

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
        url     = f'{self.server_url}/camera-attendance/api/students/encodings/'
        headers = {'X-Detection-API-Key': self._api_key} if self._api_key else {}
        try:
            r = _req.get(url, timeout=5, verify=_http_verify_ssl(), headers=headers)
            r.raise_for_status()
            self.known_students = r.json()
            print(f'[OK] {len(self.known_students)} student encodings loaded via HTTP')
        except Exception as e:
            print(f'[WARN] Could not load students: {e}')
            self.known_students = []

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
                    d = fr.face_distance(
                        [np.array(json.loads(s['encoding']))], det
                    )[0]
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

    # ── Public detect() — used by views.py live stream ────────────────────────

    def detect(self, frame) -> List[Dict]:
        """
        Single-frame synchronous detection.
        Runs the full per-person RoI pipeline and returns a list of dicts.
        Also feeds the fight detector so it accumulates frames for its 3D CNN.
        """
        if self.processor.yolo_model is None:
            return []
        try:
            timestamp     = time.time()
            person_tracks = self.processor._parse_pose_detections(frame)

            phone_dets_by_tid: Dict[int, List] = {}
            food_dets_by_tid:  Dict[int, List] = {}
            book_dets_by_tid:  Dict[int, List] = {}

            for tid, x1, y1, x2, y2, _conf, _kp in person_tracks:
                crop, origin = self.processor._extract_person_crop(
                    frame, x1, y1, x2, y2
                )
                if crop is None:
                    phone_dets_by_tid[tid] = []
                    food_dets_by_tid[tid]  = []
                    book_dets_by_tid[tid]  = []
                    continue
                p, f, b = self.processor._parse_object_detections_for_person(
                    crop, origin, tid
                )
                phone_dets_by_tid[tid] = p
                food_dets_by_tid[tid]  = f
                book_dets_by_tid[tid]  = b

            # Fight detection — feed frame into 3D CNN buffer and run inference.
            # This was previously only called inside _process_single_frame (the
            # background-loop path), so the live-stream path via detect() never
            # triggered it. Fixed here.
            fight_detected = False
            if self.processor.fight_detector is not None:
                self.processor.fight_detector.add_frame(frame)
                fight_result   = self.processor.fight_detector.predict()
                fight_detected = (
                    fight_result[0]
                    if isinstance(fight_result, tuple)
                    else bool(fight_result)
                )
                if fight_detected:
                    conf = fight_result[1] if isinstance(fight_result, tuple) else 0.0
                    print(f'[ALERT] Fight detected in live stream (conf={conf:.2f})')

            det_objs = self.processor._run_behavior_evaluation(
                frame, person_tracks,
                phone_dets_by_tid, food_dets_by_tid, book_dets_by_tid,
                timestamp, fight_detected,
            )
            return [
                {
                    'type': d.type, 'bbox': d.bbox, 'confidence': d.confidence,
                    'color': d.color, 'label': d.label, 'is_alert': d.is_alert,
                    'is_distracted': d.is_distracted, 'track_id': d.track_id,
                }
                for d in det_objs
            ]
        except Exception as e:
            print(f'[ERROR] detect(): {e}')
            return []

    def _detect_behaviors(self, frame):
        return self.detect(frame)

    # ── Incident reporting ────────────────────────────────────────────────────

    def _draw_detections(self, frame, detections):
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color, label = det['color'], det['label']
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"{label} ({det['confidence']:.2f})"
            tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            bg_y1 = max(0, y1 - th - 4)
            cv2.rectangle(out, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                out, text, (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
        return out

    def _report_incident(
        self, detection, frame, student_id,
        student_name, roll_no, all_detections=None,
    ):
        import requests as _req
        try:
            annotated = (
                self._draw_detections(frame, all_detections)
                if all_detections else frame
            )
            _, buf   = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            snap_b64 = base64.b64encode(buf).decode()
            tag = f"{student_name} ({roll_no})" if student_id else 'Unknown'
            resp = _req.post(
                f'{self.server_url}/classroom/api/incidents/report/',
                json={
                    'student_id':    student_id,
                    'camera_id':     self.camera_id,
                    'incident_type': detection['type'],
                    'confidence':    round(detection['confidence'], 3),
                    'snapshot':      snap_b64,
                    'student_name':  student_name,
                    'roll_no':       roll_no,
                    'description':   f"ALERT {detection['label']} — {tag}",
                },
                headers=(
                    {'X-Detection-API-Key': self._api_key}
                    if self._api_key else {}
                ),
                timeout=10,
                verify=_http_verify_ssl(),
            )
            print(f"[INCIDENT] {detection['label']} | {tag} | {resp.status_code}")
        except Exception as e:
            print(f'[ERROR] report_incident: {e}')

    # ── Detection loop ────────────────────────────────────────────────────────

    def _detection_loop(self):
        self.processor.start(self.camera_url)
        frame_count, last_ts = 0, 0.0
        while self.running:
            ts, frame, det_objs = self.processor.get_latest_result()
            if frame is None or ts <= last_ts:
                time.sleep(0.01)
                continue
            last_ts = ts
            detections = [
                {
                    'type': d.type, 'bbox': d.bbox, 'confidence': d.confidence,
                    'color': d.color, 'label': d.label, 'is_alert': d.is_alert,
                    'is_distracted': d.is_distracted, 'track_id': d.track_id,
                }
                for d in det_objs
            ]
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
                self._report_incident(
                    det, frame, sid, name, roll, all_detections=detections
                )
                self.last_alert_time[key] = now
            if frame_count % 300 == 0:
                print(f'[INFO] Frame {frame_count} | {len(detections)} detections')
        self.processor.stop()

    def start(self):
        if self.running:
            return
        self._init_face_recognition()
        self.running = True
        self.thread  = threading.Thread(
            target=self._detection_loop, daemon=True
        )
        self.thread.start()
        print('[OK] Behavior detection started')

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.processor.stop()
        print('[OK] Behavior detection stopped')
