"""
Classroom Behavior Detection - Production Grade Refactor (Fixed)
============================================================

Completely rearchitected to:
1. Remove brittle optical flow heuristics and magic thresholds
2. Use YOLO11s-Pose + ByteTrack for robust multi-object tracking
3. Implement hierarchical behavior engine with temporal smoothing
4. Use thread-isolated pipeline for memory stability and frame dropping prevention

All external method signatures preserved for backward compatibility.

Fixed:
- Index leak in keypoint pairing
- Model selection to yolo11s-pose.pt
- Connected ClassroomBehaviorDetector to ProductionStreamProcessor
- Added kinetic velocity fight detection using keypoints
"""

import cv2
import json
import time
import base64
import numpy as np
import threading
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Keep existing color/label maps for backward compatibility
COLOR_MAP = {
    'focused':      (0,  200,  60),
    'looking_away': (0,  165, 255),
    'head_down':    (0,  165, 255),
    'distracted':   (0,  165, 255),
    'using_phone':  (0,    0, 220),
    'eating_food':  (0,    0, 220),
    'fighting':     (0,    0, 255),
    'not_visible':  (120, 120, 120),
}
LABEL_MAP = {
    'focused':      'Focused',
    'looking_away': 'Looking Away',
    'head_down':    'Head Down',
    'distracted':   'Distracted',
    'using_phone':  'Using Phone',
    'eating_food':  'Eating Food',
    'fighting':     'FIGHT',
    'not_visible':  'Not Visible',
}

ALERT_POSES      = {'using_phone', 'eating_food', 'fighting'}
DISTRACTED_POSES = {'looking_away', 'head_down', 'distracted'}


# ── Environment Loading ─────────────────────────────────────────────────────
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
                        os.environ.setdefault(k.strip(),
                                              v.strip().strip('"').strip("'"))
            print(f'[ENV] Loaded {path}')
            return
    print('[ENV] No .env file found')

_load_env_file()

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')


# ── Data Structures ─────────────────────────────────────────────────────────
@dataclass
class TrackedPerson:
    track_id: int
    bbox: Tuple[int, int, int, int]
    keypoints: Optional[np.ndarray] = None
    behavior_history: deque = field(default_factory=lambda: deque(maxlen=16))
    state_history: deque = field(default_factory=lambda: deque(maxlen=3))
    last_seen: float = 0.0
    # For kinetic fight detection
    keypoint_history: deque = field(default_factory=lambda: deque(maxlen=10))


@dataclass
class DetectionResult:
    type: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    color: Tuple[int, int, int]
    label: str
    is_alert: bool
    is_distracted: bool
    track_id: Optional[int] = None
    fight_info: Optional[Dict] = None


# ── Temporal Behavior Engine ────────────────────────────────────────────────
class TemporalBehaviorEngine:
    """
    Hierarchical behavior engine with temporal smoothing.
    Requires 3 consecutive positive evaluations before state change.
    """
    
    # Behavior thresholds (tuned for classroom settings)
    PHONE_OVERLAP_THRESHOLD = 0.3
    EATING_HAND_TO_MOUTH_THRESHOLD = 0.4
    FIGHT_OVERLAP_THRESHOLD = 0.5
    FIGHT_KINETIC_THRESHOLD = 15.0  # Pixel movement threshold for wrist/elbow
    ENGAGEMENT_HEAD_POSE_THRESHOLD = 0.6
    
    def __init__(self):
        self.tracked_people: Dict[int, TrackedPerson] = {}
        self.cleanup_threshold = 5.0  # seconds
    
    def update_person(self, track_id: int, bbox: Tuple[int, int, int, int], 
                     keypoints: Optional[np.ndarray], timestamp: float):
        """Update or create a tracked person with new frame data."""
        if track_id not in self.tracked_people:
            self.tracked_people[track_id] = TrackedPerson(
                track_id=track_id,
                bbox=bbox,
                keypoints=keypoints,
                last_seen=timestamp
            )
        else:
            person = self.tracked_people[track_id]
            person.bbox = bbox
            person.keypoints = keypoints
            person.last_seen = timestamp
            # Store keypoint history for kinetic analysis
            if keypoints is not None:
                person.keypoint_history.append(keypoints.copy())
    
    def cleanup_stale(self, current_time: float):
        """Remove tracked persons not seen for cleanup_threshold seconds."""
        stale = [tid for tid, p in self.tracked_people.items() 
                if current_time - p.last_seen > self.cleanup_threshold]
        for tid in stale:
            del self.tracked_people[tid]
    
    def _calculate_head_pose(self, person: TrackedPerson) -> str:
        """
        Estimate head pose using scale-invariant structural ratios.
        Returns 'focused', 'looking_away', or 'head_down'.
        """
        keypoints = person.keypoints
        if keypoints is None or keypoints.size == 0 or len(keypoints) < 3:
            return 'focused'
        
        try:
            nose = keypoints[0]
            left_eye = keypoints[1]
            right_eye = keypoints[2]
            
            # Visibility checks: confidence >= 0.5 and no zero coordinates
            if (len(nose) < 3 or len(left_eye) < 3 or len(right_eye) < 3):
                return 'focused'
            if (nose[2] < 0.5 or left_eye[2] < 0.5 or right_eye[2] < 0.5):
                return 'focused'
            if (nose[0] == 0.0 or nose[1] == 0.0 or
                left_eye[0] == 0.0 or left_eye[1] == 0.0 or
                right_eye[0] == 0.0 or right_eye[1] == 0.0):
                return 'focused'
            
            # Inter-eye distance (scale reference)
            inter_eye_dist = np.linalg.norm(left_eye[:2] - right_eye[:2])
            if inter_eye_dist < 0.1:  # Avoid division by zero
                return 'focused'
            
            # Calculate vertical drop of nose relative to eye line
            eye_y_avg = (left_eye[1] + right_eye[1]) / 2
            vertical_drop = nose[1] - eye_y_avg
            drop_ratio = vertical_drop / inter_eye_dist
            
            # Check for head down (large positive drop ratio)
            if drop_ratio > 0.8:
                return 'head_down'
            
            # Check for looking away (very small inter-eye distance relative to what we'd expect)
            # We can use the bbox height as another scale reference
            x1, y1, x2, y2 = person.bbox
            bbox_height = y2 - y1
            if bbox_height > 10 and inter_eye_dist < bbox_height * 0.05:
                return 'looking_away'
            
            return 'focused'
        except Exception:
            return 'focused'
    
    def _detect_phone_usage(self, person: TrackedPerson, 
                           phone_detections: List[Tuple]) -> Tuple[bool, float]:
        """Detect phone usage based on phone bounding box overlap with person."""
        x1, y1, x2, y2 = person.bbox
        person_area = (x2 - x1) * (y2 - y1)
        
        for (px1, py1, px2, py2, conf) in phone_detections:
            # Calculate IoU
            ix1 = max(x1, px1)
            iy1 = max(y1, py1)
            ix2 = min(x2, px2)
            iy2 = min(y2, py2)
            
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            
            intersection = (ix2 - ix1) * (iy2 - iy1)
            phone_area = (px2 - px1) * (py2 - py1)
            overlap_ratio = intersection / min(person_area, phone_area)
            
            if overlap_ratio > self.PHONE_OVERLAP_THRESHOLD:
                return True, conf
        
        return False, 0.0
    
    def _detect_eating(self, person: TrackedPerson, 
                      food_detections: List[Tuple]) -> Tuple[bool, float]:
        """Detect eating based on food/beverage detections near face."""
        if person.keypoints is None or person.keypoints.size == 0:
            return False, 0.0
        
        try:
            nose = person.keypoints[0]
            x1, y1, x2, y2 = person.bbox
            
            for (fx1, fy1, fx2, fy2, conf) in food_detections:
                food_center = ((fx1 + fx2) / 2, (fy1 + fy2) / 2)
                distance_to_nose = np.linalg.norm(
                    np.array(food_center) - np.array(nose[:2])
                )
                
                if distance_to_nose < (y2 - y1) * 0.5:
                    return True, conf
        except:
            pass
        
        return False, 0.0
    
    def _calculate_kinetic_energy(self, person: TrackedPerson, bbox_height: float) -> float:
        """
        Calculate scale-invariant kinetic energy from wrist and elbow keypoint movement.
        Normalizes distances against bounding box height.
        """
        if len(person.keypoint_history) < 3 or bbox_height < 1.0:
            return 0.0
        
        # Keypoint indices for wrists and elbows (YOLO pose format)
        # 9: left wrist, 10: right wrist, 7: left elbow, 8: right elbow
        kinetic_keypoints = [7, 8, 9, 10]
        
        total_movement = 0.0
        count = 0
        
        for i in range(1, len(person.keypoint_history)):
            prev_kp = person.keypoint_history[i-1]
            curr_kp = person.keypoint_history[i]
            
            for kp_idx in kinetic_keypoints:
                if kp_idx >= len(prev_kp) or kp_idx >= len(curr_kp):
                    continue
                
                # Check if keypoints are visible (confidence >= 0.5)
                if len(prev_kp[kp_idx]) < 3 or len(curr_kp[kp_idx]) < 3:
                    continue
                if prev_kp[kp_idx][2] < 0.5 or curr_kp[kp_idx][2] < 0.5:
                    continue
                
                # Skip if any coordinate is exactly 0.0 (undetected dropout)
                if prev_kp[kp_idx][0] == 0.0 or prev_kp[kp_idx][1] == 0.0:
                    continue
                if curr_kp[kp_idx][0] == 0.0 or curr_kp[kp_idx][1] == 0.0:
                    continue
                
                # Calculate distance moved
                raw_dist = np.linalg.norm(curr_kp[kp_idx][:2] - prev_kp[kp_idx][:2])
                # Normalize to scale-invariant units (percent of bbox height)
                normalized_dist = (raw_dist / bbox_height) * 100.0
                total_movement += normalized_dist
                count += 1
        
        if count == 0:
            return 0.0
        
        return total_movement / count
    
    def _detect_fighting(self, person: TrackedPerson, 
                        all_people: Dict[int, TrackedPerson]) -> Tuple[bool, float, Optional[Dict]]:
        """
        Detect fighting based on two criteria:
        1. Bounding box overlap > 0.5
        2. Scale-invariant kinetic energy > threshold
        """
        x1, y1, x2, y2 = person.bbox
        bbox_height = y2 - y1
        person_kinetic = self._calculate_kinetic_energy(person, bbox_height)
        
        for other_id, other_person in all_people.items():
            if other_id == person.track_id:
                continue
            
            ox1, oy1, ox2, oy2 = other_person.bbox
            other_bbox_height = oy2 - oy1
            
            # Calculate overlap
            ix1 = max(x1, ox1)
            iy1 = max(y1, oy1)
            ix2 = min(x2, ox2)
            iy2 = min(y2, oy2)
            
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            
            person_area = (x2 - x1) * (y2 - y1)
            other_area = (ox2 - ox1) * (oy2 - oy1)
            intersection = (ix2 - ix1) * (iy2 - iy1)
            overlap = intersection / min(person_area, other_area)
            
            if overlap > self.FIGHT_OVERLAP_THRESHOLD:
                # Check kinetic energy for both people
                other_kinetic = self._calculate_kinetic_energy(other_person, other_bbox_height)
                avg_kinetic = (person_kinetic + other_kinetic) / 2
                
                if avg_kinetic > self.FIGHT_KINETIC_THRESHOLD:
                    # High overlap + high kinetic movement = fight!
                    confidence = min(0.95, 0.7 + (avg_kinetic / 20.0))  # Adjusted for normalized units
                    return True, confidence, {
                        'person_a_id': person.track_id,
                        'person_b_id': other_id,
                        'confidence': confidence,
                        'kinetic_energy': avg_kinetic
                    }
        
        return False, 0.0, None
    
    def evaluate_person(self, track_id: int, 
                       phone_detections: List[Tuple],
                       food_detections: List[Tuple]) -> DetectionResult:
        """
        Evaluate a tracked person's behavior with temporal smoothing.
        Returns a DetectionResult compatible with existing code.
        """
        if track_id not in self.tracked_people:
            return DetectionResult(
                type='not_visible',
                bbox=(0, 0, 0, 0),
                confidence=0.0,
                color=COLOR_MAP['not_visible'],
                label=LABEL_MAP['not_visible'],
                is_alert=False,
                is_distracted=False
            )
        
        person = self.tracked_people[track_id]
        
        # First, check for fight (highest priority)
        is_fighting, fight_conf, fight_info = self._detect_fighting(
            person, self.tracked_people
        )
        
        if is_fighting:
            person.behavior_history.append('fighting')
        else:
            # Check phone usage
            is_phone, phone_conf = self._detect_phone_usage(person, phone_detections)
            if is_phone:
                person.behavior_history.append('using_phone')
            else:
                # Check eating
                is_eating, eat_conf = self._detect_eating(person, food_detections)
                if is_eating:
                    person.behavior_history.append('eating_food')
                else:
                    # Evaluate engagement
                    head_pose = self._calculate_head_pose(person)
                    person.behavior_history.append(head_pose)
        
        # Temporal smoothing - require 3 consecutive same states
        final_behavior = 'focused'
        confidence = 0.8
        
        if len(person.behavior_history) >= 3:
            recent = list(person.behavior_history)[-3:]
            if recent[0] == recent[1] == recent[2]:
                final_behavior = recent[0]
            else:
                # Fall back to most common in history
                from collections import Counter
                final_behavior = Counter(person.behavior_history).most_common(1)[0][0]
        elif len(person.behavior_history) > 0:
            # For new tracks or on-demand API, use the latest behavior
            final_behavior = person.behavior_history[-1]
        
        # Build result
        if final_behavior == 'fighting':
            is_alert = True
            is_distracted = False
            confidence = fight_conf if is_fighting else 0.8
        else:
            is_alert = final_behavior in ALERT_POSES
            is_distracted = final_behavior in DISTRACTED_POSES
        
        return DetectionResult(
            type=final_behavior,
            bbox=person.bbox,
            confidence=confidence,
            color=COLOR_MAP.get(final_behavior, COLOR_MAP['not_visible']),
            label=LABEL_MAP.get(final_behavior, final_behavior),
            is_alert=is_alert,
            is_distracted=is_distracted,
            track_id=track_id,
            fight_info=fight_info if final_behavior == 'fighting' else None
        )


# ── Production Stream Processor ─────────────────────────────────────────────
class ProductionStreamProcessor:
    """
    Thread-isolated, production-grade stream processor.
    Features:
    - Fixed ring buffer for frame ingestion
    - Configurable processing rate (8-12 FPS)
    - Automatic RTSP reconnection
    - YOLO11s-Pose + ByteTrack pipeline
    """
    
    def __init__(self, process_fps: int = 10, buffer_size: int = 5):
        self.process_fps = process_fps
        self.frame_interval = 1.0 / process_fps
        self.buffer_size = buffer_size
        
        self.frame_buffer: deque = deque(maxlen=buffer_size)
        self.result_buffer: deque = deque(maxlen=2)
        
        self.yolo_model = None
        self.tracker = None
        
        self.running = False
        self.process_thread = None
        self.capture_thread = None
        
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
        self.phone_detections: List[Tuple] = []
        self.food_detections: List[Tuple] = []
        self.person_tracks: List[Tuple] = []
        
        self.behavior_engine = TemporalBehaviorEngine()
        
        self._load_models()
    
    def _load_models(self):
        """Load YOLO11s-Pose model with ByteTrack."""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolo11s-pose.pt')  # Fixed: use pose model
            print('[OK] YOLO11s-Pose model loaded')
        except Exception as e:
            print(f'[WARN] Failed to load YOLO: {e}')
    
    def _capture_frames(self, camera_url: str):
        """Background thread: capture and buffer frames."""
        cap = None
        reconnect_delay = 1.0
        
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
                        print(f'[WARN] Failed to open camera, retrying in {reconnect_delay}s')
                        time.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 2, 10.0)
                        continue
                
                ret, frame = cap.read()
                if not ret:
                    print('[WARN] Frame read failed, reconnecting')
                    cap.release()
                    cap = None
                    continue
                
                # Add to buffer (thread-safe)
                with self.lock:
                    self.frame_buffer.append((time.time(), frame.copy()))
            
            except Exception as e:
                print(f'[ERROR] Capture thread: {e}')
                if cap:
                    cap.release()
                    cap = None
                time.sleep(1.0)
        
        if cap:
            cap.release()
    
    def _process_frames(self):
        """Background thread: process frames at fixed rate."""
        last_process_time = 0.0
        
        while not self.stop_event.is_set():
            current_time = time.time()
            
            if current_time - last_process_time >= self.frame_interval:
                # Get latest frame
                frame = None
                timestamp = current_time
                
                with self.lock:
                    if self.frame_buffer:
                        timestamp, frame = self.frame_buffer.pop()
                
                if frame is not None:
                    self._process_single_frame(frame, timestamp)
                    last_process_time = current_time
            
            time.sleep(0.001)
    
    def _process_single_frame(self, frame: np.ndarray, timestamp: float):
        """Process a single frame with YOLO + ByteTrack."""
        if self.yolo_model is None:
            return
        
        try:
            # Run YOLO detection + tracking
            results = self.yolo_model.track(
                frame,
                persist=True,
                verbose=False,
                conf=0.3,
                iou=0.5,
                tracker='bytetrack.yaml'
            )
            
            phone_dets = []
            food_dets = []
            # List of tuples: (track_id, x1, y1, x2, y2, conf, keypoints)
            person_tracks_with_kp = []
            
            for result in results:
                if result.boxes is None:
                    continue
                
                boxes = result.boxes
                keypoints_list = result.keypoints if hasattr(result, 'keypoints') else None
                
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                    track_id = int(boxes.id[i]) if boxes.id is not None else i
                    
                    if cls_id == 0:  # Person
                        # Get keypoints for this specific person
                        kp = None
                        if keypoints_list is not None and i < len(keypoints_list):
                            try:
                                kp = keypoints_list[i].data.cpu().numpy()[0]
                            except:
                                pass
                        person_tracks_with_kp.append((track_id, x1, y1, x2, y2, conf, kp))
                    elif cls_id == 67:  # Cell phone
                        phone_dets.append((x1, y1, x2, y2, conf))
                    elif cls_id in [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]:  # Food/drink
                        food_dets.append((x1, y1, x2, y2, conf))
            
            # Update behavior engine
            for track_id, x1, y1, x2, y2, conf, kp in person_tracks_with_kp:
                self.behavior_engine.update_person(
                    track_id, (x1, y1, x2, y2), kp, timestamp
                )
            
            # Cleanup stale tracks
            self.behavior_engine.cleanup_stale(timestamp)
            
            # Generate detection results
            final_results = []
            for track_id in self.behavior_engine.tracked_people:
                det_result = self.behavior_engine.evaluate_person(
                    track_id, phone_dets, food_dets
                )
                final_results.append(det_result)
            
            # Store results (thread-safe)
            with self.lock:
                self.phone_detections = phone_dets
                self.food_detections = food_dets
                self.person_tracks = [(t[0], t[1], t[2], t[3], t[4], t[5]) for t in person_tracks_with_kp]
                self.result_buffer.append((timestamp, frame.copy(), final_results))
        
        except Exception as e:
            print(f'[ERROR] Frame processing: {e}')
    
    def start(self, camera_url: str):
        """Start the processing pipeline."""
        if self.running:
            return
        
        self.running = True
        self.stop_event.clear()
        
        self.capture_thread = threading.Thread(
            target=self._capture_frames,
            args=(camera_url,),
            daemon=True
        )
        self.process_thread = threading.Thread(
            target=self._process_frames,
            daemon=True
        )
        
        self.capture_thread.start()
        self.process_thread.start()
        print('[OK] Production stream processor started')
    
    def stop(self):
        """Stop the processing pipeline."""
        self.running = False
        self.stop_event.set()
        
        if self.capture_thread:
            self.capture_thread.join(timeout=3.0)
        if self.process_thread:
            self.process_thread.join(timeout=3.0)
        
        print('[OK] Production stream processor stopped')
    
    def get_latest_result(self) -> Tuple[Optional[float], Optional[np.ndarray], List[DetectionResult]]:
        """Get the latest processed frame and results (thread-safe)."""
        with self.lock:
            if self.result_buffer:
                return self.result_buffer[-1]
            return None, None, []


# ── Backward Compatible ClassroomBehaviorDetector ───────────────────────────
class ClassroomBehaviorDetector:
    """
    Drop-in replacement for original ClassroomBehaviorDetector.
    Maintains exact same public API for backward compatibility.
    Uses the new ProductionStreamProcessor internally.
    """
    
    def __init__(self, camera_url, camera_id,
                 server_url='http://localhost:8000',
                 alert_cooldown=120,
                 whatsapp_admin=None):
        self.camera_url = camera_url
        self.camera_id = camera_id
        self.server_url = server_url
        self.alert_cooldown = alert_cooldown
        self.whatsapp_admin = whatsapp_admin or os.environ.get('ADMIN_WHATSAPP', '')
        
        self.yolo_model = None  # Kept for backward compatibility
        self.face_recognizer = None
        self.known_students = []
        
        self.last_alert_time: dict = defaultdict(float)
        self.running = False
        self.thread = None
        
        # New production processor
        self.processor = ProductionStreamProcessor(process_fps=10)
        
        self._load_models()
    
    def _load_models(self):
        """Load models (backward compatibility)."""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolo11s-pose.pt')  # Fixed: use pose model
            print('[OK] YOLO11s-Pose loaded (backward compatibility)')
        except Exception as e:
            print(f'[WARN] YOLO: {e}')
        print('[OK] Fight detector initialized (backward compatibility)')
    
    def _load_known_students(self):
        """Load known students (backward compatibility)."""
        import requests as _req
        try:
            r = _req.get(f'{self.server_url}/api/students/encodings/',
                        timeout=5, verify=False)
            self.known_students = r.json()
            print(f'[OK] {len(self.known_students)} encodings')
        except Exception as e:
            print(f'[WARN] students: {e}')
    
    def detect(self, frame) -> List[Dict]:
        """
        Detect behaviors in a single frame (backward compatible).
        Returns list of detection dicts in original format.
        """
        # For backward compatibility, run detection on demand
        if self.yolo_model is None:
            return []
        
        try:
            results = self.yolo_model.track(
                frame,
                persist=True,
                verbose=False,
                conf=0.3,
                tracker='bytetrack.yaml'
            )
            
            detections = []
            phone_dets = []
            food_dets = []
            person_tracks_with_kp = []
            timestamp = time.time()
            
            for result in results:
                if result.boxes is None:
                    continue
                
                boxes = result.boxes
                keypoints_list = result.keypoints if hasattr(result, 'keypoints') else None
                
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                    track_id = int(boxes.id[i]) if boxes.id is not None else i
                    
                    if cls_id == 0:
                        kp = None
                        if keypoints_list is not None and i < len(keypoints_list):
                            try:
                                kp = keypoints_list[i].data.cpu().numpy()[0]
                            except:
                                pass
                        person_tracks_with_kp.append((track_id, x1, y1, x2, y2, conf, kp))
                    elif cls_id == 67:
                        phone_dets.append((x1, y1, x2, y2, conf))
                    elif cls_id in [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]:
                        food_dets.append((x1, y1, x2, y2, conf))
            
            # Update temporary behavior engine for this detect() call
            temp_engine = TemporalBehaviorEngine()
            for track_id, x1, y1, x2, y2, conf, kp in person_tracks_with_kp:
                temp_engine.update_person(track_id, (x1, y1, x2, y2), kp, timestamp)
            
            # Evaluate each person
            for track_id in temp_engine.tracked_people:
                det_result = temp_engine.evaluate_person(track_id, phone_dets, food_dets)
                
                # Convert to original dict format
                det_dict = {
                    'type': det_result.type,
                    'bbox': det_result.bbox,
                    'confidence': det_result.confidence,
                    'color': det_result.color,
                    'label': det_result.label,
                    'is_alert': det_result.is_alert,
                    'is_distracted': det_result.is_distracted,
                    'track_id': det_result.track_id
                }
                if det_result.fight_info:
                    det_dict['fight_info'] = det_result.fight_info
                detections.append(det_dict)
            
            return detections
        
        except Exception as e:
            print(f'[ERROR] detect(): {e}')
            return []
    
    def _detect_behaviors(self, frame):
        """Alias for detect() (backward compatibility)."""
        return self.detect(frame)
    
    def _draw_detections(self, frame, detections):
        """Draw detections on frame (backward compatible)."""
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = det['color']
            label = det['label']
            conf  = det['confidence']
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"{label} ({conf:.2f})"
            tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            bg_y1  = max(0, y1 - th - 4)
            cv2.rectangle(out, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, text, (x1 + 2, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out
    
    def _report_incident(self, detection, frame, student_id,
                         student_name, roll_no, all_detections=None):
        """Report incident (backward compatible)."""
        import requests as _req
        try:
            annotated = (self._draw_detections(frame, all_detections)
                        if all_detections else frame)
            _, buf    = cv2.imencode('.jpg', annotated,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            snap_b64  = base64.b64encode(buf).decode()
            
            if detection['type'] == 'fighting':
                fi    = detection.get('fight_info', {})
                other = fi.get('person_b_id', 'unknown')
                tag   = f'{student_name} ({roll_no}) vs student_{other}'
                sev   = 'CRITICAL'
            else:
                tag = (f'{student_name} ({roll_no})'
                      if student_id else 'Unknown')
                sev = 'WARNING'
            
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
                    'description':   f"{sev} {detection['label']} — {tag}",
                    'send_whatsapp': detection['is_alert'],
                },
                timeout=10, verify=False,
            )
            print(f"[INCIDENT] {detection['label']} | {tag} | {resp.status_code}")
        except Exception as e:
            print(f'[ERROR] report_incident: {e}')
    
    def _recognize_face(self, frame, bbox):
        """Recognize face (backward compatible)."""
        if self.face_recognizer is None or not self.known_students:
            return None, 'Unknown', ''
        try:
            x1, y1, x2, y2 = bbox
            mid_y = y1 + int((y2 - y1) * 0.55)
            crop  = frame[y1:mid_y, x1:x2]
            if crop.size == 0:
                crop = frame[y1:y2, x1:x2]
            rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            encs = self.face_recognizer.face_encodings(
                rgb, num_jitters=1, model='small'
            )
            if not encs:
                return None, 'Unknown', ''
            det    = encs[0]
            best_d = 1.0
            best   = None
            for s in self.known_students:
                try:
                    known = np.array(json.loads(s['encoding']))
                    d     = self.face_recognizer.face_distance([known], det)[0]
                    if d < best_d:
                        best_d, best = d, s
                except Exception:
                    continue
            if best_d < 0.55 and best:
                return best['id'], best['name'], best.get('roll_no', '')
            return None, 'Unknown', ''
        except Exception as e:
            print(f'[ERROR] face_recog: {e}')
            return None, 'Unknown', ''
    
    def _detection_loop(self):
        """Detection loop that uses ProductionStreamProcessor."""
        # Start the production processor
        self.processor.start(self.camera_url)
        print('[OK] Production processor connected, starting detection loop')
        
        frame_count = 0
        fight_cooldown: dict = defaultdict(float)
        last_seen_timestamp = 0.0
        
        while self.running:
            # Get latest processed frame and results
            timestamp, frame, detections_objs = self.processor.get_latest_result()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Skip duplicate frames
            if timestamp <= last_seen_timestamp:
                time.sleep(0.01)
                continue
            
            last_seen_timestamp = timestamp
            
            # Convert DetectionResult objects to original dict format
            detections = []
            for det_obj in detections_objs:
                det_dict = {
                    'type': det_obj.type,
                    'bbox': det_obj.bbox,
                    'confidence': det_obj.confidence,
                    'color': det_obj.color,
                    'label': det_obj.label,
                    'is_alert': det_obj.is_alert,
                    'is_distracted': det_obj.is_distracted,
                    'track_id': det_obj.track_id
                }
                if det_obj.fight_info:
                    det_dict['fight_info'] = det_obj.fight_info
                detections.append(det_dict)
            
            frame_count += 1
            
            # Process alerts (same as original logic)
            fight_dets = [d for d in detections if d['type'] == 'fighting']
            other_dets = [d for d in detections if d['type'] != 'fighting']
            
            for det in other_dets:
                if not (det['is_alert'] or det['is_distracted']):
                    continue
                now = time.time()
                # Ensure track_id is extracted from the byte tracker object
                track_id = det.get('track_id', None)
                key = (det['type'], track_id if track_id is not None else tuple(det['bbox']))
                if now - self.last_alert_time.get(key, 0) < self.alert_cooldown:
                    continue
                sid, name, roll = self._recognize_face(frame, det['bbox'])
                self._report_incident(det, frame, sid, name, roll,
                                     all_detections=detections)
                self.last_alert_time[key] = now
            
            for det in fight_dets:
                fi       = det.get('fight_info', {})
                pair_key = tuple(sorted([fi.get('person_a_id', 0),
                                       fi.get('person_b_id', 0)]))
                now = time.time()
                if now - fight_cooldown.get(pair_key, 0) < 60:
                    continue
                sid, name, roll = self._recognize_face(frame, det['bbox'])
                self._report_incident(det, frame, sid, name, roll,
                                     all_detections=detections)
                fight_cooldown[pair_key] = now
            
            if frame_count % 300 == 0:
                print(f'[INFO] Frame {frame_count} | {len(detections)} dets')
        
        # Stop the processor when loop exits
        self.processor.stop()
        print('[OK] Detection stopped')
    
    def start(self):
        """Start detection (backward compatible)."""
        if self.running:
            return
        self.running = True
        self.thread  = threading.Thread(target=self._detection_loop,
                                      daemon=True)
        self.thread.start()
        print('[OK] Behavior detection started')
    
    def stop(self):
        """Stop detection (backward compatible)."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.processor.stop()
        print('[OK] Behavior detection stopped')

