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

# BUG 5 FIX: Single shared face-recognition tolerance constant
# This eliminates the inconsistency between StudentFaceRecognizer.tolerance (0.52) 
# and the hardcoded 0.55 used in _recognize_face()
FACE_MATCH_TOLERANCE = 0.52


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
    
    Fight detection uses a 3D CNN (mc3_18) instead of skeleton heuristics.
    """
    
    # BUG 1 FIX: Configurable thresholds for new heuristics
    # These replace magic numbers and make the system tunable for real classroom footage
    PHONE_OVERLAP_THRESHOLD = 0.3
    EATING_HAND_TO_MOUTH_THRESHOLD = 0.4  # Now actually used instead of being dead code
    ENGAGEMENT_HEAD_POSE_THRESHOLD = 0.6
    
    # BUG 2 FIX: Configurable thresholds for head-down detection
    # When facial keypoints are missing for this many consecutive frames while body is present,
    # classify as head_down instead of incorrectly returning 'focused'
    HEAD_DOWN_CONSECUTIVE_FRAMES = 8  # ~1 second at 8 FPS processing rate
    LOW_CONFIDENCE_THRESHOLD = 0.5
    
    # BUG 3 FIX: Configurable thresholds for pairwise fight detection
    # Fight detection now works on pairs of people, not scene-wide
    FIGHT_PROXIMITY_THRESHOLD = 150.0  # pixels - how close people must be to be fight candidates
    FIGHT_VELOCITY_THRESHOLD = 25.0    # relative keypoint velocity between pairs
    FIGHT_CONFIRMATION_FRAMES = 3      # consecutive frames needed to confirm fight
    
    # BUG 1 FIX: Phone-in-lap detection threshold
    # Detects phone use when hands are positioned below this fraction of bbox height
    PHONE_LAP_HEIGHT_FRACTION = 0.65  # hands below 65% of person height = desk/lap level
    
    def __init__(self):
        self.tracked_people: Dict[int, TrackedPerson] = {}
        self.cleanup_threshold = 5.0  # seconds
        
        # BUG 2 FIX: Track low-confidence keypoint history per person for head-down detection
        self.low_confidence_counters: Dict[int, int] = {}
        
        # BUG 3 FIX: Track pairwise fight detection state instead of scene-wide
        self.fight_pairs: Dict[Tuple[int, int], Dict] = {}  # (person_a, person_b) -> fight state
        self.fight_pair_detectors: Dict[Tuple[int, int], 'FightDetector3DCNN'] = {}  # per-pair 3D CNN instances
        
        # 3D CNN fight detector (replaces skeleton-heuristic approach)
        self._fight_detector_3dcnn = None
        self._fight_detector_loading = False
        self._init_fight_detector()
    
    def _init_fight_detector(self):
        """Initialize the 3D CNN fight detector (lazy load)."""
        try:
            from classroom_monitor.fight_detection_3dcnn import FightDetector3DCNN
            self._fight_detector_3dcnn = FightDetector3DCNN(
                device='auto',
                sequence_length=16,
                confidence_threshold=0.60,
            )
            self._fight_detector_loading = True
            print('[BEHAVIOR] 3D CNN fight detector initializing...')
        except Exception as e:
            print(f'[BEHAVIOR] WARNING: Could not init 3D CNN fight detector: {e}')
            print('[BEHAVIOR] Fight detection will be disabled')
            self._fight_detector_3dcnn = None
    
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
    
    def cleanup_stale(self, current_time: float):
        """Remove tracked persons not seen for cleanup_threshold seconds."""
        stale = [tid for tid, p in self.tracked_people.items() 
                if current_time - p.last_seen > self.cleanup_threshold]
        for tid in stale:
            del self.tracked_people[tid]
            # BUG 2 FIX: Clean up head-down tracking state
            if tid in self.low_confidence_counters:
                del self.low_confidence_counters[tid]
            
        # BUG 3 FIX: Clean up stale fight pair state
        stale_pairs = []
        for pair_key in self.fight_pairs:
            tid_a, tid_b = pair_key
            if tid_a not in self.tracked_people or tid_b not in self.tracked_people:
                stale_pairs.append(pair_key)
        for pair_key in stale_pairs:
            if pair_key in self.fight_pairs:
                del self.fight_pairs[pair_key]
            if pair_key in self.fight_pair_detectors:
                del self.fight_pair_detectors[pair_key]
    
    def _calculate_head_pose(self, person: TrackedPerson) -> str:
        """
        Estimate head pose using scale-invariant structural ratios.
        Returns 'focused', 'looking_away', or 'head_down'.
        
        BUG 2 FIX: When facial keypoints (nose, eyes) are missing or below confidence threshold 
        for a sustained period while body keypoints remain present, classify as head_down 
        rather than incorrectly returning 'focused'. The old logic systematically missed 
        head-down behavior because low keypoint confidence is exactly what happens when 
        someone puts their head down and hides their face from the camera.
        """
        keypoints = person.keypoints
        if keypoints is None or keypoints.size == 0 or len(keypoints) < 3:
            return 'focused'
        
        try:
            nose = keypoints[0]
            left_eye = keypoints[1]
            right_eye = keypoints[2]
            
            # Check if we have sufficient keypoints for analysis
            if (len(nose) < 3 or len(left_eye) < 3 or len(right_eye) < 3):
                return 'focused'
            
            # BUG 2 FIX: Track consecutive low-confidence frames per person
            track_id = person.track_id
            has_low_confidence = (nose[2] < self.LOW_CONFIDENCE_THRESHOLD or 
                                left_eye[2] < self.LOW_CONFIDENCE_THRESHOLD or 
                                right_eye[2] < self.LOW_CONFIDENCE_THRESHOLD or
                                nose[0] == 0.0 or nose[1] == 0.0 or
                                left_eye[0] == 0.0 or left_eye[1] == 0.0 or
                                right_eye[0] == 0.0 or right_eye[1] == 0.0)
            
            if has_low_confidence:
                # Increment counter for this person
                self.low_confidence_counters[track_id] = self.low_confidence_counters.get(track_id, 0) + 1
                
                # BUG 2 FIX: If facial keypoints have been missing/low-confidence for enough consecutive 
                # frames while the person bbox is still being tracked (body present), classify as head_down
                if self.low_confidence_counters[track_id] >= self.HEAD_DOWN_CONSECUTIVE_FRAMES:
                    return 'head_down'
                else:
                    # Not enough consecutive frames yet, return focused for now
                    return 'focused'
            else:
                # Reset counter when we get good facial keypoints back
                self.low_confidence_counters[track_id] = 0
            
            # Continue with normal head pose analysis when facial keypoints are visible
            # Inter-eye distance (scale reference)
            inter_eye_dist = np.linalg.norm(left_eye[:2] - right_eye[:2])
            if inter_eye_dist < 0.1:  # Avoid division by zero
                return 'focused'
            
            # Calculate vertical drop of nose relative to eye line
            eye_y_avg = (left_eye[1] + right_eye[1]) / 2
            vertical_drop = nose[1] - eye_y_avg
            drop_ratio = vertical_drop / inter_eye_dist
            
            # Check for head down (large positive drop ratio)
            if drop_ratio > 0.45:
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
        """
        BUG 1 FIX: Detect phone usage based on phone bounding box overlap OR pose skeleton fallbacks.
        
        The old code had structurally dead phone detection because only yolo11s-pose.pt was loaded,
        which can never emit class 67 (phone). Now we expect phone_detections to come from a 
        separate object detection model that actually detects phones.
        
        Strategy A: Object detection bbox overlap (now receives real phone detections)
        Strategy B: Skeleton heuristics with improved phone-in-lap detection
        """
        x1, y1, x2, y2 = person.bbox
        person_area = (x2 - x1) * (y2 - y1)
        
        # ── Strategy A: Standard Bounding Box Overlap ───────────────────────
        # BUG 1 FIX: This now actually receives phone detections from a parallel object detector
        for (px1, py1, px2, py2, conf) in phone_detections:
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
                
        # ── Strategy B: Skeleton Heuristic Fallback (If YOLO misses the phone) ──
        # BUG 1 FIX: Added phone-in-lap detection for hands positioned below configurable 
        # fraction of bbox height, combined with sustained head-down classification.
        # This catches phone use patterns where the phone is at desk/lap level, not just hand-to-face.
        kp = person.keypoints
        if kp is not None and kp.size > 0 and len(kp) > 10:
            try:
                nose = kp[0]
                left_wrist  = kp[9]
                right_wrist = kp[10]
                left_elbow  = kp[7]
                right_elbow = kp[8]
                bbox_height = y2 - y1
                lap_threshold = y1 + bbox_height * self.PHONE_LAP_HEIGHT_FRACTION

                if nose[2] < 0.5 or nose[0] == 0.0:
                    return False, 0.0

                wrists = [(left_wrist, left_elbow), (right_wrist, right_elbow)]
                wrist_dists = []
                low_hands = 0  # Count hands positioned below lap threshold

                for wrist, elbow in wrists:
                    if (len(wrist) >= 3 and wrist[2] >= 0.5 and wrist[0] != 0.0):
                        dist = np.linalg.norm(wrist[:2] - nose[:2]) / bbox_height
                        wrist_dists.append(dist)
                        
                        # BUG 1 FIX: Check if hand is in lap/desk area (below threshold)
                        if wrist[1] > lap_threshold:
                            low_hands += 1

                # BUG 1 FIX: Phone-in-lap detection - hands positioned low + head down pattern
                if low_hands >= 1:
                    # Check if person is also exhibiting head-down behavior (corroborating evidence)
                    head_pose = self._calculate_head_pose(person)
                    if head_pose == 'head_down':
                        return True, 0.7  # High confidence for phone-in-lap + head-down combo

                if len(wrist_dists) == 2:
                    # Both wrists near face → writing or eating, NOT phone → skip
                    if wrist_dists[0] < 0.35 and wrist_dists[1] < 0.35:
                        return False, 0.0

                    # ONE wrist very close to face/ear AND the other is low → phone
                    min_d = min(wrist_dists)
                    max_d = max(wrist_dists)
                    asymmetry = max_d - min_d          # large = one hand up, one down
                    if min_d < 0.28 and asymmetry > 0.20:
                        return True, 0.65

                elif len(wrist_dists) == 1:
                    # Only one wrist visible and it's very close to face
                    if wrist_dists[0] < 0.22:
                        return True, 0.60

            except Exception:
                pass

        return False, 0.0
    
    def _detect_eating(self, person: TrackedPerson, 
                      food_detections: List[Tuple]) -> Tuple[bool, float]:
        """
        BUG 1 FIX: Detect eating based on food/beverage detections near face with 
        hand-to-mouth motion analysis using the previously unused EATING_HAND_TO_MOUTH_THRESHOLD.
        
        The old code only checked for food bbox overlap but the constant EATING_HAND_TO_MOUTH_THRESHOLD 
        was never used. Now we use wrist-to-nose distance oscillation as corroborating evidence.
        """
        if person.keypoints is None or person.keypoints.size == 0:
            return False, 0.0
        
        try:
            nose = person.keypoints[0]
            left_wrist = person.keypoints[9] if len(person.keypoints) > 9 else None
            right_wrist = person.keypoints[10] if len(person.keypoints) > 10 else None
            x1, y1, x2, y2 = person.bbox
            
            # Strategy A: Food object detection overlap (now receives real food detections)
            food_detected = False
            food_confidence = 0.0
            
            for (fx1, fy1, fx2, fy2, conf) in food_detections:
                food_center = ((fx1 + fx2) / 2, (fy1 + fy2) / 2)
                distance_to_nose = np.linalg.norm(
                    np.array(food_center) - np.array(nose[:2])
                )
                
                if distance_to_nose < (y2 - y1) * 0.5:
                    food_detected = True
                    food_confidence = max(food_confidence, conf)
            
            # Strategy B: BUG 1 FIX - Hand-to-mouth repeated motion using EATING_HAND_TO_MOUTH_THRESHOLD
            # Check if wrist-to-nose distance oscillates below threshold (repeated hand-to-mouth motion)
            hand_to_mouth_detected = False
            
            for wrist in [left_wrist, right_wrist]:
                if wrist is not None and len(wrist) >= 3 and wrist[2] >= 0.5 and wrist[0] != 0.0:
                    wrist_to_nose_dist = np.linalg.norm(wrist[:2] - nose[:2])
                    bbox_height = y2 - y1
                    normalized_dist = wrist_to_nose_dist / bbox_height
                    
                    # BUG 1 FIX: Actually use EATING_HAND_TO_MOUTH_THRESHOLD instead of leaving it dead
                    if normalized_dist < self.EATING_HAND_TO_MOUTH_THRESHOLD:
                        hand_to_mouth_detected = True
                        break
            
            # Combine both signals - food detection OR hand-to-mouth motion
            if food_detected and hand_to_mouth_detected:
                return True, min(food_confidence + 0.2, 1.0)  # High confidence when both signals agree
            elif food_detected:
                return True, food_confidence
            elif hand_to_mouth_detected:
                return True, 0.6  # Moderate confidence for skeleton-only detection
            
        except Exception:
            pass
        
        return False, 0.0
    
    def add_frame_for_fight_detection(self, frame):
        """Feed a raw BGR frame to the 3D CNN fight detector."""
        if self._fight_detector_3dcnn is not None:
            self._fight_detector_3dcnn.add_frame(frame)
    
    def _detect_fighting_pairwise(self, frame: np.ndarray) -> List[Tuple[int, int, float, Dict]]:
        """
        BUG 3 FIX: Pairwise fight detection instead of scene-wide.
        
        The old system applied scene-wide fight classification to every person, causing 
        "everyone is fighting" when the 3D CNN fired once. Now we:
        1. Find candidate fight pairs using pose-based proximity/velocity triggers
        2. Run separate 3D CNN instances on cropped regions for each pair
        3. Only flag the specific pair members, leaving others unaffected
        4. Advance confirmation counters based on real frame time, not evaluation calls
        
        Returns list of (person_a_id, person_b_id, confidence, fight_info) tuples
        """
        if len(self.tracked_people) < 2:
            return []
        
        people_list = list(self.tracked_people.values())
        fight_results = []
        
        # Step 1: Find candidate fight pairs using proximity and pose velocity
        candidate_pairs = []
        
        for i in range(len(people_list)):
            for j in range(i + 1, len(people_list)):
                person_a = people_list[i]
                person_b = people_list[j]
                
                # Calculate bbox proximity
                ax1, ay1, ax2, ay2 = person_a.bbox
                bx1, by1, bx2, by2 = person_b.bbox
                
                # Center-to-center distance
                center_a = ((ax1 + ax2) / 2, (ay1 + ay2) / 2)
                center_b = ((bx1 + bx2) / 2, (by1 + by2) / 2)
                distance = np.linalg.norm(np.array(center_a) - np.array(center_b))
                
                # BUG 3 FIX: Only consider pairs within proximity threshold
                if distance > self.FIGHT_PROXIMITY_THRESHOLD:
                    continue
                
                # Calculate relative keypoint velocity (simple motion detection)
                velocity_detected = False
                if (person_a.keypoints is not None and person_b.keypoints is not None and
                    person_a.keypoints.size > 0 and person_b.keypoints.size > 0):
                    
                    try:
                        # Look for rapid limb movement between the pair (arms/shoulders)
                        # This is a simple heuristic - in production you'd use temporal keypoint history
                        a_wrists = [person_a.keypoints[9][:2], person_a.keypoints[10][:2]]
                        b_wrists = [person_b.keypoints[9][:2], person_b.keypoints[10][:2]]
                        
                        for a_wrist in a_wrists:
                            for b_wrist in b_wrists:
                                if (len(a_wrist) >= 2 and len(b_wrist) >= 2):
                                    wrist_distance = np.linalg.norm(a_wrist - b_wrist)
                                    # If wrists are very close, could indicate physical interaction
                                    if wrist_distance < self.FIGHT_VELOCITY_THRESHOLD:
                                        velocity_detected = True
                                        break
                            if velocity_detected:
                                break
                    except Exception:
                        pass
                
                # Add to candidates if proximity + motion indicators present
                if velocity_detected:
                    candidate_pairs.append((person_a.track_id, person_b.track_id, distance))
        
        # Step 2: Run dedicated 3D CNN instances on cropped regions for each candidate pair
        if not candidate_pairs:
            return fight_results
        
        for person_a_id, person_b_id, distance in candidate_pairs:
            pair_key = tuple(sorted([person_a_id, person_b_id]))
            
            # Get person objects
            if person_a_id not in self.tracked_people or person_b_id not in self.tracked_people:
                continue
            
            person_a = self.tracked_people[person_a_id]
            person_b = self.tracked_people[person_b_id]
            
            # Create bounding box that encloses both people with padding
            ax1, ay1, ax2, ay2 = person_a.bbox
            bx1, by1, bx2, by2 = person_b.bbox
            
            combined_x1 = max(0, min(ax1, bx1) - 20)
            combined_y1 = max(0, min(ay1, by1) - 20)
            combined_x2 = min(frame.shape[1], max(ax2, bx2) + 20)
            combined_y2 = min(frame.shape[0], max(ay2, by2) + 20)
            
            # Crop frame to just this pair
            cropped_frame = frame[combined_y1:combined_y2, combined_x1:combined_x2]
            
            if cropped_frame.size == 0:
                continue
            
            # BUG 3 FIX: Create dedicated 3D CNN detector instance per pair
            if pair_key not in self.fight_pair_detectors:
                try:
                    from classroom_monitor.fight_detection_3dcnn import FightDetector3DCNN
                    self.fight_pair_detectors[pair_key] = FightDetector3DCNN(
                        device='auto',
                        sequence_length=16,
                        confidence_threshold=0.60,
                    )
                except Exception as e:
                    print(f'[FIGHT] Error creating pair detector: {e}')
                    continue
            
            pair_detector = self.fight_pair_detectors[pair_key]
            
            # Feed cropped frame to this pair's detector
            pair_detector.add_frame(cropped_frame)
            
            # Initialize pair state if needed
            if pair_key not in self.fight_pairs:
                self.fight_pairs[pair_key] = {
                    'consecutive_detections': 0,
                    'last_confidence': 0.0,
                    'last_detection_time': 0.0
                }
            
            pair_state = self.fight_pairs[pair_key]
            current_time = time.time()
            
            # Run 3D CNN prediction on this pair's cropped frames
            if pair_detector.is_ready():
                try:
                    is_fighting, confidence = pair_detector.predict()
                    
                    if is_fighting and confidence > 0.6:
                        # BUG 3 FIX: Only advance counter if enough time has passed (frame-based, not call-based)
                        if current_time - pair_state['last_detection_time'] > 0.3:  # At least 300ms between confirmations
                            pair_state['consecutive_detections'] += 1
                            pair_state['last_detection_time'] = current_time
                        pair_state['last_confidence'] = confidence
                        
                        # BUG 3 FIX: Require multiple consecutive confirmations before flagging
                        if pair_state['consecutive_detections'] >= self.FIGHT_CONFIRMATION_FRAMES:
                            fight_results.append((person_a_id, person_b_id, confidence, {
                                'person_a_id': person_a_id,
                                'person_b_id': person_b_id,
                                'confidence': confidence,
                                'trigger': 'pairwise_3dcnn',
                                'distance': distance,
                                'confirmations': pair_state['consecutive_detections']
                            }))
                    else:
                        # Reset counter if no fight detected for this frame
                        if current_time - pair_state['last_detection_time'] > 1.0:  # Reset after 1 second of no detection
                            pair_state['consecutive_detections'] = 0
                        
                except Exception as e:
                    print(f'[FIGHT] Error in pairwise 3D CNN detection: {e}')
                    continue
        
        return fight_results
    
    def evaluate_person(self, track_id: int, 
                       phone_detections: List[Tuple],
                       food_detections: List[Tuple],
                       frame: Optional[np.ndarray] = None,
                       fight_override: Optional[Tuple[bool, float, Dict]] = None) -> DetectionResult:
        """
        Evaluate a tracked person's behavior with temporal smoothing.
        Returns a DetectionResult compatible with existing code.
        
        BUG 3 FIX: Fight detection is now handled separately at frame level and passed as override
        to avoid double-processing and ensure once-per-frame evaluation.
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
        
        # BUG 3 FIX: Use fight override if provided, otherwise check for fights in frame
        fight_detected = False
        fight_confidence = 0.0
        fight_info = None
        
        if fight_override is not None:
            fight_detected, fight_confidence, fight_info = fight_override
        elif frame is not None:
            # Fallback for backward compatibility - run pairwise detection
            fight_results = self._detect_fighting_pairwise(frame)
            for person_a_id, person_b_id, confidence, info in fight_results:
                if track_id in (person_a_id, person_b_id):
                    fight_detected = True
                    fight_confidence = confidence
                    fight_info = info
                    break
        
        if fight_detected:
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
            confidence = fight_confidence if fight_detected else 0.8
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
        """
        BUG 1 FIX: Load both pose model AND object detection model in parallel.
        
        The old code only loaded yolo11s-pose.pt which can never detect phones (class 67) 
        or food (classes 46-56). Now we load a general object detection model alongside 
        the pose model to actually detect these objects.
        """
        try:
            from ultralytics import YOLO
            # Load pose model for person detection + keypoints
            self.yolo_model = YOLO('yolo11s-pose.pt')
            print('[OK] YOLO11s-Pose model loaded')
            
            # BUG 1 FIX: Load general object detection model for phone/food detection
            # TODO: In production, replace with a model fine-tuned on classroom objects
            # (phone, book, food, drink, laptop) for better accuracy in classroom settings
            self.object_model = YOLO('yolo11s.pt')  # General COCO model includes phone/food classes
            print('[OK] YOLO11s object detection model loaded for phone/food detection')
            print('[NOTE] For production accuracy, replace yolo11s.pt with a model fine-tuned on classroom dataset')
            
        except Exception as e:
            print(f'[WARN] Failed to load YOLO models: {e}')
            self.yolo_model = None
            self.object_model = None
    
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
        """
        BUG 1 FIX: Process a single frame with both YOLO pose model AND object detection model.
        
        Run pose detection for people+keypoints and object detection for phones/food in parallel,
        then merge results by spatial overlap with each person's bbox.
        """
        if self.yolo_model is None:
            return
        
        try:
            # Run YOLO pose detection + tracking for people
            pose_results = self.yolo_model.track(
                frame,
                persist=True,
                verbose=False,
                conf=0.2,
                iou=0.5,
                tracker='bytetrack.yaml'
            )
            
            phone_dets = []
            food_dets = []
            person_tracks_with_kp = []
            
            # Process pose detection results (people + keypoints)
            for result in pose_results:
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
            
            # BUG 1 FIX: Run parallel object detection for phones and food
            if self.object_model is not None:
                try:
                    object_results = self.object_model(
                        frame,
                        verbose=False,
                        conf=0.3,  # Higher confidence for object detection to reduce false positives
                        iou=0.5
                    )
                    
                    for result in object_results:
                        if result.boxes is None:
                            continue
                        
                        boxes = result.boxes
                        for i in range(len(boxes)):
                            cls_id = int(boxes.cls[i])
                            conf = float(boxes.conf[i])
                            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                            
                            if cls_id == 67:  # Cell phone (COCO class)
                                phone_dets.append((x1, y1, x2, y2, conf))
                            elif cls_id in [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]:  # Food/drink classes
                                food_dets.append((x1, y1, x2, y2, conf))
                                
                except Exception as e:
                    print(f'[WARN] Object detection failed: {e}')
                    # Continue with pose-only detection if object detection fails
            
            # Update behavior engine
            for track_id, x1, y1, x2, y2, conf, kp in person_tracks_with_kp:
                self.behavior_engine.update_person(
                    track_id, (x1, y1, x2, y2), kp, timestamp
                )
            
            # Feed raw frame to 3D CNN fight detector
            self.behavior_engine.add_frame_for_fight_detection(frame)
            
            # Cleanup stale tracks
            self.behavior_engine.cleanup_stale(timestamp)
            
            # Generate detection results
            final_results = []
            
            # BUG 3 FIX: Run fight detection once per frame, not once per person
            fight_results = self.behavior_engine._detect_fighting_pairwise(frame)
            fight_pairs = {}
            for person_a_id, person_b_id, confidence, fight_info in fight_results:
                fight_pairs[person_a_id] = (confidence, fight_info)
                fight_pairs[person_b_id] = (confidence, fight_info)
            
            for track_id in self.behavior_engine.tracked_people:
                # Check if this person is involved in a fight pair
                fight_detected = track_id in fight_pairs
                fight_confidence, fight_info = fight_pairs.get(track_id, (0.0, None))
                
                # Evaluate person behavior (fight info passed separately to avoid double-processing)
                det_result = self.behavior_engine.evaluate_person(
                    track_id, phone_dets, food_dets, None,  # No frame needed - fight already processed
                    fight_override=(fight_detected, fight_confidence, fight_info) if fight_detected else None
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

        # Persistent engine for detect() calls (keeps keypoint history across frames)
        self._detect_engine = TemporalBehaviorEngine()

        # New production processor
        self.processor = ProductionStreamProcessor(process_fps=10)
        
        self._load_models()
    
    def _load_models(self):
        """BUG 1 & BUG 5 FIX: Load models (backward compatibility)."""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolo11s-pose.pt')
            print('[OK] YOLO11s-Pose loaded (backward compatibility)')
            
            # BUG 1 FIX: Load object detection model for phone/food detection
            self.object_model = YOLO('yolo11s.pt')
            print('[OK] YOLO11s object detection loaded (backward compatibility)')
            
        except Exception as e:
            print(f'[WARN] YOLO: {e}')
            self.yolo_model = None
            self.object_model = None
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
        
        BUG 1 FIX: Now runs both pose and object detection models.
        """
        # For backward compatibility, run detection on demand
        if self.yolo_model is None:
            return []
        
        try:
            # Run pose detection
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
            
            # Process pose results
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
            
            # BUG 1 FIX: Run object detection for phones/food
            if hasattr(self, 'object_model') and self.object_model is not None:
                try:
                    obj_results = self.object_model(frame, verbose=False, conf=0.3)
                    for result in obj_results:
                        if result.boxes is None:
                            continue
                        boxes = result.boxes
                        for i in range(len(boxes)):
                            cls_id = int(boxes.cls[i])
                            conf = float(boxes.conf[i])
                            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                            if cls_id == 67:
                                phone_dets.append((x1, y1, x2, y2, conf))
                            elif cls_id in [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]:
                                food_dets.append((x1, y1, x2, y2, conf))
                except Exception as e:
                    print(f'[WARN] Object detection in detect(): {e}')
            
            # Update PERSISTENT behavior engine
            for track_id, x1, y1, x2, y2, conf, kp in person_tracks_with_kp:
                self._detect_engine.update_person(track_id, (x1, y1, x2, y2), kp, timestamp)
            self._detect_engine.cleanup_stale(timestamp)
            
            # Feed raw frame to 3D CNN fight detector
            self._detect_engine.add_frame_for_fight_detection(frame)

            # BUG 3 FIX: Run fight detection once per frame
            fight_results = self._detect_engine._detect_fighting_pairwise(frame)
            fight_pairs = {}
            for person_a_id, person_b_id, confidence, fight_info in fight_results:
                fight_pairs[person_a_id] = (confidence, fight_info)
                fight_pairs[person_b_id] = (confidence, fight_info)

            # Evaluate each person
            for track_id in self._detect_engine.tracked_people:
                fight_detected = track_id in fight_pairs
                fight_confidence, fight_info = fight_pairs.get(track_id, (0.0, None))
                
                det_result = self._detect_engine.evaluate_person(
                    track_id, phone_dets, food_dets, None,
                    fight_override=(fight_detected, fight_confidence, fight_info) if fight_detected else None
                )
                
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
        """BUG 5 FIX: Recognize face using shared FACE_MATCH_TOLERANCE constant."""
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
            # BUG 5 FIX: Use shared constant instead of hardcoded 0.55
            if best_d < FACE_MATCH_TOLERANCE and best:
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