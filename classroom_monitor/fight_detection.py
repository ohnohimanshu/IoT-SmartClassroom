"""
Classroom Fight Detection - Backward Compatible Wrapper
==============================================

Maintains exact same public API as original FightDetector
but internally uses the new YOLO + ByteTrack system
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque


class PersonTrack:
    """Backward compatible PersonTrack dataclass."""
    def __init__(self, track_id):
        self.track_id = track_id
        self.bbox_history = deque(maxlen=45)
        self.centroid_history = deque(maxlen=45)
        self.upper_flow_history = deque(maxlen=30)
        self.lower_flow_history = deque(maxlen=30)
        self.last_seen_frame = 0
        self.confidence_scores = deque(maxlen=30)
    
    def avg_upper_flow(self, last_n=12):
        return 0.0
    
    def avg_lower_flow(self, last_n=12):
        return 0.0
    
    def is_upper_active(self, thresh=1.8, last_n=12):
        return False
    
    def is_lower_still(self, thresh=1.2, last_n=12):
        return True


class PersonTracker:
    """Backward compatible PersonTracker."""
    def __init__(self, max_disappeared=45):
        self.next_id = 0
        self.objects = {}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
    
    def update(self, detections, frame_number):
        tracks = []
        for det in detections:
            track = PersonTrack(self.next_id)
            track.bbox_history.append(det[:4])
            cx, cy = (det[0]+det[2])/2, (det[1]+det[3])/2
            track.centroid_history.append((cx, cy))
            track.last_seen_frame = frame_number
            self.objects[self.next_id] = track
            tracks.append(track)
            self.next_id += 1
        return tracks
    
    def _register(self, detection, centroid, frame_number):
        pass


class FightDetector:
    """
    Drop-in replacement for original FightDetector.
    Maintains exact same public API:
    - __init__(fps=30)
    - process_frame(frame, person_detections)
    - get_track_bbox(track_id)
    """
    
    CONFIDENCE_THRESHOLD = 0.45
    
    def __init__(self, fps=30):
        self.fps = fps
        self.tracker = PersonTracker(max_disappeared=45)
        self.frame_count = 0
        self.confirmed_fights = {}
        self._current_detections = {}
    
    def process_frame(self, frame, person_detections):
        """
        Backward compatible process_frame method.
        In new system, fight detection is handled in behavior_detection.py
        This returns empty list for backward compatibility.
        """
        self.frame_count += 1
        
        # Store detections for get_track_bbox backward compatibility
        self._current_detections = {}
        for i, det in enumerate(person_detections):
            self._current_detections[i] = det[:4]
        
        # Return empty list - fight detection now handled in behavior engine
        return []
    
    def get_track_bbox(self, track_id):
        """Get track bbox for backward compatibility."""
        if track_id in self._current_detections:
            return self._current_detections[track_id]
        return None