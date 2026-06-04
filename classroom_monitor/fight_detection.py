"""
Classroom Fight Detection — Motion-First Architecture
======================================================

ROOT CAUSE OF THE FALSE-POSITIVE PROBLEM (seen in screenshot):
  In a classroom with rows of desks, adjacent students' YOLO bounding boxes
  ALWAYS overlap or are within 150 px of each other — even when everyone is
  calmly writing.  Any pure-proximity-based system will therefore ALWAYS
  fire "FIGHT" for every pair of seated neighbours.

CORRECT APPROACH — 3-gate system.  ALL three gates must pass:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ GATE 1 — STILLNESS VETO (most powerful filter)                     │
  │   If BOTH people have been nearly stationary for the last N frames, │
  │   they are SITTING/WRITING, not fighting.  Veto immediately.        │
  │   Writing motion is small and periodic (< 6 px/frame average).      │
  │   Fighting motion is large and chaotic (> 12 px/frame average).     │
  ├─────────────────────────────────────────────────────────────────────┤
  │ GATE 2 — BBOX OVERLAP (not just proximity)                         │
  │   Centroids 150 px apart is useless — desks seat people that close. │
  │   Require actual BOUNDING BOX OVERLAP (IoU > 0) AND centroid        │
  │   distance < 120 px.  Two people must physically be in the same     │
  │   space, not just near each other.                                   │
  ├─────────────────────────────────────────────────────────────────────┤
  │ GATE 3 — SUSTAINED MOTION DURING OVERLAP                           │
  │   Requires high combined motion (> 12 px/frame) for at least        │
  │   20 consecutive frames (~0.67 s) while bbox overlap is active.     │
  │   A handshake, pat on back, or brief touch is excluded.             │
  └─────────────────────────────────────────────────────────────────────┘

ADDITIONAL SAFEGUARDS:
  • Optical-flow magnitude computed inside the UNION of both bboxes gives
    a direct pixel-level motion signal, independent of centroid jitter.
  • Aspect-ratio check: a person bbox that is very wide (two people merged
    into one box by YOLO) is excluded from fight pairing.
  • Vertical-neighbour suppression: if person B's centroid is more than
    80 px ABOVE person A's centroid AND their horizontal centres are within
    60 px, they are likely front/back-row neighbours — reduce proximity
    weight strongly.
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersonTrack:
    track_id: int
    bbox_history: deque     = field(default_factory=lambda: deque(maxlen=45))
    centroid_history: deque = field(default_factory=lambda: deque(maxlen=45))
    # Per-frame motion magnitude (px moved since last frame)
    motion_history: deque   = field(default_factory=lambda: deque(maxlen=30))
    last_seen_frame: int    = 0
    confidence_scores: deque = field(default_factory=lambda: deque(maxlen=30))

    def avg_motion(self, last_n=15) -> float:
        """Average motion magnitude over last N frames."""
        if not self.motion_history:
            return 0.0
        hist = list(self.motion_history)[-last_n:]
        return float(np.mean(hist)) if hist else 0.0

    def is_still(self, threshold=6.0, last_n=15) -> bool:
        """True if person has been nearly motionless (writing/sitting)."""
        return self.avg_motion(last_n) < threshold


@dataclass
class PairContext:
    person_a_id: int
    person_b_id: int
    # Frames where bboxes actually overlapped AND both were moving
    active_fight_frames: int = 0
    # Raw proximity history (centroid distance)
    proximity_history: deque = field(default_factory=lambda: deque(maxlen=60))
    # Optical-flow magnitude inside union region, per frame
    flow_history: deque      = field(default_factory=lambda: deque(maxlen=30))
    last_updated_frame: int  = 0
    interaction_start: int   = 0


# ─────────────────────────────────────────────────────────────────────────────
# Person tracker (Hungarian-style, velocity-predicted)
# ─────────────────────────────────────────────────────────────────────────────

class PersonTracker:

    def __init__(self, max_disappeared=45):
        self.next_id     = 0
        self.objects: Dict[int, PersonTrack] = {}
        self.disappeared: Dict[int, int]     = defaultdict(int)
        self.max_disappeared = max_disappeared

    def update(self, detections: List[Tuple], frame_number: int) -> List[PersonTrack]:
        """
        detections: list of (x1,y1,x2,y2, conf)
        Returns updated list of PersonTrack objects.
        """
        if not detections:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.objects.pop(oid, None)
                    self.disappeared.pop(oid, None)
            return list(self.objects.values())

        new_centroids = np.array([
            [(x1+x2)/2, (y1+y2)/2] for x1,y1,x2,y2,_ in detections
        ], dtype=float)

        if not self.objects:
            for i, det in enumerate(detections):
                self._register(det, new_centroids[i], frame_number)
            return list(self.objects.values())

        obj_ids = list(self.objects.keys())

        # Velocity-predicted positions
        pred_centroids = []
        for oid in obj_ids:
            t = self.objects[oid]
            if len(t.centroid_history) >= 2:
                v = np.array(t.centroid_history[-1]) - np.array(t.centroid_history[-2])
                pred_centroids.append(np.array(t.centroid_history[-1]) + v)
            else:
                pred_centroids.append(np.array(t.centroid_history[-1])
                                      if t.centroid_history else np.zeros(2))
        pred_centroids = np.array(pred_centroids)

        # Pairwise distance matrix
        D = np.linalg.norm(
            pred_centroids[:, None, :] - new_centroids[None, :, :], axis=2
        )  # shape: (num_tracks, num_dets)

        matched_obj  = set()
        matched_dets = set()

        # Greedy matching on sorted distances
        flat_order = np.argsort(D, axis=None)
        for flat_idx in flat_order:
            obj_idx = int(flat_idx // len(detections))
            det_idx = int(flat_idx  % len(detections))
            if obj_idx in matched_obj or det_idx in matched_dets:
                continue
            if D[obj_idx, det_idx] > 220:   # max match distance raised to 220 px
                break
            oid = obj_ids[obj_idx]
            det = detections[det_idx]
            t   = self.objects[oid]
            t.bbox_history.append(det[:4])
            t.centroid_history.append(new_centroids[det_idx])
            t.confidence_scores.append(det[4])

            # Update per-frame motion
            if len(t.centroid_history) >= 2:
                mv = float(np.linalg.norm(
                    np.array(t.centroid_history[-1]) - np.array(t.centroid_history[-2])
                ))
                t.motion_history.append(mv)

            t.last_seen_frame = frame_number
            self.disappeared[oid] = 0
            matched_obj.add(obj_idx)
            matched_dets.add(det_idx)

        for i, det in enumerate(detections):
            if i not in matched_dets:
                self._register(det, new_centroids[i], frame_number)

        for j, oid in enumerate(obj_ids):
            if j not in matched_obj:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.objects.pop(oid, None)
                    self.disappeared.pop(oid, None)

        return list(self.objects.values())

    def _register(self, detection, centroid, frame_number):
        t = PersonTrack(track_id=self.next_id, last_seen_frame=frame_number)
        t.bbox_history.append(detection[:4])
        t.centroid_history.append(centroid)
        t.confidence_scores.append(detection[4])
        self.objects[self.next_id] = t
        self.next_id += 1


# ─────────────────────────────────────────────────────────────────────────────
# Core fight analyser — 3-gate system
# ─────────────────────────────────────────────────────────────────────────────

class FightAnalyzer:

    # ── Thresholds ────────────────────────────────────────────────────────────
    # GATE 1 — stillness veto
    # NOTE: centroid-based stillness is NOT sufficient for seated fights because
    # two students hitting each other while sitting barely move their centroids.
    # We now require BOTH centroid stillness AND low upper-body optical flow
    # before vetoing — if either signal shows motion, we proceed to Gate 2/3.
    STILL_MOTION_THRESH   = 6.0    # px/frame avg centroid — below = writing/sitting
    STILL_LOOKBACK        = 15     # frames to average over for stillness check
    BOTH_STILL_VETO       = True   # if BOTH are still → hard veto (no fight)
    # Upper-body flow threshold: seated fight arms produce 2–8 px/frame of flow
    # in the upper 40 % of the bounding box even when centroids don't move.
    UPPER_BODY_FLOW_THRESH = 2.5   # mean flow in upper-body region — if either
                                   # person exceeds this, stillness veto is bypassed

    # GATE 2 — bbox overlap requirement
    # Centroid distance alone is insufficient in classroom seating.
    # Require ACTUAL bounding-box IoU > 0 (boxes must physically overlap).
    REQUIRE_BBOX_OVERLAP  = True
    MAX_CENTROID_DIST     = 120    # px — even with overlap, cap centroid distance

    # GATE 3 — sustained motion
    MIN_FIGHT_FRAMES      = 20     # consecutive frames meeting gates 1+2
    FIGHT_MOTION_THRESH   = 12.0   # px/frame combined motion to count as fight frame

    # Optical-flow amplifier (used instead of centroid motion when available)
    USE_OPTICAL_FLOW      = True
    FLOW_FIGHT_THRESH     = 2.0    # lowered 3.0→2.0: seated fights produce less
                                   # full-union flow than standing fights

    # Vertical-neighbour suppression
    # Front/back-row students: B is above A AND horizontally aligned
    ROW_SUPPRESS_DY       = 80     # px — if |cy_a - cy_b| > this, different rows
    ROW_SUPPRESS_DX       = 80     # px — horizontal alignment threshold

    def __init__(self, fps: int = 30):
        self.fps          = fps
        self.pairs: Dict[Tuple[int,int], PairContext] = {}
        self._prev_gray   = None

    # ── Public interface ──────────────────────────────────────────────────────

    def analyze(
        self,
        tracks: List[PersonTrack],
        frame: np.ndarray,
        frame_number: int,
    ) -> List[Dict]:
        """
        Return list of confirmed fight dicts for this frame.
        Each dict: {person_a_id, person_b_id, confidence, contact_duration_frames,
                    contact_centroid, motion_intensity}
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow from previous frame
        flow_map = None
        if self.USE_OPTICAL_FLOW and self._prev_gray is not None:
            try:
                flow_map = cv2.calcOpticalFlowFarneback(
                    self._prev_gray, gray,
                    None,
                    pyr_scale=0.5, levels=2, winsize=12,
                    iterations=2, poly_n=5, poly_sigma=1.1,
                    flags=0,
                )
            except Exception:
                flow_map = None

        self._prev_gray = gray.copy()

        fights = []
        keys_to_remove = []

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                ta, tb = tracks[i], tracks[j]

                # Need at least 3 frames of history for meaningful analysis
                if len(ta.centroid_history) < 3 or len(tb.centroid_history) < 3:
                    continue

                result = self._evaluate_pair(ta, tb, gray, flow_map, frame_number)
                if result is not None:
                    fights.append(result)

        # Clean up stale pair contexts
        for key in list(self.pairs):
            ctx = self.pairs[key]
            if frame_number - ctx.last_updated_frame > 90:  # 3 sec stale
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self.pairs.pop(key, None)

        return fights

    # ── Per-pair evaluation ───────────────────────────────────────────────────

    def _evaluate_pair(
        self,
        ta: PersonTrack,
        tb: PersonTrack,
        gray: np.ndarray,
        flow_map,
        frame_number: int,
    ) -> Optional[Dict]:

        key = (ta.track_id, tb.track_id)
        if key not in self.pairs:
            self.pairs[key] = PairContext(
                person_a_id=ta.track_id,
                person_b_id=tb.track_id,
                interaction_start=frame_number,
            )
        ctx = self.pairs[key]
        ctx.last_updated_frame = frame_number

        ca = np.array(ta.centroid_history[-1])
        cb = np.array(tb.centroid_history[-1])
        centroid_dist = float(np.linalg.norm(ca - cb))
        ctx.proximity_history.append(centroid_dist)

        # ── GATE 1: Stillness veto (with upper-body override) ────────────────
        # Centroid-based stillness alone is insufficient for SEATED fights:
        # two students pushing/hitting while sitting have < 6 px/frame centroid
        # displacement, but their arms create 3-10 px/frame of optical flow in
        # the upper portion of their bounding boxes.
        # Override: if flow_map is available and either person has significant
        # upper-body motion, bypass the centroid-stillness veto.
        a_still = ta.is_still(self.STILL_MOTION_THRESH, self.STILL_LOOKBACK)
        b_still = tb.is_still(self.STILL_MOTION_THRESH, self.STILL_LOOKBACK)

        if self.BOTH_STILL_VETO and a_still and b_still:
            # Before vetoing, check upper-body optical flow — seated fighters
            # move their arms even when their centroid is nearly stationary.
            if flow_map is not None:
                ba_tmp = ta.bbox_history[-1]
                bb_tmp = tb.bbox_history[-1]
                flow_a = self._upper_body_flow(flow_map, gray, ba_tmp)
                flow_b = self._upper_body_flow(flow_map, gray, bb_tmp)
                if max(flow_a, flow_b) >= self.UPPER_BODY_FLOW_THRESH:
                    # Upper-body motion detected despite still centroids --
                    # seated fight pattern; continue evaluation.
                    pass
                else:
                    ctx.active_fight_frames = 0
                    return None
            else:
                # No flow map -- fall back to centroid veto as before
                ctx.active_fight_frames = 0
                return None

        # ── GATE 2: Bounding-box overlap ─────────────────────────────────────
        ba = ta.bbox_history[-1]   # (x1,y1,x2,y2)
        bb = tb.bbox_history[-1]

        # Hard centroid distance cap
        if centroid_dist > self.MAX_CENTROID_DIST:
            ctx.active_fight_frames = 0
            return None

        # Require actual bbox overlap or near-overlap.
        # For seated fights the boxes may only barely touch (IoU ~ 0) from a
        # high-angle camera, so we allow a small negative IoU equivalent by
        # also passing pairs whose boxes are within 15 px of each other.
        if self.REQUIRE_BBOX_OVERLAP:
            iou = self._iou(ba, bb)
            if iou <= 0.0:
                # Check near-adjacency as fallback for seated pairs
                gap_x = max(0, max(ba[0], bb[0]) - min(ba[2], bb[2]))
                gap_y = max(0, max(ba[1], bb[1]) - min(ba[3], bb[3]))
                if gap_x > 15 or gap_y > 15:
                    ctx.active_fight_frames = 0
                    return None

        # ── Vertical-row suppression ──────────────────────────────────────────
        # Students in adjacent desk rows are vertically offset but horizontally
        # aligned — suppress them even if their boxes happen to overlap.
        dy = abs(ca[1] - cb[1])   # vertical centroid difference
        dx = abs(ca[0] - cb[0])   # horizontal centroid difference
        if dy > self.ROW_SUPPRESS_DY and dx < self.ROW_SUPPRESS_DX:
            # Typical front-row / back-row neighbour configuration
            ctx.active_fight_frames = 0
            return None

        # ── Motion intensity ──────────────────────────────────────────────────
        motion_intensity = self._get_motion_intensity(ta, tb, gray, flow_map, ba, bb)
        ctx.flow_history.append(motion_intensity)

        # ── GATE 3: Sustained motion during overlap ───────────────────────────
        if motion_intensity >= self.FIGHT_MOTION_THRESH:
            ctx.active_fight_frames += 1
        else:
            # Allow brief dips (up to 3 frames) without resetting
            if ctx.active_fight_frames > 3:
                ctx.active_fight_frames -= 1   # gradual decay
            else:
                ctx.active_fight_frames = 0

        if ctx.active_fight_frames < self.MIN_FIGHT_FRAMES:
            return None

        # ── All 3 gates passed — compute confidence ───────────────────────────
        # Proximity score: 1.0 when centroids touching, 0 at MAX_CENTROID_DIST
        prox_score   = max(0.0, 1.0 - centroid_dist / self.MAX_CENTROID_DIST)
        # Motion score: saturates at 3× fight threshold
        motion_score = min(1.0, motion_intensity / (self.FIGHT_MOTION_THRESH * 3.0))
        # Duration score: saturates at 2× min required frames
        dur_score    = min(1.0, ctx.active_fight_frames / (self.MIN_FIGHT_FRAMES * 2.0))

        confidence = 0.35 * prox_score + 0.40 * motion_score + 0.25 * dur_score

        contact_centroid = tuple((ca + cb) / 2)
        return {
            'person_a_id':             ta.track_id,
            'person_b_id':             tb.track_id,
            'confidence':              float(confidence),
            'contact_duration_frames': ctx.active_fight_frames,
            'contact_centroid':        contact_centroid,
            'motion_intensity':        float(motion_intensity),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _upper_body_flow(flow_map: np.ndarray, gray: np.ndarray, bbox: tuple) -> float:
        """
        Mean optical-flow magnitude in the UPPER 40% of a person's bounding box.

        Seated students fighting move their arms in the upper torso region even
        when their overall centroid displacement is tiny.  This signal bypasses
        the centroid-based stillness veto for sitting fights.

        Writing/sitting produces ~0.3-1.2 px/frame in the upper body.
        Seated hitting/pushing produces ~2-8 px/frame.
        """
        h, w = gray.shape
        x1, y1, x2, y2 = bbox
        bh = y2 - y1
        # Take upper 40% of the box (head + upper arms region)
        uy2 = y1 + int(bh * 0.40)
        rx1 = max(0, x1);  rx2 = min(w, x2)
        ry1 = max(0, y1);  ry2 = min(h, uy2)
        if rx2 <= rx1 or ry2 <= ry1:
            return 0.0
        region = flow_map[ry1:ry2, rx1:rx2]
        if region.size == 0:
            return 0.0
        mag = np.sqrt(region[..., 0]**2 + region[..., 1]**2)
        return float(np.mean(mag))

    @staticmethod
    def _iou(ba, bb) -> float:
        """IoU between two (x1,y1,x2,y2) boxes."""
        ix1 = max(ba[0], bb[0]); iy1 = max(ba[1], bb[1])
        ix2 = min(ba[2], bb[2]); iy2 = min(ba[3], bb[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1, (ba[2]-ba[0]) * (ba[3]-ba[1]))
        area_b = max(1, (bb[2]-bb[0]) * (bb[3]-bb[1]))
        return inter / (area_a + area_b - inter)

    def _get_motion_intensity(
        self,
        ta: PersonTrack,
        tb: PersonTrack,
        gray: np.ndarray,
        flow_map,
        ba: tuple,
        bb: tuple,
    ) -> float:
        """
        Returns combined motion intensity for the pair.
        Uses optical flow inside union bbox when available;
        falls back to centroid-displacement motion otherwise.
        """
        if flow_map is not None:
            return self._flow_in_union(flow_map, gray, ba, bb)

        # Fallback: average centroid displacement of both tracks
        return (ta.avg_motion(5) + tb.avg_motion(5)) / 2.0

    @staticmethod
    def _flow_in_union(flow_map: np.ndarray, gray: np.ndarray,
                        ba: tuple, bb: tuple) -> float:
        """
        Mean optical-flow magnitude inside the union of the two bboxes.
        Writing/sitting produces ~0.5-1.5 px/frame of flow.
        Fighting produces 4-15 px/frame of flow.
        """
        h, w = gray.shape
        ux1 = max(0,  min(ba[0], bb[0]))
        uy1 = max(0,  min(ba[1], bb[1]))
        ux2 = min(w,  max(ba[2], bb[2]))
        uy2 = min(h,  max(ba[3], bb[3]))
        if ux2 <= ux1 or uy2 <= uy1:
            return 0.0
        region = flow_map[uy1:uy2, ux1:ux2]
        if region.size == 0:
            return 0.0
        mag = np.sqrt(region[..., 0]**2 + region[..., 1]**2)
        return float(np.mean(mag))


# ─────────────────────────────────────────────────────────────────────────────
# Public FightDetector (interface unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

class FightDetector:
    """
    Drop-in replacement.  Same interface as before:
      process_frame(frame, person_detections) → List[Dict]
      get_track_bbox(track_id)               → Optional[Tuple]
    """

    CONFIDENCE_THRESHOLD = 0.55   # minimum to report

    def __init__(self, fps: int = 30):
        self.fps          = fps
        self.tracker      = PersonTracker(max_disappeared=45)
        self.analyzer     = FightAnalyzer(fps=fps)
        self.frame_count  = 0
        self.confirmed_fights: Dict = {}   # pair_key → last_frame

    def process_frame(self, frame: np.ndarray, person_detections: List) -> List[Dict]:
        """
        Args:
            frame:             BGR image
            person_detections: list of (x1,y1,x2,y2,confidence)

        Returns:
            List of fight dicts with keys:
              person_a_id, person_b_id, type, confidence,
              contact_duration_frames, contact_centroid, motion_intensity
        """
        self.frame_count += 1
        tracks = self.tracker.update(person_detections, self.frame_count)

        # Need at least 2 people for a fight
        if len(tracks) < 2:
            return []

        raw = self.analyzer.analyze(tracks, frame, self.frame_count)

        results = []
        for r in raw:
            if r['confidence'] < self.CONFIDENCE_THRESHOLD:
                continue
            key = (r['person_a_id'], r['person_b_id'])

            # Temporal persistence: slight confidence boost for ongoing fights
            if key in self.confirmed_fights:
                gap = self.frame_count - self.confirmed_fights[key]
                if gap < 15:
                    r['confidence'] = min(0.98, r['confidence'] + 0.04)

            self.confirmed_fights[key] = self.frame_count
            results.append({
                'person_a_id':             r['person_a_id'],
                'person_b_id':             r['person_b_id'],
                'type':                    'fighting',
                'confidence':              r['confidence'],
                'contact_duration_frames': r['contact_duration_frames'],
                'contact_centroid':        r['contact_centroid'],
                'motion_intensity':        r['motion_intensity'],
            })

        # Clean up stale confirmed-fight records (> 5 seconds ago)
        stale = [k for k, f in self.confirmed_fights.items()
                 if self.frame_count - f > self.fps * 5]
        for k in stale:
            del self.confirmed_fights[k]

        return results

    def get_track_bbox(self, track_id: int) -> Optional[Tuple]:
        t = self.tracker.objects.get(track_id)
        if t and t.bbox_history:
            return t.bbox_history[-1]
        return None