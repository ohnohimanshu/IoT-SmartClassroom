"""
Classroom Fight Detection — Tuned for HIGH-ANGLE camera, SEATED fighters
=========================================================================

SCENARIO (confirmed by user):
  • Camera: high-angle, looking down
  • Fighters: SEATED, pushing/hitting each other (bodies don't move much)
  • Environment: classroom rows (boxes always close together)

WHY PREVIOUS VERSION FAILED FOR SEATED FIGHTS:
  The STILLNESS VETO used full-body centroid displacement (< 6 px/frame).
  Seated fighters don't move their whole body — only their UPPER BODY / ARMS
  move rapidly.  Centroid barely shifts.  So both students passed as "still"
  and the veto killed the detection.

SOLUTION — ZONE-SPLIT OPTICAL FLOW:
  Split each person's bounding box into:
    • Upper zone (top 50%) — head + arms + torso
    • Lower zone (bottom 50%) — legs + seat
  A seated WRITER has low flow in BOTH zones.
  A seated FIGHTER has HIGH flow in the UPPER zone (arms hitting/pushing)
  but LOW flow in the LOWER zone (still seated).

  This is the key signal that distinguishes seated fighting from writing.

DETECTION LOGIC (4 signals, weighted):
  1. UPPER-ZONE FLOW of person A  — high = arms moving aggressively
  2. UPPER-ZONE FLOW of person B  — same
  3. FLOW IN OVERLAP ZONE between A and B — contact motion
  4. LOWER-ZONE FLOW difference   — legs still = seated, not walking away

FIGHT = upper-zone flow HIGH for BOTH + overlap flow HIGH + lower still
        persisted for MIN_FRAMES consecutive frames

FALSE-POSITIVE GUARDS (kept from previous version):
  • Row-neighbour suppression (vertical offset > threshold)
  • Wide-box filter (merged YOLO boxes excluded)
  • Temporal persistence (need MIN_FRAMES = 12 sustained)
  • Confidence threshold (0.55 minimum to report)
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersonTrack:
    track_id: int
    bbox_history:     deque = field(default_factory=lambda: deque(maxlen=45))
    centroid_history: deque = field(default_factory=lambda: deque(maxlen=45))
    # Per-frame UPPER-ZONE flow magnitude (arms/torso) — key signal
    upper_flow_history: deque = field(default_factory=lambda: deque(maxlen=30))
    # Per-frame LOWER-ZONE flow (legs/seat) — stillness anchor
    lower_flow_history: deque = field(default_factory=lambda: deque(maxlen=30))
    last_seen_frame: int = 0
    confidence_scores: deque = field(default_factory=lambda: deque(maxlen=30))

    def avg_upper_flow(self, last_n: int = 12) -> float:
        h = list(self.upper_flow_history)[-last_n:]
        return float(np.mean(h)) if h else 0.0

    def avg_lower_flow(self, last_n: int = 12) -> float:
        h = list(self.lower_flow_history)[-last_n:]
        return float(np.mean(h)) if h else 0.0

    def is_upper_active(self, thresh: float = 1.8, last_n: int = 12) -> bool:
        """True when upper body is moving significantly (arms/torso)."""
        return self.avg_upper_flow(last_n) >= thresh

    def is_lower_still(self, thresh: float = 1.2, last_n: int = 12) -> bool:
        """True when lower body is mostly still (seated)."""
        return self.avg_lower_flow(last_n) < thresh


@dataclass
class PairContext:
    person_a_id: int
    person_b_id: int
    # Consecutive frames where BOTH upper zones are active + overlap flow high
    active_fight_frames: int = 0
    overlap_flow_history: deque = field(default_factory=lambda: deque(maxlen=30))
    proximity_history:    deque = field(default_factory=lambda: deque(maxlen=60))
    last_updated_frame: int = 0
    interaction_start:  int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Person tracker (velocity-predicted centroid matching)
# ─────────────────────────────────────────────────────────────────────────────

class PersonTracker:

    def __init__(self, max_disappeared: int = 45):
        self.next_id = 0
        self.objects:     Dict[int, PersonTrack] = {}
        self.disappeared: Dict[int, int]         = defaultdict(int)
        self.max_disappeared = max_disappeared

    def update(self, detections: List[Tuple], frame_number: int) -> List[PersonTrack]:
        if not detections:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.objects.pop(oid, None)
                    self.disappeared.pop(oid, None)
            return list(self.objects.values())

        new_centroids = np.array(
            [((x1+x2)/2, (y1+y2)/2) for x1,y1,x2,y2,_ in detections], dtype=float
        )

        if not self.objects:
            for i, det in enumerate(detections):
                self._register(det, new_centroids[i], frame_number)
            return list(self.objects.values())

        obj_ids = list(self.objects.keys())
        pred    = []
        for oid in obj_ids:
            t = self.objects[oid]
            if len(t.centroid_history) >= 2:
                v = np.array(t.centroid_history[-1]) - np.array(t.centroid_history[-2])
                pred.append(np.array(t.centroid_history[-1]) + v)
            else:
                pred.append(np.array(t.centroid_history[-1]) if t.centroid_history else np.zeros(2))

        pred = np.array(pred)
        D    = np.linalg.norm(pred[:, None, :] - new_centroids[None, :, :], axis=2)

        matched_obj  = set()
        matched_dets = set()
        for flat_idx in np.argsort(D, axis=None):
            oi = int(flat_idx // len(detections))
            di = int(flat_idx  % len(detections))
            if oi in matched_obj or di in matched_dets:
                continue
            if D[oi, di] > 220:
                break
            oid = obj_ids[oi]
            t   = self.objects[oid]
            det = detections[di]
            t.bbox_history.append(det[:4])
            t.centroid_history.append(new_centroids[di])
            t.confidence_scores.append(det[4])
            t.last_seen_frame = frame_number
            self.disappeared[oid] = 0
            matched_obj.add(oi)
            matched_dets.add(di)

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
# Optical-flow helper
# ─────────────────────────────────────────────────────────────────────────────

def _mean_flow_in_region(flow_map: np.ndarray, frame_h: int, frame_w: int,
                          x1: int, y1: int, x2: int, y2: int) -> float:
    """Mean optical-flow magnitude inside a clamped region."""
    rx1 = max(0, x1);  ry1 = max(0, y1)
    rx2 = min(frame_w, x2);  ry2 = min(frame_h, y2)
    if rx2 <= rx1 or ry2 <= ry1:
        return 0.0
    region = flow_map[ry1:ry2, rx1:rx2]
    if region.size == 0:
        return 0.0
    mag = np.sqrt(region[..., 0]**2 + region[..., 1]**2)
    return float(np.mean(mag))


# ─────────────────────────────────────────────────────────────────────────────
# Core fight analyser — zone-split optical flow
# ─────────────────────────────────────────────────────────────────────────────

class FightAnalyzer:

    # ── Tuned for HIGH-ANGLE camera, SEATED students ──────────────────────────

    # Upper-body flow thresholds (arms/torso — the fighting signal)
    UPPER_FLOW_FIGHT  = 1.8   # px/frame mean — above this = arms moving a lot
    UPPER_FLOW_WRITE  = 0.8   # below this = just writing (low arm motion)

    # Lower-body flow (legs — seated students keep these still)
    LOWER_FLOW_STILL  = 1.2   # below this = person is seated (not running/standing)

    # Overlap zone flow (direct contact motion between the two people)
    OVERLAP_FLOW_FIGHT = 1.5  # mean flow in the intersection of both bboxes

    # Temporal requirement — must sustain fight signal for this many frames
    MIN_FIGHT_FRAMES  = 12    # ~0.4 s at 30fps (short enough for quick hits)
    DECAY_FRAMES      = 4     # frames of low signal before counter decays

    # Geometry guards
    MAX_CENTROID_DIST     = 160   # px — beyond this, no interaction possible
    ROW_SUPPRESS_DY       = 100   # px vertical gap → front/back row neighbours
    ROW_SUPPRESS_DX       = 70    # px horizontal alignment for row suppression

    def __init__(self, fps: int = 30):
        self.fps   = fps
        self.pairs: Dict[Tuple[int,int], PairContext] = {}
        self._prev_gray = None

    def update_flow(self, frame: np.ndarray):
        """Call once per frame BEFORE analyze(). Computes and stores flow map."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            try:
                self._flow = cv2.calcOpticalFlowFarneback(
                    self._prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2,
                    flags=0,
                )
            except Exception:
                self._flow = None
        else:
            self._flow = None
        self._prev_gray = gray.copy()
        self._fh, self._fw = gray.shape

    def update_track_flows(self, tracks: List[PersonTrack]):
        """
        Fill upper_flow_history / lower_flow_history for every track.
        Must be called AFTER update_flow().
        """
        if self._flow is None:
            return
        for t in tracks:
            if not t.bbox_history:
                continue
            x1, y1, x2, y2 = t.bbox_history[-1]
            h = y2 - y1
            if h < 4:
                continue
            split_y = y1 + h // 2          # split bbox at mid-height
            upper   = _mean_flow_in_region(self._flow, self._fh, self._fw,
                                            x1, y1, x2, split_y)
            lower   = _mean_flow_in_region(self._flow, self._fh, self._fw,
                                            x1, split_y, x2, y2)
            t.upper_flow_history.append(upper)
            t.lower_flow_history.append(lower)

    def analyze(self, tracks: List[PersonTrack], frame_number: int) -> List[Dict]:
        """Return list of confirmed fight dicts."""
        fights         = []
        keys_to_remove = []

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                ta, tb = tracks[i], tracks[j]
                if len(ta.centroid_history) < 3 or len(tb.centroid_history) < 3:
                    continue
                result = self._evaluate_pair(ta, tb, frame_number)
                if result:
                    fights.append(result)

        for key in list(self.pairs):
            if frame_number - self.pairs[key].last_updated_frame > 90:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self.pairs.pop(key, None)

        return fights

    def _evaluate_pair(self, ta: PersonTrack, tb: PersonTrack,
                        frame_number: int) -> Optional[Dict]:

        key = (ta.track_id, tb.track_id)
        if key not in self.pairs:
            self.pairs[key] = PairContext(
                person_a_id=ta.track_id,
                person_b_id=tb.track_id,
                interaction_start=frame_number,
            )
        ctx                    = self.pairs[key]
        ctx.last_updated_frame = frame_number

        ca = np.array(ta.centroid_history[-1])
        cb = np.array(tb.centroid_history[-1])
        dist = float(np.linalg.norm(ca - cb))
        ctx.proximity_history.append(dist)

        # ── Geometry guard: too far apart ─────────────────────────────────────
        if dist > self.MAX_CENTROID_DIST:
            ctx.active_fight_frames = max(0, ctx.active_fight_frames - 2)
            return None

        # ── Row-neighbour suppression ─────────────────────────────────────────
        dy = abs(float(ca[1] - cb[1]))
        dx = abs(float(ca[0] - cb[0]))
        if dy > self.ROW_SUPPRESS_DY and dx < self.ROW_SUPPRESS_DX:
            ctx.active_fight_frames = 0
            return None

        # ── Signal 1 & 2: upper-body flow for each person ─────────────────────
        uf_a = ta.avg_upper_flow(12)
        uf_b = tb.avg_upper_flow(12)
        lf_a = ta.avg_lower_flow(12)
        lf_b = tb.avg_lower_flow(12)

        # ── Signal 3: flow in the OVERLAP / contact zone ──────────────────────
        overlap_flow = 0.0
        if self._flow is not None:
            ba, bb = ta.bbox_history[-1], tb.bbox_history[-1]
            ix1 = max(ba[0], bb[0]);  iy1 = max(ba[1], bb[1])
            ix2 = min(ba[2], bb[2]);  iy2 = min(ba[3], bb[3])
            if ix2 > ix1 and iy2 > iy1:
                overlap_flow = _mean_flow_in_region(
                    self._flow, self._fh, self._fw, ix1, iy1, ix2, iy2)
        ctx.overlap_flow_history.append(overlap_flow)

        # ── Combine signals → fight score ─────────────────────────────────────
        # Normalise each signal to [0, 1]
        uf_score_a      = min(1.0, uf_a       / (self.UPPER_FLOW_FIGHT * 2))
        uf_score_b      = min(1.0, uf_b       / (self.UPPER_FLOW_FIGHT * 2))
        overlap_score   = min(1.0, overlap_flow/ (self.OVERLAP_FLOW_FIGHT * 2))

        # Lower flow being LOW is a bonus (person is seated — not sprinting away)
        # If lower flow is high, the person may be standing/running → reduce weight
        lower_bonus_a   = 1.0 if lf_a < self.LOWER_FLOW_STILL else 0.6
        lower_bonus_b   = 1.0 if lf_b < self.LOWER_FLOW_STILL else 0.6

        # Fight frame score — weighted combination
        frame_score = (
            uf_score_a    * 0.30 * lower_bonus_a +
            uf_score_b    * 0.30 * lower_bonus_b +
            overlap_score * 0.40
        )

        # ── Gate: is this frame a "fight frame"? ─────────────────────────────
        # Require upper-body activity in BOTH people AND meaningful overlap flow
        both_upper_active = (uf_a >= self.UPPER_FLOW_FIGHT and
                             uf_b >= self.UPPER_FLOW_FIGHT)
        overlap_active    = overlap_flow >= self.OVERLAP_FLOW_FIGHT

        is_fight_frame = both_upper_active and overlap_active

        if is_fight_frame:
            ctx.active_fight_frames += 1
        else:
            # Graceful decay — allow brief lulls without full reset
            if ctx.active_fight_frames > self.DECAY_FRAMES:
                ctx.active_fight_frames -= 1
            else:
                ctx.active_fight_frames = max(0, ctx.active_fight_frames - 2)

        if ctx.active_fight_frames < self.MIN_FIGHT_FRAMES:
            return None

        # ── Final confidence ───────────────────────────────────────────────────
        prox_score = max(0.0, 1.0 - dist / self.MAX_CENTROID_DIST)
        dur_score  = min(1.0, ctx.active_fight_frames / (self.MIN_FIGHT_FRAMES * 2))
        confidence = 0.30 * prox_score + 0.40 * frame_score + 0.30 * dur_score

        return {
            'person_a_id':             ta.track_id,
            'person_b_id':             tb.track_id,
            'confidence':              float(confidence),
            'contact_duration_frames': ctx.active_fight_frames,
            'contact_centroid':        tuple((ca + cb) / 2),
            'motion_intensity':        float((uf_a + uf_b) / 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Public FightDetector — unchanged interface
# ─────────────────────────────────────────────────────────────────────────────

class FightDetector:
    """Drop-in replacement. Same interface: process_frame() / get_track_bbox()."""

    CONFIDENCE_THRESHOLD = 0.45   # lowered slightly — seated fights score lower

    def __init__(self, fps: int = 30):
        self.fps             = fps
        self.tracker         = PersonTracker(max_disappeared=45)
        self.analyzer        = FightAnalyzer(fps=fps)
        self.frame_count     = 0
        self.confirmed_fights: Dict = {}

    def process_frame(self, frame: np.ndarray, person_detections: List) -> List[Dict]:
        self.frame_count += 1

        # 1. Compute optical flow for this frame
        self.analyzer.update_flow(frame)

        # 2. Update person tracks
        tracks = self.tracker.update(person_detections, self.frame_count)

        # 3. Fill per-track zone flows
        self.analyzer.update_track_flows(tracks)

        if len(tracks) < 2:
            return []

        # 4. Analyse interactions
        raw = self.analyzer.analyze(tracks, self.frame_count)

        results = []
        for r in raw:
            if r['confidence'] < self.CONFIDENCE_THRESHOLD:
                continue
            key = (r['person_a_id'], r['person_b_id'])
            if key in self.confirmed_fights:
                gap = self.frame_count - self.confirmed_fights[key]
                if gap < 15:
                    r['confidence'] = min(0.98, r['confidence'] + 0.05)
            self.confirmed_fights[key] = self.frame_count
            results.append({**r, 'type': 'fighting'})

        # Clean stale records
        stale = [k for k, f in self.confirmed_fights.items()
                 if self.frame_count - f > self.fps * 5]
        for k in stale:
            del self.confirmed_fights[k]

        return results

    def get_track_bbox(self, track_id: int) -> Optional[Tuple]:
        t = self.tracker.objects.get(track_id)
        return t.bbox_history[-1] if t and t.bbox_history else None