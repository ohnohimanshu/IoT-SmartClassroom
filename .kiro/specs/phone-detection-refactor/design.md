# Phone Detection Pipeline — Architectural Design

---

## 1. Overview

The refactored pipeline replaces brittle full-frame spatial heuristics with a three-layer architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│  ProductionStreamProcessor._process_single_frame()              │
│                                                                 │
│  ┌─────────────────┐     ┌──────────────────────────────────┐   │
│  │  Pose Model      │────▶│  RoI Crop Extractor              │   │
│  │  (YOLO Pose)     │     │  pad=15%, clamp to frame bounds  │   │
│  └─────────────────┘     └──────────────┬───────────────────┘   │
│                                          │  per-person crop      │
│                           ┌─────────────▼───────────────────┐   │
│                           │  Object Detector (YOLO / RF)     │   │
│                           │  - YOLO first on crop            │   │
│                           │  - Roboflow fallback on crop     │   │
│                           │    only when YOLO misses         │   │
│                           └─────────────┬───────────────────┘   │
│                                          │  local detections     │
│                           ┌─────────────▼───────────────────┐   │
│                           │  Coordinate Remapper             │   │
│                           │  local → global frame coords     │   │
│                           └─────────────┬───────────────────┘   │
│                                          │  global detections    │
│                           ┌─────────────▼───────────────────┐   │
│                           │  PhoneDetector.detect()          │   │
│                           │  (time-series state engine)      │   │
│                           └─────────────┬───────────────────┘   │
│                                          │  (is_phone, conf)     │
│                           ┌─────────────▼───────────────────┐   │
│                           │  TemporalBehaviorEngine          │   │
│                           │  evaluate_final_behavior()       │   │
│                           └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Breakdown

### 2.1 `DetectionState` (new class in `phone_detection.py`)

Owns all per-track sliding-window history. Lives inside a `PhoneDetector`-level registry keyed by `track_id`.

```python
@dataclass
class DetectionState:
    track_id:               int
    window_length:          int                    # e.g. 25 at 10 FPS
    yolo_phone_history:     deque[bool]
    head_down_history:      deque[bool]
    wrist_proximity_history: deque[float]          # 0.0–1.0, spatial-invariant
    last_seen:              float                  # monotonic timestamp
```

**Confidence accumulator (called once per frame per track):**

```
path_a = sum(yolo_phone_history) / window_length          # [0, 1]
head_density  = sum(head_down_history) / window_length    # [0, 1]
wrist_density = mean(wrist_proximity_history)             # [0, 1]
path_b = head_density * wrist_density                     # [0, 1]
confidence = 0.75 * path_a + 0.25 * path_b
is_phone = confidence >= PHONE_CONFIRM_THRESHOLD (0.35)
```

Path A dominates: a real phone appears in YOLO/Roboflow repeatedly.  
Path B catches phone-in-lap with no object hit but strong behavioural signal.

---

### 2.2 Spatial-Invariance Wrist Logic

Replace all absolute threshold constants with per-person relative geometry:

| Old (brittle)                        | New (spatial-invariant)                                  |
|--------------------------------------|----------------------------------------------------------|
| `wrist.y > y1 + bbox_h * 0.55`       | `rel_y = (wrist.y - y1) / bbox_h`                        |
| `PHONE_LAP_HEIGHT_FRACTION = 0.55`   | `wrist_centre_proximity = 1 - abs(rel_x - 0.5) / 0.5`  |
| `WRITING_DESK_Y_MIN = 0.70`          | `rel_x = (wrist.x - x1) / bbox_w`                       |
| Fixed pixel spread threshold         | `spread_norm = dist(left_wrist, right_wrist) / bbox_h`  |

`wrist_centre_proximity` is the signal stored in `wrist_proximity_history`. A value near 1.0 means the wrist is centred on the body (holding something in front). A value near 0.0 means the wrist is at the body edge (writing at a desk).

---

### 2.3 RoI Crop Extractor (new method in `ProductionStreamProcessor`)

```python
def _extract_person_crop(self, frame, x1, y1, x2, y2, pad=0.15):
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px1 = max(0, int(x1 - bw * pad))
    py1 = max(0, int(y1 - bh * pad))
    px2 = min(w, int(x2 + bw * pad))
    py2 = min(h, int(y2 + bh * pad))
    return frame[py1:py2, px1:px2], (px1, py1)   # crop + origin offset
```

The returned `(px1, py1)` origin is used to remap local detection coords back to global:

```python
def _remap_to_global(self, dets, origin):
    ox, oy = origin
    return [(x1+ox, y1+oy, x2+ox, y2+oy, conf) for (x1,y1,x2,y2,conf) in dets]
```

---

### 2.4 Refactored `_parse_object_detections`

Old signature: `_parse_object_detections(frame)` — runs on full frame, Roboflow on full frame.

New signature: `_parse_object_detections_for_person(crop, origin, track_id)` — per-person crop only.

```
for each person_track:
    crop, origin = _extract_person_crop(frame, bbox, pad=0.15)
    if crop too small → skip
    local_phone_dets = YOLO object model on crop (cls=67, conf≥0.30)
    if not local_phone_dets:
        local_phone_dets = roboflow_fallback(crop)   ← crop, not full frame
    global_phone_dets = _remap_to_global(local_phone_dets, origin)
    local_food_dets  = YOLO object model on crop (food cls)
    global_food_dets = _remap_to_global(local_food_dets, origin)
    → pass global coords to PhoneDetector / FoodDetector
```

This prevents small-object downscaling loss because the object model sees a large crop of one person rather than a tiny phone in a wide classroom shot.

---

### 2.5 Memory Cleanup

`PhoneDetector` holds the `DetectionState` registry. Cleanup is triggered from `TemporalBehaviorEngine.cleanup_stale()` which already runs each frame. We extend it to also call `PhoneDetector.cleanup_stale(current_time)`:

```python
def cleanup_stale(self, current_time: float, stale_timeout: float = 3.0):
    with self._lock:
        expired = [tid for tid, s in self._states.items()
                   if current_time - s.last_seen > stale_timeout]
        for tid in expired:
            del self._states[tid]
```

---

## 3. Data-Flow Summary

```
frame
 └─ pose model → person_tracks [(tid, x1,y1,x2,y2, conf, kp)]
      └─ for each track:
           ├─ extract padded crop
           ├─ YOLO object on crop → local phone/food/book dets
           │    └─ if no phone → Roboflow on same crop
           ├─ remap local → global coords
           ├─ PhoneDetector.detect(person, global_phone_dets, ...)
           │    ├─ update DetectionState histories
           │    └─ compute path_a + path_b confidence → (is_phone, conf)
           └─ behavior evaluation → DetectionResult
```

---

## 4. Key Configuration Constants (all in `phone_detection.py`)

| Constant                  | Default | Meaning                                         |
|---------------------------|---------|-------------------------------------------------|
| `PHONE_CONFIRM_THRESHOLD` | `0.35`  | Min combined confidence to fire phone alert     |
| `PATH_A_WEIGHT`           | `0.75`  | Weight for YOLO/RF temporal density             |
| `PATH_B_WEIGHT`           | `0.25`  | Weight for head+wrist behavioural signal        |
| `WINDOW_SECONDS`          | `2.5`   | Sliding window duration                         |
| `STALE_TIMEOUT`           | `3.0`   | Seconds before DetectionState is purged         |
| `ROI_PAD`                 | `0.15`  | Fractional bounding box padding for crop        |
| `MIN_CROP_PIXELS`         | `32`    | Min crop dimension before skipping inference    |
| `YOLO_PHONE_CONF`         | `0.30`  | Min YOLO phone confidence to count as a hit     |

---

## 5. Backward Compatibility

- `ClassroomBehaviorDetector.detect(frame)` public API is unchanged.
- `DetectionResult` dataclass is unchanged.
- `TemporalBehaviorEngine` is unchanged except for a new `cleanup_stale` call delegating to `PhoneDetector`.
- `_parse_object_detections(frame)` is kept as a thin wrapper calling the new per-person method, so any code that calls it directly still works.
