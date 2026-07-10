# Phone Detection Pipeline — Requirements
> Notation: EARS (Easy Approach to Requirements Syntax)

---

## 1. Time-Series State Engine

**REQ-01** — Ubiquitous  
The system SHALL maintain a `DetectionState` instance per active `track_id`, storing three sliding-window `deque` histories: `yolo_phone_history`, `head_down_history`, and `wrist_proximity_history`.

**REQ-02** — Ubiquitous  
The window length for all history deques SHALL be derived from the configured processing FPS using the formula `window = round(fps × 2.5)`, so that at 10 FPS the window covers 25 frames (≈ 2.5 seconds).

**REQ-03** — Ubiquitous  
The system SHALL append a boolean signal to `yolo_phone_history` on every processed frame — `True` when a YOLO or Roboflow phone detection is associated with the person, `False` otherwise.

**REQ-04** — Ubiquitous  
The system SHALL append a boolean to `head_down_history` on every frame, reflecting whether the head-pose estimate for the track is `head_down` or `looking_away`.

**REQ-05** — Ubiquitous  
The system SHALL append a normalised float `[0.0, 1.0]` to `wrist_proximity_history` on every frame, representing the proximity of the closest wrist to the dynamic horizontal centre of the person bounding box, computed via spatial-invariance logic (REQ-08).

**REQ-06** — When a `track_id` has not been observed for longer than `stale_timeout` seconds, the system SHALL automatically delete its `DetectionState` and release associated memory.

**REQ-07** — Ubiquitous  
The system SHALL compute a final `phone_confidence` score using a two-path probabilistic accumulator:
- **Path A** — temporal density of phone object hits: `sum(yolo_phone_history) / window_length`, weighted at **0.75**.
- **Path B** — sustained behavioural anomaly: `head_down_density × wrist_centre_density`, weighted at **0.25**.
- Combined score: `confidence = 0.75 × path_a + 0.25 × path_b`.
- Detection fires when `confidence ≥ PHONE_CONFIRM_THRESHOLD` (default `0.35`).

**REQ-08** — Ubiquitous  
All wrist positional comparisons SHALL use spatial-invariance logic: wrist coordinates are expressed as offsets relative to the person's bounding box centre and normalised by bounding box width and height, not by absolute pixel coordinates or fixed image fractions.

**REQ-09** — Ubiquitous  
The system SHALL NOT use the constants `PHONE_LAP_HEIGHT_FRACTION`, `PHONE_SINGLE_HAND_Y_MIN`, `PHONE_SINGLE_HAND_Y_MAX`, or `WRITING_DESK_Y_MIN` as hard pixel thresholds. All geometry SHALL be relative to the per-person bounding box.

---

## 2. Region-of-Interest (RoI) Cropping

**REQ-10** — Ubiquitous  
For each person track detected by the pose model, the system SHALL extract a padded crop of the person bounding box before passing any frame region to the object detector, with a padding margin of `PAD = 0.15` (15% of bounding box width and height on each side).

**REQ-11** — Ubiquitous  
The crop SHALL be clamped to the frame boundary before extraction, so that `x1_pad ≥ 0`, `y1_pad ≥ 0`, `x2_pad ≤ frame_width`, `y2_pad ≤ frame_height` at all times.

**REQ-12** — Ubiquitous  
Object detections returned from inference on a crop SHALL have their bounding box coordinates mapped back to global frame coordinates before any downstream processing or storage, using the formula: `global_x = crop_x1 + local_x`.

**REQ-13** — When the padded crop area is smaller than `MIN_CROP_PIXELS` (default `32 × 32`), the system SHALL skip object detection for that track on that frame and treat it as a miss.

**REQ-14** — Ubiquitous  
The full-resolution frame SHALL NOT be passed to the object detector for phone or food detection. Only per-person crops SHALL be used.

---

## 3. Network Isolation & Fallbacks

**REQ-15** — When local YOLO object inference on a person crop returns zero phone detections with confidence ≥ `YOLO_PHONE_CONF_THRESHOLD` (default `0.30`), the system SHALL submit only that person's crop to the Roboflow API as a fallback.

**REQ-16** — When local YOLO inference returns one or more phone detections above threshold for a person, the system SHALL NOT call the Roboflow API for that track on that frame.

**REQ-17** — When `ROBOFLOW_API_KEY` is not set in the environment, the system SHALL skip all Roboflow calls silently without raising exceptions.

**REQ-18** — When the Roboflow API call raises any network or HTTP exception, the system SHALL log a warning at `[WARN]` level and treat the result as zero detections, without crashing the detection loop.

**REQ-19** — When a `track_id` is absent from the current frame's pose results and has been absent for longer than `stale_timeout` (default `3.0` seconds), the system SHALL purge its `DetectionState` from the tracking registry.

**REQ-20** — Ubiquitous  
The `DetectionState` registry SHALL be protected by a `threading.Lock` to ensure thread-safe reads and writes between the detection worker and the save worker.
