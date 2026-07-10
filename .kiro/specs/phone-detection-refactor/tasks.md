# Phone Detection Pipeline — Implementation Tasks

Review design.md and requirements.md before starting. Complete tasks in order — each builds on the previous.

---

## Task 1 — Add `DetectionState` class to `phone_detection.py`

**File:** `classroom_monitor/phone_detection.py`

1. Add imports: `from collections import deque`, `from dataclasses import dataclass, field`, `import threading`, `import time`.
2. Add the `DetectionState` dataclass above `PhoneDetector`:
   ```python
   @dataclass
   class DetectionState:
       track_id: int
       window_length: int
       yolo_phone_history:      deque = field(default_factory=deque)
       head_down_history:       deque = field(default_factory=deque)
       wrist_proximity_history: deque = field(default_factory=deque)
       last_seen: float = 0.0
   ```
3. In `__init__`, initialise each deque with `maxlen=window_length` from a class-level `WINDOW_SECONDS = 2.5` and a passed-in `process_fps` argument (default `10`).

**Acceptance:** `DetectionState(track_id=1, window_length=25)` constructs without error. Each deque has `maxlen=25`.

---

## Task 2 — Add `DetectionState` registry to `PhoneDetector`

**File:** `classroom_monitor/phone_detection.py`

1. Add class-level constants:
   ```python
   PHONE_CONFIRM_THRESHOLD = 0.35
   PATH_A_WEIGHT  = 0.75
   PATH_B_WEIGHT  = 0.25
   WINDOW_SECONDS = 2.5
   STALE_TIMEOUT  = 3.0
   YOLO_PHONE_CONF = 0.30
   ```
2. In `PhoneDetector.__init__(self, process_fps=10)`:
   - Store `self._process_fps = process_fps`
   - Store `self._window_length = round(process_fps * self.WINDOW_SECONDS)`
   - Initialise `self._states: dict[int, DetectionState] = {}`
   - Initialise `self._lock = threading.Lock()`
3. Add method `_get_or_create_state(self, track_id) -> DetectionState` that creates a new `DetectionState` with correct `window_length` if the `track_id` is not yet in `self._states`.
4. Add method `cleanup_stale(self, current_time: float)` that deletes states where `current_time - state.last_seen > self.STALE_TIMEOUT`.

**Acceptance:** Calling `_get_or_create_state(5)` twice returns the same object. `cleanup_stale` removes states older than `STALE_TIMEOUT`.

---

## Task 3 — Replace heuristic constants with spatial-invariance logic

**File:** `classroom_monitor/phone_detection.py`

1. Remove class-level constants: `PHONE_LAP_HEIGHT_FRACTION`, `PHONE_CUPPED_SPREAD_MAX`, `PHONE_SINGLE_HAND_Y_MIN`, `PHONE_SINGLE_HAND_Y_MAX`, `WRITING_DESK_Y_MIN`.
2. Add a private static method `_wrist_centre_proximity(wrist, x1, y1, x2, y2) -> float`:
   - Computes `rel_x = (wrist[0] - x1) / max(x2 - x1, 1)`
   - Returns `1.0 - abs(rel_x - 0.5) * 2.0` clamped to `[0.0, 1.0]`.
   - A value near 1.0 = wrist centred on body (holding object); near 0.0 = at body edge.
3. Rewrite Path 2 (heuristic) in `detect_phone_usage` to use only `rel_y = (wrist.y - y1) / bbox_h` and `_wrist_centre_proximity` — no absolute pixel fractions.

**Acceptance:** With a person bbox `(100, 100, 200, 300)` and wrist at `(150, 200)`, `_wrist_centre_proximity` returns `1.0` (perfectly centred). With wrist at `(105, 200)` it returns near `0.1` (at edge).

---

## Task 4 — Wire history updates into `detect_phone_usage`

**File:** `classroom_monitor/phone_detection.py`

1. Add `track_id: int` and `timestamp: float` as required parameters to `detect_phone_usage`.
2. At the start of `detect_phone_usage`:
   - Call `state = self._get_or_create_state(track_id)`
   - Update `state.last_seen = timestamp`
3. After evaluating `head_is_down`, append to `state.head_down_history`.
4. After evaluating best wrist proximity, append the float to `state.wrist_proximity_history`.
5. After Path 1 (YOLO hit) evaluation append `True` to `state.yolo_phone_history` on a hit, `False` otherwise.
6. After Path 2 (heuristic) replace the immediate `return True, conf` with: append `True` to `yolo_phone_history` and fall through to the accumulator.
7. Replace both final `return` statements with the two-path accumulator:
   ```python
   window = state.window_length or 1
   path_a = sum(state.yolo_phone_history) / window
   head_d = sum(state.head_down_history) / window
   wrist_d = (sum(state.wrist_proximity_history) / len(state.wrist_proximity_history)
              if state.wrist_proximity_history else 0.0)
   path_b = head_d * wrist_d
   confidence = self.PATH_A_WEIGHT * path_a + self.PATH_B_WEIGHT * path_b
   return confidence >= self.PHONE_CONFIRM_THRESHOLD, round(confidence, 3)
   ```

**Acceptance:** On a fresh state (all histories empty) `detect_phone_usage` returns `(False, 0.0)`. After injecting 15 `True`s into `yolo_phone_history` of a 25-length window, it returns `(True, ≈0.45)`.

---

## Task 5 — Add RoI crop extractor to `ProductionStreamProcessor`

**File:** `classroom_monitor/behavior_detection.py`

1. Add method `_extract_person_crop(self, frame, x1, y1, x2, y2, pad=0.15)`:
   - Compute padded coords, clamp to frame dims.
   - If `(px2 - px1) < 32` or `(py2 - py1) < 32`, return `None, None`.
   - Return `(crop_ndarray, (px1, py1))`.
2. Add method `_remap_to_global(self, dets, origin)`:
   - `origin = (ox, oy)`
   - Return `[(x1+ox, y1+oy, x2+ox, y2+oy, conf) for x1,y1,x2,y2,conf in dets]`.

**Acceptance:** Given a 640×480 frame and a person bbox `(10, 10, 110, 210)` with `pad=0.15`, the returned crop should start at `(0, 0)` (clamped) and a detection at local `(20, 30, 50, 60, 0.8)` should remap to global `(20, 30, 50, 60, 0.8)` (origin `(0,0)` in this case).

---

## Task 6 — Refactor `_parse_object_detections` to per-person RoI

**File:** `classroom_monitor/behavior_detection.py`

1. Add new method `_parse_object_detections_for_person(self, crop, origin, track_id)` that:
   - Runs YOLO object model on `crop` for phone cls `{67}`, food cls, book cls.
   - If zero phone detections: runs Roboflow fallback on `crop` only (move existing Roboflow block here, scoped to `crop`).
   - Remaps all detections to global coords via `_remap_to_global`.
   - Returns `(phone_dets, food_dets, book_dets)` in global coords.
2. Refactor `_process_single_frame` to:
   - Call `_parse_pose_detections(frame)` as before.
   - For each person track, call `_extract_person_crop` then `_parse_object_detections_for_person`.
   - Accumulate all global detections before passing to `_run_behavior_evaluation`.
3. Keep the old `_parse_object_detections(frame)` method as a deprecated wrapper (calls new method with full-frame crop and origin `(0,0)`) so external callers don't break.

**Acceptance:** With a single person in frame, the object model is called once (on the crop), not once on the full frame.

---

## Task 7 — Pass `track_id` and `timestamp` through the call chain

**File:** `classroom_monitor/behavior_detection.py`

1. In `_run_behavior_evaluation`, when calling `self.phone_detector.detect_phone_usage(...)`, pass `track_id=tid` and `timestamp=timestamp`.
2. Update the `detect_phone_usage` signature to accept these new parameters (done in Task 4).
3. After `_run_behavior_evaluation`, call `self.phone_detector.cleanup_stale(timestamp)`.

**Acceptance:** No `TypeError` on calling `detect_phone_usage` with the updated signature.

---

## Task 8 — Integrate `PhoneDetector.cleanup_stale` with `TemporalBehaviorEngine`

**File:** `classroom_monitor/behavior_detection.py` and `classroom_monitor/behavior_detection_core.py`

1. In `ProductionStreamProcessor.__init__`, store a reference to `self.phone_detector` (already exists as `self.processor.phone_detector` in `ClassroomBehaviorDetector`).
2. At the end of `_process_single_frame`, after `self.behavior_engine.cleanup_stale(timestamp)`, call `self.phone_detector.cleanup_stale(timestamp)`.
3. Ensure `cleanup_stale` on `PhoneDetector` is thread-safe via `self._lock`.

**Acceptance:** After a `track_id` disappears from frames for 4+ seconds, its `DetectionState` is no longer in `self.phone_detector._states`.

---

## Task 9 — Update `ClassroomBehaviorDetector` to pass `process_fps` to `PhoneDetector`

**File:** `classroom_monitor/behavior_detection.py`

1. In `ProductionStreamProcessor.__init__`, pass `process_fps` to `PhoneDetector(process_fps=process_fps)`.
2. Verify `window_length = round(process_fps * 2.5)` is correct at init.

**Acceptance:** `ProductionStreamProcessor(process_fps=10).phone_detector._window_length == 25`. `ProductionStreamProcessor(process_fps=5).phone_detector._window_length == 12`.

---

## Task 10 — Remove dead code and validate end-to-end

**Files:** `classroom_monitor/phone_detection.py`, `classroom_monitor/behavior_detection.py`

1. Remove the now-unused old constants: `PHONE_LAP_HEIGHT_FRACTION`, `PHONE_CUPPED_SPREAD_MAX`, `PHONE_SINGLE_HAND_Y_MIN`, `PHONE_SINGLE_HAND_Y_MAX`, `WRITING_DESK_Y_MIN` from `PhoneDetector`.
2. Remove the Roboflow block from the old `_parse_object_detections(frame)` method (it is now in `_parse_object_detections_for_person`).
3. Run `getDiagnostics` on both files and fix all reported issues.
4. Manually trace through the data flow with a mock `TrackedPerson` to verify path A and path B scores produce expected results.

**Acceptance:** No diagnostics errors. `ClassroomBehaviorDetector().detect(mock_frame)` returns a list of dicts without exceptions.
