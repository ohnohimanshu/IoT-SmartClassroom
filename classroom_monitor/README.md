# classroom_monitor

Real-time classroom behavior analysis. Processes live camera feeds to detect student engagement, phone usage, eating, fighting, and hand raises. Logs incidents and sends email alerts for critical behaviors.

---

## What it does

- Live MJPEG stream with behavior detection overlays rendered directly on frames
- Per-student behavior classification with temporal smoothing
- Incident reporting (phone, eating, fighting) with snapshot + email alert
- Engagement snapshots saved periodically (focused/distracted/phone/eating counts + score)
- Session management (start/end monitoring sessions per camera)
- Video upload and post-session frame analysis
- Face recognition to identify which student is causing an incident
- Admin review queue for incidents

---

## Models

### `ClassroomCamera`
One physical or virtual camera for classroom monitoring. Separate from `camera_attendance.Camera`.

### `ClassSession`
A monitoring session — started/ended by admin. One active session per camera at a time.

### `EngagementSnapshot`
Periodic aggregate (every 10s) of behavior counts for a session. Stores focused/distracted/phone/eating counts and an engagement score (focused / total × 100).

### `StudentZoneLog`
Per-student behavior record within an `EngagementSnapshot`. One row per detected person per snapshot.

### `ClassroomVideo`
Uploaded video for offline post-analysis.

### `VideoAnalysisFrame` / `VideoStudentZone`
Per-frame results from offline video analysis.

### `IncidentReport`
One record per confirmed behavior incident.

| Field | Description |
|---|---|
| `incident_type` | `using_phone`, `eating_food`, `fighting`, `distracted`, `looking_away`, `head_down` |
| `severity` | `low` / `medium` / `high` / `critical` |
| `snapshot` | JPEG of the frame at detection time |
| `confidence` | Detection confidence (0.0–1.0) |
| `whatsapp_sent` | Legacy field (WhatsApp integration was replaced by email) |
| `is_reviewed` | Admin has reviewed this incident |

---

## Detection Pipeline

### Entry Point — `_generate_video_stream()` in `views.py`

The live stream view drives a 3-thread pipeline:

```
Camera frame (cv2.VideoCapture)
  │
  ├─► pose_q  (every ~3 frames, ~8fps)   → _pose_worker
  │     └─► _parse_pose_detections()      — keeps ByteTrack IDs stable
  │
  └─► detect_q (every ~12 frames, ~2fps) → _detection_worker
        └─► detector.detect(frame)
              └─► ProductionStreamProcessor._process_single_frame()
                    ├─► _parse_pose_detections()  — YOLO pose + ByteTrack
                    ├─► _parse_object_detections() — YOLO object (phone/food/book)
                    ├─► fight_detector.predict()   — 3D CNN fight detection
                    └─► _run_behavior_evaluation() — per-person classification
```

Results flow back to the main thread via `result_lock`. The main thread annotates frames with colored bounding boxes and streams them as MJPEG.

---

## YOLO Models

Two YOLO11s models are loaded once (shared singleton `_SharedYOLOModels`):

| Model | File | Purpose |
|---|---|---|
| `YOLO('yolo11s-pose.pt')` | `yolo11s.pt` | Detect people + 17 body keypoints (COCO skeleton) |
| `YOLO('yolo11s.pt')` | `yolo11s.pt` | Detect objects: phone (class 67), food classes (46–55), book (73) |

Both run via [Ultralytics YOLO11](https://docs.ultralytics.com/). The pose model is run with `persist=True` to enable **ByteTrack** multi-object tracking — this assigns consistent `track_id`s across frames so per-person behavior history accumulates correctly.

Optional: a locally-trained classroom-specific phone model (`classroom_phone_yolo.pt`) can be placed in `classroom_monitor/model_weights/`. If found, it's used as a primary source and merged with COCO results by IoU deduplication.

---

## Per-Person Behavior Classification

Each tracked person gets a `TrackedPerson` object (identified by ByteTrack `track_id`) with:
- `behavior_history` — deque of last 20 raw behavior labels
- `keypoint_history` — deque of last 30 keypoint arrays + timestamps
- `last_final_behavior` — last confirmed label

Classification runs in priority order each frame:

```
1. Fighting?          → FightDetector3DCNN (if confirmed over 3+ frames)
2. Hand raised?       → HandRaiseDetector  (wrist above nose, elbow raised)
3. Using phone?       → PhoneDetector      (3 paths: YOLO object, face-phone posture, lap posture)
4. Eating?            → FoodDetector       (YOLO food bbox near mouth)
5. Head pose          → HeadPoseDetector   → focused / looking_away / head_down / not_visible
```

---

## `HeadPoseDetector`

Uses YOLO keypoints 0 (nose), 1 (left eye), 2 (right eye).

- **Two eyes visible**: computes inter-eye distance, measures nose offset (yaw) and nose drop below eye line (pitch). Thresholds: yaw\_ratio > 0.45 → `looking_away`; drop\_ratio > 0.45 → `head_down`; very small inter-eye distance (< 7% of bbox height) → `looking_away`
- **One eye visible**: estimates inter-eye distance from bbox width × 0.12, applies yaw threshold 0.45
- **No confident keypoints**: increments consecutive low-confidence counter; after 2 frames → `head_down`

---

## `PhoneDetector`

Three independent detection paths, evaluated in order:

**Path 1 — YOLO object evidence**
- Phone bbox (class 67 or local model) must pass confidence ≥ 0.35, size sanity check (< 70% of person height, > 3%), and be inside or near the person's bounding box
- Then: must be within 35% of bbox height from any wrist or elbow keypoint (indices 7, 8, 9, 10)
- High-confidence override: conf ≥ 0.55 + phone bbox overlaps person > 50% → accept without wrist match

**Path 2 — Lap posture heuristic** (no YOLO object)
- Requires `head_is_down = True`
- Suppressed if `calculate_wrist_motion_variance()` detects writing motion (variance > 300)
- Suppressed if `wrist_motion_available_and_variance()` shows variance > 60 (wrist not still enough)
- Single hand in mid-torso zone (relative y ∈ [0.45, 0.70], x-offset < 35% of width, not near book) → `using_phone` at 0.60
- Two cupped hands at lap level (both below 55% height, spread < 22% of height) → `using_phone` at 0.55

**Path 3 — Phone to face/ear posture** (no YOLO object)
- Does NOT require head_down (phone held up = head level or tilted back)
- For each arm: wrist within ±12% of nose height (above) or ±42% (below), x-offset < 30% of bbox width, wrist above elbow by ≥ 2% of bbox height
- → `using_phone` at 0.58

---

## `TemporalBehaviorEngine` — Smoothing

Prevents single-frame noise from generating alerts. Each behavior type has a confirmation window:

| Behavior | Window | Required majority |
|---|---|---|
| `using_phone`, `eating_food`, `fighting` (ALERT_POSES) | 3 frames | 67% |
| `hand_raised` | 3 frames | 60% |
| Normal (focused / distracted / not_visible) | 3 frames | 100% (all same) |

For normal behaviors, a 8-frame sliding window with > 50% majority is also applied as a fallback. ALERT_POSES labels can only be promoted via the strict alert window, never via the normal/majority path.

---

## Fight Detection — `FightDetector3DCNN`

Architecture: **mc3_18** (Mixed Convolutional 3D ResNet-18), pre-trained on Kinetics-400, with a custom 2-class head fine-tuned on a surveillance fight dataset.

Process:
1. Every frame is preprocessed: resize to 112×112, BGR→RGB, normalize with Kinetics-400 mean/std
2. Frames accumulate in a 16-frame ring buffer
3. Every 8 new frames: run inference — `(C=3, T=16, H=112, W=112)` tensor → softmax → fight probability
4. If `fight_prob >= 0.60` → `fight_detected = True`

Then in `ProductionStreamProcessor`:
5. `_fight_streak` counter: only triggers alert after **3 consecutive positive frames**
6. `_localize_fight_participants()`: identifies which track IDs are fighting via spatial region or proximity + keypoint motion heuristic

If the fine-tuned weights file (`fight_mc3_18_finetuned.pth`) is not present, fight detection is disabled (returns `False, 0.0` always).

Falls back to `r3d_18` if `mc3_18` is unavailable. Model is a **process-wide singleton** — all detector instances share it.

---

## Incident Saving and Alerts

When an alert behavior is confirmed, `_save_incident_direct()` is called from a background `save_worker` thread (non-blocking for the stream):

1. Encodes the current frame as JPEG
2. Creates an `IncidentReport` DB record
3. If the incident type is in `EMAIL_ALERT_TYPES` (`using_phone`, `eating_food`, `fighting`), sends an SMTP email alert with the snapshot attached

Cooldown: 90 seconds per `(incident_type, track_id)` pair prevents alert storms for the same student.

---

## Face Recognition in Stream (`face_recognition_helper.py`)

`StudentFaceRecognizer` loads all enrolled students' face encodings from the DB and runs dlib recognition every 5 seconds (configurable `FACEREC_INTERVAL`). Matched students are cached by `track_id` so the name appears on subsequent frames without re-running recognition.

Uses `DLIB_LOCK` (threading.Lock) to ensure dlib is never called concurrently from different threads (dlib is not thread-safe).

---

## Key Dependencies

| Library | Purpose |
|---|---|
| `ultralytics` (YOLO11) | Person detection, pose keypoints, object detection, ByteTrack tracking |
| `torch` + `torchvision` | 3D CNN fight detection (mc3_18/r3d_18) |
| `opencv-python-headless` | Frame capture, encoding, annotation |
| `face_recognition` (dlib) | Student identity from face in incidents |
| `numpy` | Keypoint math |
| `inference-sdk` | Optional Roboflow cloud phone model |
