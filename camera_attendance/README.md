# camera_attendance

Handles camera-based attendance using face recognition and emotion detection. Manages IP cameras, proxies their streams, and records per-student entry/exit events with emotion snapshots.

---

## What it does

- Add and configure IP cameras (RTSP, HTTP/MJPEG, local webcam index)
- Test camera connectivity
- Proxy camera MJPEG streams server-side (avoids browser mixed-content blocks)
- Record attendance via a REST API called by the detection script
- Show per-day attendance logs with emotion data and mood comparison
- Dashboard with live stats and emotion distribution charts
- Serve student face encodings to the detection script

---

## Models

### `Camera`
Represents one physical or virtual camera.

| Field | Description |
|---|---|
| `name` | Human-readable label |
| `url` | Webcam index (`0`, `1`) or HTTP URL (`http://192.168.x.x:8080/video`) |
| `location` | Physical location string |
| `is_active` | Whether detection scripts should use it |

### `CameraAttendanceLog`
One record per student per day. A single record covers both entry and exit (open log pattern).

| Field | Description |
|---|---|
| `student` | FK to `entrance_cam.Student` |
| `camera` | FK to `Camera` |
| `date` | Calendar date |
| `entry_time` | Timestamp when first detected |
| `exit_time` | Timestamp when last detected leaving (null if still inside) |
| `entry_emotion` / `exit_emotion` | Dominant emotion at entry and exit |
| `entry_emotion_score` / `exit_emotion_score` | Confidence 0.0–1.0 |
| `entry_snapshot` / `exit_snapshot` | Cropped face JPEG |
| `mood_comparison` | `improved` / `declined` / `stable` / `unknown` |
| `duration_minutes` | Computed on exit |
| `is_present` | Always `True` once created |

**Open log pattern**: On each detection, the API checks if an open log exists for the student today (entry without exit). If yes → mark exit. If no → create new entry. A `select_for_update()` transaction prevents race conditions when two camera frames are processed simultaneously.

---

## Detection Script (`detection_script_v2.py`)

A standalone Python script that reads from one camera and posts attendance events to the Django API. Run via the management command or directly.

### Pipeline

```
Camera frame
  └─► detect_faces()          — MediaPipe FaceDetection or Haar cascade fallback
        └─► match_student()   — face_recognition (dlib ResNet) distance matching
              └─► ConfirmTracker — require 3 consecutive matching frames before logging
                    └─► EmotionWorker (async thread) — DeepFace emotion analysis
                          └─► log_attendance() — POST to /camera-attendance/api/log/
                                └─► blink_esp32() — optional LED blink on ESP32-CAM
```

### Face Detection

Primary: **MediaPipe FaceDetection** (model_selection=1 = full-range, min_confidence=0.7)
- Returns normalized bounding boxes, converted to pixel coordinates
- Clamps boxes to valid frame dimensions

Fallback (if MediaPipe unavailable): **OpenCV Haar Cascade** (`haarcascade_frontalface_default.xml`)
- Applies `equalizeHist` for better contrast
- Validates detections by aspect ratio (0.75–1.3) and mean pixel value (30–230)

### Face Matching

Uses `face_recognition.face_encodings()` with `num_jitters=2` for better accuracy.

For each detected face:
1. Computes the 128-d embedding
2. Computes L2 distance against every stored student encoding
3. Accepts match if: `best_distance < 0.42` AND `margin >= 0.08` (margin = second_best − best)

The margin check prevents ambiguous matches (when two students look similar) from being logged.

### Multi-frame Confirmation (`ConfirmTracker`)

Prevents false positives from reflections, photos on walls, or partial occlusions. A student is only logged once they appear in **3 consecutive frames**. After logging, they're marked "active" and won't be re-logged until they exit and re-enter.

### Emotion Detection (`EmotionWorker`)

Runs asynchronously in a daemon thread to avoid blocking the main capture loop.

Uses **DeepFace** with `actions=['emotion']`, `enforce_detection=False`.
- Pads the face crop by 30% before analysis
- Maps DeepFace labels to the system's emotion vocabulary (e.g. `contempt` → `disgust`)
- Returns `(emotion_string, confidence_float)` via callback

If DeepFace is unavailable, falls back to `('neutral', 0.0)`.

### MJPEG Handling

For HTTP cameras, uses `MJPEGCapture` — a dedicated thread that reads the MJPEG multipart byte stream, parses JPEG SOI (`\xff\xd8`) / EOI (`\xff\xd9`) markers, and keeps only the latest frame in memory. This avoids buffer buildup and always processes the freshest frame.

`MJPEGRebroadcaster` re-serves the latest frame as MJPEG on `localhost:8765+N` so Django's proxy endpoint can serve it to the browser without opening a second connection to the ESP32-CAM (which supports only one concurrent stream).

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/camera-attendance/api/log/` | POST | Log entry/exit with emotion and snapshot |
| `/camera-attendance/api/students/encodings/` | GET | Return active students with face encodings |
| `/camera-attendance/api/live-detections/` | GET | Return today's attendance logs (used for sync on startup) |

`/api/log/` logic:
1. Validates `student_id` (must be active + have face encoding) and `camera_id`
2. Opens a `select_for_update()` transaction
3. Looks for an open log (entry_time set, exit_time null) for today
4. If found → fills exit fields + computes mood comparison + duration
5. If not found → creates new entry record
6. Decodes optional base64 JPEG snapshot and saves to `entry_snapshot` / `exit_snapshot`

---

## Mood Comparison (`attendance_utils.py`)

```python
POSITIVE_EMOTIONS = {'happy', 'surprise'}
NEGATIVE_EMOTIONS = {'sad', 'angry', 'fear', 'disgust'}

negative_entry + positive_exit → 'improved'
positive_entry + negative_exit → 'declined'
same           → 'stable'
unknown        → 'unknown'
```

---

## Key Dependencies

| Library | Purpose |
|---|---|
| `face_recognition` | Face embedding + distance matching (dlib backend) |
| `mediapipe` | Primary face detection |
| `deepface` + `tf-keras` | Emotion analysis |
| `opencv-python-headless` | Frame capture, image encoding, Haar cascade fallback |
| `numpy` | Embedding math |
| `requests` | API calls from detection script → Django |
