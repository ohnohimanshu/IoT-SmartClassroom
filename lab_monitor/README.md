# lab_monitor

Monitors students during individual lab/computer sessions. Students share their screen and webcam via WebRTC; admins watch all active sessions from a live dashboard. Also captures periodic screenshots and webcam frames with emotion and head pose analysis.

---

## What it does

- Students start/end lab sessions from their portal
- Screen sharing via WebRTC (browser-native, no plugins)
- Webcam frames periodically sent to the server for emotion and head pose analysis
- Screenshots with active tab title captured and stored
- Activity log (active/idle/tab change) recorded per session
- Admin dashboard showing all active sessions with live screen previews and emotion data
- Admin can view any session's WebRTC stream in real time

---

## Models

### `LabSession`
One row per lab session (student + time range).

| Field | Description |
|---|---|
| `student` | FK to `entrance_cam.Student` |
| `start_time` / `end_time` | Session timestamps |
| `duration_minutes` | Calculated on end |
| `is_active` | True while session is running |
| `webrtc_offer` / `webrtc_answer` | SDP offers/answers stored in DB as JSON (used as signaling channel) |
| `webrtc_ice_candidates_student` / `webrtc_ice_candidates_admin` | ICE candidate arrays for peer connection setup |
| `webrtc_screen_stream_id` / `webrtc_camera_stream_id` | Track IDs for demuxing the two streams |

### `Screenshot`
One JPEG per periodic screen capture.

| Field | Description |
|---|---|
| `session` | FK to `LabSession` |
| `image` | Saved to `lab/screenshots/` |
| `tab_title` | Browser tab title at capture time |

### `CameraSnapshot`
One JPEG per webcam frame, with analyzed emotion and head pose.

| Field | Description |
|---|---|
| `session` | FK to `LabSession` |
| `image` | Saved to `lab/camera/` |
| `emotion` | Dominant emotion: `happy/sad/angry/neutral/surprise/fear/disgust/unknown` |
| `emotion_score` | Confidence (0.0–1.0) |
| `pose` | `focused / looking_away / head_down / unknown` |

### `ActivityLog`
Lightweight log of user activity events.

| Field | Description |
|---|---|
| `activity_type` | `active`, `idle`, or `tab_change` |
| `tab_title` | Active tab at log time |

---

## Session Lifecycle

```
Student logs in → student_dashboard → student_session page
  → browser POSTs /lab/session/start/    → creates LabSession, returns session_id
  → browser initiates WebRTC:
      createOffer() → POSTs /lab/api/webrtc/offer/<session_id>/
      Admin polls /lab/api/session/<session_id>/ → sees offer
      Admin browser createAnswer() → POSTs /lab/api/webrtc/answer/<session_id>/
      ICE candidates exchanged via /lab/api/webrtc/ice/<session_id>/
  → screen + camera tracks connected, admin browser renders them live
  → periodic uploads:
      Screenshot (canvas.toDataURL)  → /lab/receive/screenshot/
      Camera frame (webcam canvas)   → /lab/receive/camera/
      Activity events                → /lab/receive/activity/
  → Student ends session → POSTs /lab/session/end/ → LabSession closed
```

WebRTC signaling uses the **database as the signaling server** — no separate websocket or STUN/TURN server is required for LAN deployments. SDP and ICE candidates are written to the DB and polled by the other peer.

---

## Emotion and Head Pose Analysis (`receive_camera_frame`)

Each webcam frame POST goes through:

### Emotion Detection
1. Try **DeepFace** (lazy-loaded, `enforce_detection=False`): returns dominant emotion + confidence
2. Fallback — if DeepFace unavailable: use **MediaPipe FaceLandmarker** (Tasks API, `face_landmarker.task`) to compute:
   - Mouth Aspect Ratio (MAR) from landmarks 13, 14, 78, 308: MAR > 0.25 → `happy`
   - Brow raise from landmarks 107, 336 vs eye landmarks 33, 263: brow_raise < -0.03 → `surprise`
   - Otherwise → `neutral`

### Head Pose
Uses **MediaPipe FaceLandmarker** 468-point mesh. From landmarks:
- Nose (1), left eye (33), right eye (263), chin (152), forehead (10)

Computes:
- `yaw = (nose.x − eye_center.x) / eye_width × 90` — values outside ±18° → `looking_away`
- `pitch = (nose.y − eye_center.y) / face_height × 180` — values < −12 → `head_down`
- Otherwise → `focused`

The `face_landmarker.task` MediaPipe model file must be present at the project root.

---

## Admin API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/lab/api/active-sessions/` | GET | All active sessions with latest screenshot + emotion |
| `/lab/api/session/<id>/` | GET | Full detail for one session including WebRTC state |
| `/lab/api/webrtc/offer/<id>/` | POST | Student submits SDP offer |
| `/lab/api/webrtc/answer/<id>/` | POST | Admin submits SDP answer |
| `/lab/api/webrtc/ice/<id>/` | POST | Either side submits ICE candidates |
| `/lab/api/student-session-webrtc/<id>/` | GET | Student polls for admin's answer + ICE |

---

## Key Dependencies

| Library | Purpose |
|---|---|
| `mediapipe` (Tasks API) | Face landmark detection for pose + fallback emotion |
| `deepface` + `tf-keras` | Primary emotion detection |
| `Pillow` | Image decoding from base64 |
| `numpy` | Landmark math |
| Django built-ins | Authentication, file upload, JSON field |
| WebRTC (browser) | Screen sharing + webcam streaming (no server-side video processing) |
