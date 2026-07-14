# entrance_cam

The core app of the system. Manages students, ESP32 fingerprint devices, and serves as the central authentication + attendance hub. It also exposes the REST API that ESP32 hardware communicates with.

---

## What it does

- Student CRUD (create, read, update, delete) with photo upload
- Automatic face encoding generation on photo save (via Django signals)
- Automatic Django `User` account creation per student (email = username, branch+roll = default password)
- ESP32 fingerprint scanner device management
- Fingerprint enrollment workflow (admin triggers → ESP32 polls command → enrolls on device → reports result back)
- Combined attendance log (camera + fingerprint) in one view
- Admin account management (superuser-only)
- REST API for ESP32 devices to poll commands, report enrollment results, upload fingerprint images, and mark attendance

---

## Models

### `Student`
Central model — every other app references this.

| Field | Description |
|---|---|
| `name`, `roll_no`, `email` | Basic identity |
| `course`, `branch`, `year` | Academic info |
| `photo` | Uploaded student photo |
| `face_encoding` | JSON array of 128 floats — generated automatically by `face_recognition` when a photo is saved |
| `is_enrolled` | `True` only when `face_encoding` was successfully generated (ready for camera attendance) |
| `fingerprint_id` | Slot number (1–127) assigned on the AS608/R307 sensor |
| `fp_confidence`, `fp_scan_count`, `fp_last_seen` | Stats updated on every fingerprint scan |
| `fp_image` | Grayscale PNG reconstructed from the 4-bit raw image sent by ESP32 |
| `user` | Linked Django `User` (auto-created via signal) |

### `ESP32Device`
Represents one physical ESP32 fingerprint scanner.

| Field | Description |
|---|---|
| `ip_address` | Device IP or server URL |
| `api_key` | Per-device secret for `X-API-Key` authentication |
| `enrollment_mode` | Flag set while a student is mid-enrollment |
| `pending_fingerprint_id` | The slot being enrolled |
| `last_seen` | Updated on every ESP32 API call; used to show online/offline status (online = last seen < 120s ago) |

### `FingerprintAttendance`
One record per fingerprint scan event.

| Field | Description |
|---|---|
| `attendance_type` | `entry` or `exit` (alternates automatically) |
| `fingerprint_id` | Which sensor slot matched |
| `confidence` | Match score returned by AS608/R307 sensor (0–255) |
| `duration_minutes` | Calculated on exit by looking back for the paired entry |

---

## Face Encoding — How It Works

`signals.py` listens to `Student.post_save`. When a student is saved with a photo:

1. Loads the image with `face_recognition.load_image_file()`
2. Calls `face_recognition.face_encodings()` — produces a 128-dimensional HOG+deep-metric embedding (dlib's ResNet under the hood)
3. Stores the embedding as a JSON string in `face_encoding`
4. Sets `is_enrolled = True` if successful

This runs synchronously in the Django request cycle on save. For batch re-encoding, use the management command `generate_face_encodings`.

The `face_recognition` library uses **dlib's 68-point shape predictor** to detect face landmarks and a **ResNet-based metric-learning network** (trained on a large face dataset) to produce the 128-d embedding. Distance threshold used during matching (in `camera_attendance`) is 0.42.

---

## Fingerprint Enrollment Flow

```
Admin clicks Enroll → fingerprint_id slot assigned →
  cache.set('esp32_command', {ENROLL, slot}) →
    ESP32 polls /api/esp32/command/ every 2s →
      Receives ENROLL command, prompts student →
        Student places finger twice →
          ESP32 POSTs /api/esp32/enroll-result/ {success: true/false} →
            If success → fingerprint_id stays, enrollment is complete
            If fail    → fingerprint_id cleared
          ESP32 optionally POSTs /api/esp32/upload-image/ with 4-bit grayscale PNG
            → Decoded to 256×288 grayscale PNG, saved to student.fp_image
```

The command queue uses Django's cache (`cache.set/get`) rather than a database field, so it's ephemeral and doesn't require a DB write on every ESP32 poll.

---

## Attendance Marking Flow (Fingerprint)

```
Student places finger → ESP32 matches template locally →
  ESP32 POSTs /api/mark-attendance/ {fingerprint_id, confidence} →
    Django looks up Student by fingerprint_id →
      Checks last FingerprintAttendance for today →
        If last = entry  → record exit, calculate duration
        If last = exit or none → record entry
      Updates student.fp_confidence / fp_scan_count / fp_last_seen
      Returns {name, roll_no, action: IN/OUT, time, confidence}
```

---

## ESP32 Authentication

All `/api/` routes use the `esp32_auth` decorator which checks the `X-API-Key` header. A key is accepted if it matches either:
- The global `ESP32_API_KEY` environment variable, or
- Any `ESP32Device.api_key` in the database

This lets you give each physical device its own key.

---

## Key Dependencies

| Library | Purpose |
|---|---|
| `face_recognition` | Face embedding generation (dlib backend) |
| `django` | Web framework, ORM, signals |
| `Pillow` | Image validation on upload |
| `zlib` + `struct` | Pure-Python PNG encoder for fingerprint images (no Pillow needed at runtime for this) |
