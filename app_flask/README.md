# Flask Fingerprint Attendance App (`app.py`)

A standalone Flask application — a self-contained, simpler alternative to the Django system for fingerprint-only attendance. No camera recognition, no classroom monitoring. Just students + fingerprint scanners + an attendance log.

> Note: this app is independent of the Django project. It runs separately and has its own SQLite database (`attendance.db`).

---

## What it does

- Admin portal: add/edit/delete students, enroll fingerprints, view and export attendance
- Student portal: login, view personal attendance history, change password
- Fingerprint enrollment workflow via ESP32 command queue
- Attendance marking via ESP32 REST API
- Credential email delivery (HTML email with SMTP)
- Attendance export to styled Excel (`.xlsx`)
- Fingerprint image upload and viewer (4-bit grayscale PNG reconstruction)

---

## Models

### `Student`
| Field | Description |
|---|---|
| `name`, `roll_no`, `email` | Identity |
| `course`, `branch`, `year` | Academic info |
| `username` / `password_hash` | Portal credentials (Werkzeug PBKDF2) |
| `fingerprint_id` | Sensor slot (1–127) |
| `is_enrolled` | Set to `True` by ESP32 enroll-result callback |
| `fp_confidence` | Last scan confidence (0–255) |
| `fp_scan_count` | Lifetime successful scans |
| `fp_last_seen` | Last attendance timestamp |
| `fp_image_path` | Filename of reconstructed PNG |

### `AttendanceRecord`
One row per entry/exit event. `action` alternates between `IN` and `OUT` based on the previous record.

### `Admin`
Separate model for admin accounts (not linked to students).

---

## Fingerprint Enrollment Flow

Same pattern as the Django app but implemented with an in-memory dict + threading lock instead of Django's cache:

```
Admin POSTs /admin/enroll-fingerprint/<id>
  → assigns next free slot, sets esp32_command = {ENROLL, slot}
    → ESP32 polls GET /api/esp32/command
      → Receives ENROLL, enrolls finger on sensor
        → ESP32 POSTs /api/esp32/enroll-result {success, fingerprint_id}
          → Updates student.is_enrolled
        → ESP32 optionally POSTs /api/esp32/upload-image {fingerprint_id, image_base64}
          → Decodes 4-bit grayscale (256×288 = 36864 bytes) to 8-bit
          → Builds PNG manually (zlib + struct, no Pillow)
          → Saves to static/fingerprints/
```

---

## Attendance Marking

```
ESP32 POST /api/mark-attendance {fingerprint_id, confidence}
  → finds Student (must be is_enrolled=True)
  → checks last AttendanceRecord for this student
  → action = OUT if last was IN, else IN
  → creates AttendanceRecord
  → updates fp_confidence, fp_scan_count, fp_last_seen
  → returns {name, roll_no, action, time, confidence}
```

---

## Excel Export

Uses `openpyxl` to build a styled workbook:
- Merged title cell with date
- Header row with navy fill + white bold text
- Green rows for IN records, red rows for OUT records
- Summary row with total, IN count, OUT count
- Downloaded as `attendance_<date>.xlsx`

---

## Credential Emails

HTML email sent via SMTP (Gmail default) when a student is added. Contains username and password in a styled card layout. Resend credentials generates a new 10-character random password, updates the hash, and re-sends.

---

## Authentication

- Admin: session-based (`session['admin_logged_in']`)
- Student: session-based (`session['student_id']`)
- ESP32: `X-API-Key` header validated by `esp32_auth` decorator against `ESP32_API_KEY` env var

---

## Key Dependencies

| Library | Purpose |
|---|---|
| `flask` | Web framework |
| `flask-sqlalchemy` | ORM (SQLite) |
| `werkzeug` | Password hashing |
| `openpyxl` | Excel export |
| `smtplib` (stdlib) | Credential emails |
| `zlib` + `struct` (stdlib) | PNG encoding for fingerprint images |
