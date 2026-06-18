# API Documentation

## Overview
This document describes all available API endpoints for the Classroom IoT System.

## Authentication

### Admin/User Authentication
Most UI endpoints require session-based authentication (login via `/login/`).

### ESP32 Device Authentication
ESP32 devices must include an `X-API-Key` header with their API key.

---

## Entrance Camera & Fingerprint API

### 1. Get ESP32 Command
**Endpoint**: `GET /api/esp32/command/`  
**Authentication**: ESP32 API Key (X-API-Key header)

**Response Example**:
```json
{
  "command": "ENROLL",
  "fingerprint_id": 5
}
```

### 2. Submit Enrollment Result
**Endpoint**: `POST /api/esp32/enroll-result/`  
**Authentication**: ESP32 API Key

**Request Body**:
```json
{
  "fingerprint_id": 5,
  "success": true
}
```

**Response Example**:
```json
{"ok": true}
```

### 3. Upload Fingerprint Image
**Endpoint**: `POST /api/esp32/upload-image/`  
**Authentication**: ESP32 API Key

**Request Body**:
```json
{
  "fingerprint_id": 5,
  "image": "base64-encoded-image-data"
}
```

### 4. Mark Fingerprint Attendance
**Endpoint**: `POST /api/mark-attendance/`  
**Authentication**: ESP32 API Key

**Description**: Records a fingerprint-based attendance entry/exit.

---

## Camera Attendance API

### 1. Get Student Encodings
**Endpoint**: `GET /camera-attendance/api/students/encodings/`  
**Authentication**: None (CSRF exempt)

**Response Example**:
```json
[
  {
    "id": 1,
    "name": "John Doe",
    "roll_no": "23CS001",
    "encoding": "[...]"
  }
]
```

### 2. Log Camera Attendance
**Endpoint**: `POST /camera-attendance/api/log/`  
**Authentication**: None (CSRF exempt)

**Request Body**:
```json
{
  "student_id": 1,
  "camera_id": 1,
  "emotion": "happy",
  "score": 0.9,
  "snapshot": "base64-encoded-image"
}
```

**Response Example (Entry)**:
```json
{
  "status": "entry_logged",
  "student": "John Doe",
  "entry_emotion": "happy"
}
```

**Response Example (Exit)**:
```json
{
  "status": "exit_logged",
  "student": "John Doe",
  "exit_emotion": "neutral",
  "duration": 60,
  "mood_comparison": "stable"
}
```

### 3. Get Live Detections
**Endpoint**: `GET /camera-attendance/api/live-detections/?camera_id=1`  
**Authentication**: None (CSRF exempt)

---

## Classroom Monitor API

### 1. Start Class Session
**Endpoint**: `POST /classroom/session/start/`  
**Authentication**: Admin/Staff login required

**Request Body**:
```json
{
  "camera_id": 1,
  "subject": "Computer Science",
  "teacher": "Prof. Smith"
}
```

### 2. End Class Session
**Endpoint**: `POST /classroom/session/end/`  
**Authentication**: Admin/Staff login required

**Request Body**:
```json
{
  "session_id": 123
}
```

### 3. Submit Engagement Snapshot
**Endpoint**: `POST /classroom/api/snapshot/`  
**Authentication**: None (CSRF exempt)

**Request Body**:
```json
{
  "camera_id": 1,
  "session_id": 123,
  "frame_snapshot_b64": "base64-image",
  "students": [
    {
      "zone_id": 1,
      "pose": "focused",
      "possibly_talking": false,
      "confidence": 0.9
    }
  ]
}
```

### 4. Get Active Session
**Endpoint**: `GET /classroom/api/active-session/?camera_id=1`  
**Authentication**: None (CSRF exempt)

### 5. Get Session Stats
**Endpoint**: `GET /classroom/api/stats/<session_id>/`  
**Authentication**: None (CSRF exempt)

### 6. Report Behavior Incident
**Endpoint**: `POST /classroom/api/incidents/report/`  
**Authentication**: None (CSRF exempt)

---

## Lab Monitor API

### 1. Start Lab Session
**Endpoint**: `POST /lab/session/start/`  
**Authentication**: Student login required

### 2. End Lab Session
**Endpoint**: `POST /lab/session/end/`  
**Authentication**: Student login required

**Request Body**:
```json
{"session_id": 456}
```

### 3. Upload Screenshot
**Endpoint**: `POST /lab/screenshot/`  
**Authentication**: Student login required

**Request Body**:
```json
{
  "session_id": 456,
  "image_b64": "base64-image",
  "tab_title": "VS Code"
}
```

### 4. Upload Camera Frame
**Endpoint**: `POST /lab/camera-frame/`  
**Authentication**: Student login required

**Request Body**:
```json
{
  "session_id": 456,
  "image_b64": "base64-image"
}
```

### 5. Log Activity
**Endpoint**: `POST /lab/activity/`  
**Authentication**: Student login required

**Request Body**:
```json
{
  "session_id": 456,
  "tab_title": "YouTube",
  "activity_type": "distracted"
}
```

### 6. Get Active Sessions
**Endpoint**: `GET /lab/api/active-sessions/`  
**Authentication**: Admin/Staff login required

### 7. Get Session Detail
**Endpoint**: `GET /lab/api/session/<session_id>/`  
**Authentication**: Admin/Staff login required

### 8. WebRTC - Send Offer
**Endpoint**: `POST /lab/api/session/<session_id>/webrtc/offer/`  
**Authentication**: Student login required

### 9. WebRTC - Send Answer
**Endpoint**: `POST /lab/api/session/<session_id>/webrtc/answer/`  
**Authentication**: Admin/Staff login required

### 10. WebRTC - Add ICE Candidate
**Endpoint**: `POST /lab/api/session/<session_id>/webrtc/ice/`  
**Authentication**: Student or Admin login required

---

## Error Responses
All endpoints may return the following error status codes:

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Missing or invalid credentials |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource doesn't exist |
| 405 | Method Not Allowed - Wrong HTTP verb |
| 500 | Internal Server Error |

**Error Response Example**:
```json
{"error": "Student not found"}
```
