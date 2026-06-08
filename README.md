
# Classroom IoT - Smart Campus Monitoring System

A comprehensive, AI-powered classroom monitoring and attendance system combining facial recognition, fingerprint verification, behavior analysis, and lab monitoring capabilities.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Environment Variables](#environment-variables)
7. [Usage](#usage)
8. [Docker Deployment](#docker-deployment)
9. [Tech Stack](#tech-stack)
10. [Development Notes](#development-notes)

---

## Project Overview

Classroom IoT is an integrated solution for smart campus management. It provides:
- Dual-mode attendance (facial recognition + fingerprint verification)
- Classroom behavior monitoring (engagement, phone/eating detection, fight detection)
- Lab session monitoring with screen sharing
- Real-time camera streaming
- Automated alerts (WhatsApp notifications)
- Comprehensive dashboard and reporting

---

## Key Features

### 1. Student Management
- Student profile creation with photos
- Automatic face encoding generation for facial recognition
- Fingerprint enrollment support via ESP32 devices
- Course, branch, and year categorization

### 2. Attendance System
#### Camera-based Attendance
- Real-time face recognition from camera streams
- Emotion detection on entry/exit
- Automatic duration calculation
- Mood comparison between entry/exit
- Snapshots for verification

#### Fingerprint-based Attendance
- ESP32-based fingerprint scanner integration
- Fingerprint template storage
- Entry/exit detection
- Confidence-based matching

### 3. Classroom Monitoring
- Real-time behavior analysis using YOLO11s-Pose
- Engagement detection (focused, looking away, head down)
- Alert behaviors: phone usage, eating, fighting
- Incident reporting system
- Temporal smoothing for false positive reduction
- Post-session video analysis

### 4. Lab Monitor
- Student dashboard
- Lab session tracking
- Screen sharing via WebRTC
- Activity logging

### 5. Camera & Device Management
- Support for multiple IP cameras (RTSP, HTTP, MJPEG)
- ESP32 device management for fingerprint scanners
- Camera status monitoring
- Automatic reconnection for failed streams

### 6. Notifications & Alerts
- WhatsApp alerts for critical incidents (fighting)
- Incident review system
- Admin dashboard for all reports

---

## System Architecture

```
Classroom IoT System
├── Web Application (Django)
│   ├── entrance_cam (student & ESP32 management)
│   ├── camera_attendance (camera-based attendance)
│   ├── classroom_monitor (behavior & engagement analysis)
│   └── lab_monitor (lab session tracking)
│
├── AI/ML Services
│   ├── Face Recognition (face_recognition library)
│   ├── Emotion Detection (DeepFace)
│   ├── Behavior Analysis (YOLO11s-Pose + ByteTrack)
│   └── Fight Detection (kinetic energy + overlap analysis)
│
├── IoT Devices
│   ├── ESP32 (Fingerprint Scanner)
│   └── IP Cameras (RTSP/HTTP)
│
└── Data Storage
    └── PostgreSQL Database
```

---

## Project Structure

```
classroom_iot/
├── camera_attendance/          # Camera-based attendance app
│   ├── models.py               # Camera & attendance log models
│   ├── views.py                # Views for attendance management
│   ├── urls.py                 # URL routing
│   ├── forms.py                # Forms for camera configuration
│   ├── detection_script_v2.py  # Face recognition & emotion detection script
│   └── management/commands/    # Custom Django commands for detection
│
├── classroom_monitor/          # Classroom behavior analysis app
│   ├── models.py               # Classroom camera, session, incident models
│   ├── views.py                # Monitoring views
│   ├── urls.py                 # URL routing
│   ├── behavior_detection.py   # YOLO-based behavior detector
│   ├── fight_detection.py      # Fight detection logic
│   └── face_recognition_helper.py # Face recognition utilities
│
├── entrance_cam/               # Main app: students & fingerprint devices
│   ├── models.py               # Student, ESP32Device, FingerprintAttendance models
│   ├── views.py                # Dashboard & management views
│   ├── urls.py                 # URL routing & ESP32 API endpoints
│   ├── signals.py              # Auto-generate face encodings on photo upload
│   ├── whatsapp_service.py     # WhatsApp alert integration
│   └── management/commands/    # Encoding generation, camera reload commands
│
├── lab_monitor/                # Lab session monitoring app
│   ├── models.py               # LabSession, Screenshot, CameraSnapshot models
│   ├── views.py                # Admin & student views
│   ├── urls.py                 # URL routing
│   └── student_urls.py         # Student-specific URLs
│
├── classroom_iot/              # Django project config
│   ├── settings.py             # Project settings
│   ├── urls.py                 # Project URL configuration
│   ├── wsgi.py                 # WSGI config
│   └── wsgi_fix.py             # SSL & broken pipe fixes
│
├── templates/                  # HTML templates
│   ├── camera_attendance/
│   ├── classroom_monitor/
│   └── entrance_cam/
│
├── media/                      # Uploaded media (photos, snapshots)
├── static/                     # Static files (CSS, JS, images)
│
├── attendance_utils.py         # Utility functions for mood comparison, etc.
├── esp32_attendance.ino        # ESP32 fingerprint scanner firmware
├── docker-compose.yml          # Docker Compose configuration
├── Dockerfile                  # Docker image for Django
└── requirements.txt            # Python dependencies
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- PostgreSQL 15+
- (Optional) ESP32 device for fingerprint attendance
- (Optional) IP cameras for video streaming

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd classroom_iot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (see [Environment Variables](#environment-variables) section below)

5. **Set up the database**
   ```bash
   python manage.py migrate
   ```

6. **Create a superuser**
   ```bash
   python manage.py createsuperuser
   ```

7. **Run the development server**
   ```bash
   # With SSL (recommended for production-like testing)
   python manage.py runsslserver 0.0.0.0:8000 --certificate cert.pem --key key.pem
   # Without SSL
   python manage.py runserver
   ```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Django
DJANGO_SECRET_KEY="your-secret-key-here"
DJANGO_DEBUG="True"
DJANGO_ALLOWED_HOSTS="localhost,127.0.0.1,192.168.1.100"
DJANGO_USE_SSL="False"

# PostgreSQL
POSTGRES_DB="classroom_iot_db"
POSTGRES_USER="classroom_iot_user"
POSTGRES_PASSWORD="your-db-password"
POSTGRES_HOST="localhost"
POSTGRES_PORT="5432"

# WhatsApp Alerts (using Twilio or similar)
ADMIN_WHATSAPP="whatsapp:+911234567890"

# ESP32 API Key
ESP32_API_KEY="your-esp32-api-key"
```

---

## Usage

### Starting Detection Services

The project includes custom Django management commands to run detection services:

```bash
# Run attendance detection
python manage.py run_detection

# Run classroom behavior detection
python manage.py run_behavior_detection

# Run both + Django server together
python manage.py runserver_with_detection

# Generate face encodings for all students
python manage.py generate_face_encodings

# Reload camera URLs from database
python manage.py reload_camera_urls
```

### Using the Web Interface

1. **Dashboard** - Overview of attendance, students, devices
2. **Students** - Add, edit, or delete student profiles; enroll fingerprints
3. **Cameras** - Configure IP cameras for attendance or classroom monitoring
4. **Attendance** - View attendance logs for both camera and fingerprint systems
5. **Classroom Monitor** - Live camera feed with behavior analysis; view incidents
6. **Lab Monitor** - Manage lab sessions and student activity

### ESP32 Fingerprint Scanner

Upload `esp32_attendance.ino` to your ESP32 board. Configure the Wi-Fi credentials and server URL in the sketch.

---

## Docker Deployment

The project includes Docker support for easy deployment:

1. **Build and start containers**
   ```bash
   docker-compose up -d --build
   ```

2. **Apply migrations**
   ```bash
   docker-compose exec web python manage.py migrate
   ```

3. **Create a superuser**
   ```bash
   docker-compose exec web python manage.py createsuperuser
   ```

4. **Access the application** at `http://localhost:8000`

---

## Tech Stack

### Backend
- **Django 5.1.4** - Web framework
- **PostgreSQL** - Database
- **psycopg2-binary** - PostgreSQL adapter

### Computer Vision & AI
- **OpenCV** - Image processing
- **face_recognition** - Face detection & recognition
- **DeepFace** - Emotion detection
- **Ultralytics YOLO11s-Pose** - Human pose estimation
- **ByteTrack** - Multi-object tracking
- **MediaPipe** - (Legacy) Face mesh detection

### Other
- **Pillow** - Image handling
- **NumPy** - Numerical computing
- **django-sslserver** - SSL for development
- **Docker** - Containerization

---

## Development Notes

### Important Models
| Model | App | Description |
|-------|-----|-------------|
| `Student` | `entrance_cam` | Student profile with face encoding and fingerprint ID |
| `ESP32Device` | `entrance_cam` | ESP32 fingerprint scanner device config |
| `FingerprintAttendance` | `entrance_cam` | Fingerprint-based attendance log |
| `Camera` | `camera_attendance` | Camera configuration for attendance |
| `CameraAttendanceLog` | `camera_attendance` | Camera-based attendance log with emotions |
| `ClassroomCamera` | `classroom_monitor` | Camera for behavior monitoring |
| `ClassSession` | `classroom_monitor` | Active classroom session |
| `IncidentReport` | `classroom_monitor` | Behavior incident log |
| `LabSession` | `lab_monitor` | Lab session tracking |

### Custom Django Commands
- `generate_face_encodings` - Generate face encodings for all students
- `regenerate_face_encodings` - Regenerate encodings (e.g., after model update)
- `reload_camera_urls` - Reload camera configs without restarting
- `run_detection` - Run attendance detection script
- `run_behavior_detection` - Run classroom behavior detector
- `runserver_with_detection` - Run server + detection together

### Behavior Detection Pipeline
1. **YOLO11s-Pose** detects people, keypoints, phones, food
2. **ByteTrack** tracks people across frames
3. **TemporalBehaviorEngine** analyzes behavior with 3-frame smoothing
4. **Incident reporting** for alert behaviors (phone, eating, fighting)

---

## License

[Add your license here]

---

## Contributing

[Add contribution guidelines here]

---

## Acknowledgments

- [face_recognition](https://github.com/ageitgey/face_recognition) library
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [DeepFace](https://github.com/serengil/deepface)
