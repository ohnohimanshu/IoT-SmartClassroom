# Classroom IoT System

A comprehensive classroom management system with attendance, behavior monitoring, and lab supervision features.

## Features

- **Multi-Modal Attendance**: Fingerprint (ESP32) and Face Recognition (Camera)
- **Emotion Detection**: Track student mood changes during entry/exit
- **Classroom Behavior Monitoring**: Real-time engagement, phone usage, and fight detection
- **Lab Supervision**: Screen sharing, camera monitoring, and activity logging
- **WhatsApp Alerts**: Critical incidents (fighting, phone use) sent via WhatsApp
- **Analytics & Reports**: Attendance statistics, engagement scores, incident tracking

## Tech Stack

### Backend
- Python 3.11
- Django 5.1.4
- PostgreSQL 15
- Gunicorn (WSGI)
- Supervisor (Process Management)

### ML/AI
- Face Recognition: `face-recognition`, `DeepFace`
- Pose & Engagement: `MediaPipe`
- Object Detection: `Ultralytics YOLO`
- Fight Detection: PyTorch 3D CNN

### Frontend
- Django Templates
- HTML5, CSS3, JavaScript
- WebRTC (for lab monitoring)

### Infrastructure
- Docker
- Docker Compose
- Nginx (Reverse Proxy)
- SSL/TLS Support

## Project Structure

```
classroom_iot/
├── camera_attendance/     # Camera-based attendance app
├── classroom_monitor/     # Classroom behavior monitoring app
├── entrance_cam/          # Main entrance & fingerprint enrollment
├── lab_monitor/           # Lab supervision & screen sharing
├── classroom_iot/         # Django project settings
├── docker/                # Docker configs (supervisor, gunicorn)
├── certs/                 # SSL certificates
├── static/                # Static files (CSS, JS, images)
├── templates/             # Django templates
├── media/                 # Uploaded media (photos, snapshots)
├── manage.py              # Django management script
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker build file
└── docker-compose.yml     # Docker Compose orchestration
```

## Installation

### Prerequisites
- Python 3.11
- PostgreSQL 15
- Docker & Docker Compose (optional, for containerized deployment)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd classroom_iot
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create .env file** (see Environment Variables section)

5. **Run migrations**
   ```bash
   python manage.py migrate
   ```

6. **Create a superuser**
   ```bash
   python manage.py createsuperuser
   ```

7. **Collect static files**
   ```bash
   python manage.py collectstatic
   ```

8. **Start the server**
   ```bash
   python manage.py runsslserver 0.0.0.0:8000
   ```
   or without SSL:
   ```bash
   python manage.py runserver
   ```

## Environment Variables

Create a `.env` file in the project root:

```env
# Django
DJANGO_SECRET_KEY=your-secret-key-here
DJANGO_DEBUG=True
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1

# Database
POSTGRES_DB=classroom_iot_db
POSTGRES_USER=classroom_iot_user
POSTGRES_PASSWORD=your-db-password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# SSL
DJANGO_USE_SSL=True

# WhatsApp (Twilio) - Optional
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
TWILIO_WHATSAPP_FROM=whatsapp:+1234567890
TWILIO_WHATSAPP_TO=whatsapp:+0987654321
```

## Build Process

### Docker Build
```bash
docker build -t classroom-iot .
```

### Docker Compose
```bash
docker-compose up -d --build
```

## Deployment Instructions

### Docker Compose Deployment
1. Set environment variables in `docker-compose.yml` or `.env`
2. Place SSL certificates in `certs/` directory
3. Run:
   ```bash
   docker-compose up -d
   ```

### Manual Deployment
1. Install dependencies
2. Configure PostgreSQL
3. Set up Gunicorn + Supervisor
4. Configure Nginx as reverse proxy
5. Set up SSL certificates

See [Deployment Guide](docs/deployment-guide.md) for detailed instructions.

## Screenshots

*(Placeholder - add screenshots here)*

## License

*(Add license information here)*

## Documentation

- [Architecture Overview](docs/architecture.md)
- [API Documentation](docs/api-documentation.md)
- [Database Schema](docs/database-schema.md)
- [Deployment Guide](docs/deployment-guide.md)
- [User Manual](docs/user-manual.md)
- [Developer Guide](docs/developer-guide.md)
- [Troubleshooting](docs/troubleshooting.md)
