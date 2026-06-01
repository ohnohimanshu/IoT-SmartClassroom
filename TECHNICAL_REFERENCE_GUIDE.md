# Technical Reference Guide - Classroom IoT System

**Date:** June 1, 2026  
**Version:** 1.0.0  
**Audience:** Developers, DevOps, System Administrators

---

## 📚 Table of Contents

1. [Module Reference](#module-reference)
2. [API Reference](#api-reference)
3. [Database Reference](#database-reference)
4. [Configuration Reference](#configuration-reference)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Performance Tuning](#performance-tuning)
7. [Security Hardening](#security-hardening)

---

## 🔧 Module Reference

### entrance_cam Module

#### Models

**Student**
```python
class Student(models.Model):
    # Basic info
    name: CharField(100)
    roll_no: CharField(30, unique=True)
    email: EmailField(unique=True)
    course: CharField(20, choices=COURSE_CHOICES)
    branch: CharField(100)
    year: IntegerField(choices=YEAR_CHOICES)
    photo: ImageField(upload_to='students/photos/')
    
    # Face recognition
    face_encoding: TextField()  # JSON array of 128 floats
    is_enrolled: BooleanField()
    
    # Fingerprint
    fingerprint_id: IntegerField(unique=True, nullable=True)
    fp_confidence: IntegerField(nullable=True)
    fp_scan_count: IntegerField()
    fp_last_seen: DateTimeField(nullable=True)
    fp_image: ImageField(upload_to='fingerprints/')
    
    # Account
    user: OneToOneField(User)
    created_at: DateTimeField(auto_now_add=True)
    is_active: BooleanField()
```

**ESP32Device**
```python
class ESP32Device(models.Model):
    name: CharField(100)
    ip_address: CharField(100)
    location: CharField(100)
    is_active: BooleanField()
    last_seen: DateTimeField(nullable=True)
    api_key: CharField(64, unique=True)
    enrollment_mode: BooleanField()
    enrollment_student: ForeignKey(Student, nullable=True)
    pending_fingerprint_id: IntegerField(nullable=True)
    created_at: DateTimeField(auto_now_add=True)
    updated_at: DateTimeField(auto_now=True)
```

**FingerprintAttendance**
```python
class FingerprintAttendance(models.Model):
    student: ForeignKey(Student)
    device: ForeignKey(ESP32Device)
    date: DateField()
    timestamp: DateTimeField()
    attendance_type: CharField()  # 'entry' or 'exit'
    confidence: FloatField()
    duration_minutes: IntegerField(nullable=True)
```

#### Views

**api_log_entry(request)**
- **Method:** POST
- **URL:** `/api/log-entry/`
- **Purpose:** Log fingerprint attendance
- **Input:** JSON with student_id, device_id, fingerprint_id, confidence, attendance_type
- **Output:** JSON with status and student name
- **Error Codes:** 400 (invalid), 404 (not found), 500 (error)

```python
@csrf_exempt
@transaction.atomic
def api_log_entry(request):
    # Validates input
    # Checks for open entry log
    # Creates/updates FingerprintAttendance
    # Sends WhatsApp alert
    # Returns status
```

**dashboard(request)**
- **Method:** GET
- **URL:** `/`
- **Purpose:** Admin dashboard
- **Features:**
  - Permission check (staff/superuser)
  - Caching (60 seconds)
  - Statistics (students, cameras, today's attendance)
  - Recent activity
- **Error Handling:** Redirects to login if unauthorized

**next_fingerprint_slot()**
- **Purpose:** Allocate fingerprint slot (1-127)
- **Features:**
  - Atomic transaction
  - Database-level locking
  - Race condition prevention
- **Returns:** Slot number (1-127)
- **Raises:** RuntimeError if full

```python
@transaction.atomic
def next_fingerprint_slot():
    used = set(
        Student.objects.select_for_update()
                      .filter(fingerprint_id__isnull=False)
                      .values_list('fingerprint_id', flat=True)
    )
    for slot in range(1, 128):
        if slot not in used:
            return slot
    raise RuntimeError("Fingerprint sensor is full")
```

---

### camera_attendance Module

#### Models

**CameraAttendanceLog**
```python
class CameraAttendanceLog(models.Model):
    # Relationships
    student: ForeignKey(Student)
    camera: ForeignKey(Camera)
    
    # Date/Time
    date: DateField()
    entry_time: DateTimeField(nullable=True)
    exit_time: DateTimeField(nullable=True)
    
    # Emotions
    entry_emotion: CharField(20, choices=EMOTION_CHOICES)
    exit_emotion: CharField(20, choices=EMOTION_CHOICES)
    entry_emotion_score: FloatField()
    exit_emotion_score: FloatField()
    
    # Snapshots
    entry_snapshot: ImageField(upload_to='camera/entry/')
    exit_snapshot: ImageField(upload_to='camera/exit/')
    
    # Analysis
    mood_comparison: CharField(20, choices=MOOD_COMPARISON_CHOICES)
    duration_minutes: IntegerField(nullable=True)
    is_present: BooleanField()
    
    # Metadata
    created_at: DateTimeField(auto_now_add=True)
    updated_at: DateTimeField(auto_now=True)
```

#### Views

**api_log_camera_attendance(request)**
- **Method:** POST
- **URL:** `/camera-attendance/api/log/`
- **Purpose:** Log camera-based attendance
- **Input:** JSON with student_id, camera_id, emotion, score, snapshot
- **Logic:**
  1. Validates input
  2. Checks for open entry log
  3. If found: Updates exit time, emotion, mood
  4. If not found: Creates new entry log
  5. Saves snapshots
  6. Calculates mood comparison
- **Output:** JSON with status, student, emotion, duration, mood
- **Error Codes:** 400, 404, 405, 500

```python
@csrf_exempt
@transaction.atomic
def api_log_camera_attendance(request):
    # Validate method
    # Parse JSON
    # Validate required fields
    # Get student and camera
    # Check for open log
    # If exit: Update log
    # If entry: Create log
    # Save snapshots
    # Calculate mood
    # Return status
```

**camera_attendance_dashboard(request)**
- **Method:** GET
- **URL:** `/camera-attendance/dashboard/`
- **Purpose:** Dashboard with caching
- **Features:**
  - Permission check
  - 60-second cache
  - Statistics (students, encodings, entries, inside)
  - Emotion distribution
  - Recent activity
- **Cache Key:** `camera_dashboard_{user_id}`

**camera_attendance_list(request)**
- **Method:** GET
- **URL:** `/camera-attendance/`
- **Purpose:** View attendance logs
- **Features:**
  - Pagination (50 items/page)
  - Date filtering
  - Student filtering
  - Statistics (entries, exits, inside)
- **Query Optimization:** select_related('student', 'camera')

---

### classroom_monitor Module

#### Models

**ClassSession**
```python
class ClassSession(models.Model):
    camera: ForeignKey(ClassroomCamera)
    subject: CharField(100)
    teacher: CharField(100)
    start_time: DateTimeField(auto_now_add=True)
    end_time: DateTimeField(nullable=True)
    is_active: BooleanField()
    total_students_detected: IntegerField()
```

**EngagementSnapshot**
```python
class EngagementSnapshot(models.Model):
    session: ForeignKey(ClassSession)
    timestamp: DateTimeField(auto_now_add=True)
    frame_image: ImageField(upload_to='classroom/frames/')
    
    # Behavior counts
    focused_count: IntegerField()
    looking_away_count: IntegerField()
    head_down_count: IntegerField()
    using_phone_count: IntegerField()
    eating_count: IntegerField()
    not_visible_count: IntegerField()
    talking_count: IntegerField()
    
    # Analysis
    total_detected: IntegerField()
    engagement_score: FloatField()
```

**IncidentReport**
```python
class IncidentReport(models.Model):
    student: ForeignKey(Student, nullable=True)
    camera: ForeignKey(Camera, nullable=True)
    incident_type: CharField(20, choices=BEHAVIOR_INCIDENT_CHOICES)
    severity: CharField(20, choices=SEVERITY_CHOICES)
    description: TextField()
    snapshot: ImageField(upload_to='incidents/')
    confidence: FloatField()
    detected_at: DateTimeField(auto_now_add=True)
    whatsapp_sent: BooleanField()
    whatsapp_sent_at: DateTimeField(nullable=True)
    is_reviewed: BooleanField()
    reviewed_by: ForeignKey(User, nullable=True)
    reviewed_at: DateTimeField(nullable=True)
    admin_notes: TextField()
```

#### Behavior Detection

**ClassroomBehaviorDetector**
```python
class ClassroomBehaviorDetector:
    def __init__(self, camera_url, camera_id, server_url, alert_cooldown):
        self.camera_url = camera_url
        self.camera_id = camera_id
        self.yolo_model = YOLO('yolo11s.pt')
        self._frontal_cascade = cv2.CascadeClassifier(...)
        self._profile_cascade = cv2.CascadeClassifier(...)
    
    def detect(self, frame):
        """
        Detect behaviors in frame
        Returns: List of detection dicts with type, bbox, confidence, color, label
        """
        # YOLO person detection
        # For each person:
        #   - Haar cascade head pose
        #   - Object detection (phone, food)
        # Return detections
    
    def _classify_head_pose(self, gray_frame, x1, y1, x2, y2):
        """
        Classify head pose: focused, looking_away, head_down
        Uses Haar cascades (frontal, profile)
        """
    
    def _report_incident(self, detection, frame, student_id, student_name):
        """
        Save incident to DB and send WhatsApp alert
        """
```

#### Detection Pipeline

```
Frame Input
    ↓
YOLO11 Detection (persons, objects)
    ↓
For each person:
    ├─ Haar Cascade Head Pose
    │  ├─ Frontal face → Focused (GREEN)
    │  ├─ Profile face → Looking Away (ORANGE)
    │  └─ No face → Head Down (ORANGE)
    │
    └─ Object Detection
       ├─ Phone → Using Phone (RED)
       └─ Food → Eating Food (RED)
    ↓
Draw rectangles on frame
    ↓
Return detections
```

---

## 🔌 API Reference

### Authentication
All endpoints except `/api/log-entry/` and `/camera-attendance/api/log/` require login.

### Error Responses

**400 Bad Request**
```json
{
  "error": "Invalid JSON body" | "Missing required fields" | "Invalid input"
}
```

**404 Not Found**
```json
{
  "error": "Student not found" | "Camera not found" | "Device not found"
}
```

**405 Method Not Allowed**
```json
{
  "error": "GET only" | "POST only"
}
```

**500 Internal Server Error**
```json
{
  "error": "Internal server error"
}
```

### Success Responses

**Attendance Logged**
```json
{
  "status": "entry_logged" | "exit_logged",
  "student": "John Doe",
  "entry_emotion": "happy",
  "exit_emotion": "neutral",
  "duration": 45,
  "mood_comparison": "declined"
}
```

**Encodings Retrieved**
```json
[
  {
    "id": 1,
    "name": "John Doe",
    "roll_no": "2021CS001",
    "encoding": "[0.123, 0.456, ...]"
  }
]
```

**Live Detections**
```json
{
  "today_total": 50,
  "inside_now": 10,
  "logs": [
    {
      "student_name": "John Doe",
      "roll_no": "2021CS001",
      "entry_time": "2026-06-01T09:00:00",
      "exit_time": "2026-06-01T10:30:00",
      "entry_emotion": "happy",
      "exit_emotion": "neutral",
      "mood_comparison": "declined",
      "duration": 90
    }
  ]
}
```

---

## 🗄️ Database Reference

### Indexes for Performance

```sql
-- Student lookups
CREATE INDEX idx_student_roll_no ON entrance_cam_student(roll_no);
CREATE INDEX idx_student_email ON entrance_cam_student(email);

-- Attendance queries
CREATE INDEX idx_fingerprint_student_date ON entrance_cam_fingerprintattendance(student_id, date);
CREATE INDEX idx_camera_attendance_student_date ON camera_attendance_cameraattendancelog(student_id, date);

-- Session analysis
CREATE INDEX idx_engagement_session_timestamp ON classroom_monitor_engagementsnapshot(session_id, timestamp);

-- Incident tracking
CREATE INDEX idx_incident_student_date ON classroom_monitor_incidentreport(student_id, detected_at);
```

### Query Optimization

**Good Queries**
```python
# Use select_related for ForeignKey
logs = CameraAttendanceLog.objects.select_related('student', 'camera')

# Use prefetch_related for reverse relations
students = Student.objects.prefetch_related('camera_attendance_logs')

# Use filter before slicing
logs = CameraAttendanceLog.objects.filter(date=today).order_by('-entry_time')[:50]

# Use values() for specific fields
emotions = CameraAttendanceLog.objects.values('entry_emotion').annotate(count=Count('id'))
```

**Bad Queries**
```python
# N+1 queries
for log in logs:
    print(log.student.name)  # Query per iteration

# Loading all records
logs = CameraAttendanceLog.objects.all()  # Loads everything

# Unnecessary joins
logs = CameraAttendanceLog.objects.filter(student__is_active=True)  # Extra join
```

---

## ⚙️ Configuration Reference

### settings.py

**Database**
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
        'OPTIONS': {
            'timeout': 20,
        },
        'CONN_MAX_AGE': 60,
    }
}
```

**Caching**
```python
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}
```

**Logging**
```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/attendance.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'attendance': {
            'handlers': ['file'],
            'level': 'INFO',
        },
    },
}
```

**Media Files**
```python
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'
```

---

## 🔍 Troubleshooting Guide

### Issue: "Fingerprint slot already taken"

**Cause:** Race condition in slot allocation  
**Solution:** Already fixed with `@transaction.atomic` and `select_for_update()`  
**Verification:**
```python
# Check if fix is applied
@transaction.atomic
def next_fingerprint_slot():
    used = set(
        Student.objects.select_for_update()  # This line is critical
                      .filter(fingerprint_id__isnull=False)
                      .values_list('fingerprint_id', flat=True)
    )
```

### Issue: "Internal server error" on API

**Cause:** Unhandled exception  
**Solution:** Check logs
```bash
tail -f logs/attendance.log
```

**Common Causes:**
- Invalid JSON: Check request body format
- Missing student: Verify student_id exists
- Missing camera: Verify camera_id exists
- Database error: Check database connection

### Issue: Dashboard is slow

**Cause:** Cache not configured or expired  
**Solution:** Configure Redis
```python
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
    }
}
```

**Verify Cache:**
```python
from django.core.cache import cache
cache.set('test', 'value', 60)
print(cache.get('test'))  # Should print 'value'
```

### Issue: Face recognition not working

**Cause:** Face not detected in photo  
**Solution:**
1. Use clear, front-facing photo
2. Good lighting
3. Face should be 50%+ of image
4. Regenerate encoding:
```bash
python manage.py regenerate_face_encodings
```

### Issue: Permission denied on dashboard

**Cause:** User is not staff/superuser  
**Solution:** Add user to staff group
```bash
python manage.py shell
>>> from django.contrib.auth.models import User
>>> user = User.objects.get(username='john')
>>> user.is_staff = True
>>> user.save()
```

---

## ⚡ Performance Tuning

### Database Optimization

**Enable WAL Mode**
```python
# In settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
        'OPTIONS': {
            'timeout': 20,
            'journal_mode': 'WAL',
        },
    }
}
```

**Add Indexes**
```python
class Meta:
    indexes = [
        models.Index(fields=['student', 'date']),
        models.Index(fields=['date', '-entry_time']),
    ]
```

### Query Optimization

**Use select_related()**
```python
# Before: N+1 queries
logs = CameraAttendanceLog.objects.all()
for log in logs:
    print(log.student.name)  # Query per iteration

# After: 1 query
logs = CameraAttendanceLog.objects.select_related('student')
for log in logs:
    print(log.student.name)  # No additional queries
```

**Use prefetch_related()**
```python
# Before: N+1 queries
students = Student.objects.all()
for student in students:
    logs = student.camera_attendance_logs.all()  # Query per student

# After: 2 queries
students = Student.objects.prefetch_related('camera_attendance_logs')
for student in students:
    logs = student.camera_attendance_logs.all()  # No additional queries
```

### Caching Strategy

**Cache Dashboard**
```python
from django.core.cache import cache

cache_key = f'dashboard_data_{user_id}'
context = cache.get(cache_key)

if context is None:
    # Build context
    context = {...}
    cache.set(cache_key, context, 60)  # Cache for 60 seconds
```

**Cache API Responses**
```python
from django.views.decorators.cache import cache_page

@cache_page(60)  # Cache for 60 seconds
def api_students_encodings(request):
    # ...
```

### Memory Optimization

**Use iterator() for large querysets**
```python
# Before: Loads all records into memory
for log in CameraAttendanceLog.objects.all():
    process(log)

# After: Streams records
for log in CameraAttendanceLog.objects.all().iterator():
    process(log)
```

**Use values() for specific fields**
```python
# Before: Loads full objects
emotions = CameraAttendanceLog.objects.all()

# After: Loads only needed fields
emotions = CameraAttendanceLog.objects.values('entry_emotion', 'exit_emotion')
```

---

## 🔐 Security Hardening

### Input Validation

**Always validate input**
```python
def validate_emotion(emotion):
    emotion = (emotion or '').lower().strip()
    valid_emotions = {'happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust', 'unknown'}
    return emotion if emotion in valid_emotions else 'unknown'

def clamp_score(score, min_val=0.0, max_val=1.0):
    try:
        score = float(score)
        return max(min_val, min(max_val, score))
    except (ValueError, TypeError):
        return min_val
```

### Permission Checks

**Always check permissions**
```python
@login_required
def dashboard(request):
    if not request.user.is_staff and not request.user.is_superuser:
        logger.warning(f"Unauthorized access by {request.user}")
        return redirect('login')
    # ... rest of view
```

### Error Handling

**Never expose sensitive information**
```python
# Bad: Exposes database details
except Exception as e:
    return JsonResponse({'error': str(e)}, status=500)

# Good: Generic error message
except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
    return JsonResponse({'error': 'Internal server error'}, status=500)
```

### CSRF Protection

**Always use CSRF tokens**
```html
<form method="POST">
    {% csrf_token %}
    <!-- form fields -->
</form>
```

### SQL Injection Prevention

**Always use ORM**
```python
# Bad: SQL injection risk
Student.objects.raw(f"SELECT * FROM student WHERE name = '{name}'")

# Good: ORM prevents injection
Student.objects.filter(name=name)
```

---

## 📊 Monitoring Checklist

- [ ] API response time < 200ms
- [ ] Database queries < 5 per request
- [ ] Cache hit rate > 90%
- [ ] Error rate < 0.1%
- [ ] Fingerprint slot usage < 80%
- [ ] Disk space > 10GB
- [ ] Memory usage < 80%
- [ ] CPU usage < 70%

---

## 🚀 Deployment Checklist

- [ ] Review all changes
- [ ] Run full test suite
- [ ] Backup production database
- [ ] Run migrations
- [ ] Collect static files
- [ ] Configure cache backend
- [ ] Configure logging
- [ ] Test all endpoints
- [ ] Verify permissions
- [ ] Monitor error logs
- [ ] Set up alerts

---

**Last Updated:** June 1, 2026  
**Version:** 1.0.0
