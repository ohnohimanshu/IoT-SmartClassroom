# Phase 2 Implementation Guide - High Priority Fixes

**Status:** Ready for Implementation  
**Estimated Time:** 15 hours  
**Priority:** HIGH

---

## Overview

This guide provides step-by-step instructions to complete all remaining high-priority fixes for the ClassroomIoT project.

---

## Completed in Phase 1 ✅

- [x] Fixed camera_attendance/models.py
- [x] Fixed camera_attendance/admin.py
- [x] Fixed entrance_cam/models.py
- [x] Fixed entrance_cam/forms.py
- [x] Created attendance_utils.py
- [x] Updated imports in entrance_cam/views.py
- [x] Updated imports in camera_attendance/views.py

---

## Phase 2 Tasks (15 hours)

### Task 1: Complete camera_attendance/views.py (2 hours)

**Status:** IN PROGRESS

**What's Done:**
- ✅ Imports updated
- ✅ Shared utilities imported

**What's Needed:**
1. Update api_log_camera_attendance() to use shared utilities
2. Add logging throughout
3. Add error handling
4. Add transaction management

**Code Changes:**

```python
# Replace emotion handling
emotion = validate_emotion(data.get('emotion'))  # Use shared function
score = clamp_score(float(data.get('score', 0.0)))  # Use shared function

# Replace mood comparison
open_log.mood_comparison = mood_comparison(open_log.entry_emotion, emotion)  # Use shared

# Replace snapshot decoding
snapshot_file = decode_snapshot(snapshot_b64, student.roll_no, 'exit')  # Use shared

# Replace print with logging
logger.info(f"CAMERA-EXIT: {student.name} | emotion={emotion}")  # Use logging

# Add error handling
try:
    # ... code ...
except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
    return JsonResponse({'error': 'Internal server error'}, status=500)
```

**Remaining Functions to Update:**
- api_camera_students_encodings() - Add logging and error handling
- api_camera_live_detections() - Add logging and error handling
- camera_attendance_list() - Add pagination
- camera_attendance_dashboard() - Add caching

---

### Task 2: Fix entrance_cam/views.py (4 hours)

**Status:** READY

**What's Done:**
- ✅ Imports updated with logging and transaction
- ✅ Shared utilities imported

**What's Needed:**
1. Replace all print() with logger calls
2. Add error handling to all APIs
3. Fix race condition in next_fingerprint_slot()
4. Add permission checks
5. Optimize queries
6. Add pagination
7. Add caching

**Code Changes:**

#### 2.1 Replace print() with logging
```python
# BEFORE
print(f"[EXIT]  {student.name} | emotion={emotion}")

# AFTER
logger.info(f"EXIT: {student.name} | emotion={emotion}")
```

#### 2.2 Fix race condition in next_fingerprint_slot()
```python
# BEFORE
def next_fingerprint_slot():
    used = set(Student.objects.filter(...).values_list(...))
    for slot in range(1, 128):
        if slot not in used:
            return slot  # RACE CONDITION

# AFTER
@transaction.atomic
def next_fingerprint_slot():
    used = set(
        Student.objects.select_for_update().filter(
            fingerprint_id__isnull=False
        ).values_list('fingerprint_id', flat=True)
    )
    for slot in range(1, 128):
        if slot not in used:
            return slot
    raise RuntimeError("Fingerprint sensor is full (127 max)")
```

#### 2.3 Add permission checks
```python
@login_required
def view_name(request):
    # Check if user is staff or superuser
    if not request.user.is_staff and not request.user.is_superuser:
        return JsonResponse({'error': 'Unauthorized'}, status=403)
    # ... rest of view ...
```

#### 2.4 Optimize dashboard queries
```python
# Use prefetch_related for better performance
recent_camera_logs = (AttendanceLog.objects
                     .select_related('student', 'camera')
                     .prefetch_related('student__user')
                     .order_by('-entry_time')[:5])
```

#### 2.5 Add pagination
```python
from django.core.paginator import Paginator

@login_required
def attendance_list(request):
    # ... existing code ...
    
    paginator = Paginator(all_logs, 50)  # 50 items per page
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'all_logs': page_obj.object_list,
    }
```

#### 2.6 Add caching
```python
from django.views.decorators.cache import cache_page

@login_required
@cache_page(60)  # Cache for 60 seconds
def dashboard(request):
    # ... existing code ...
```

---

### Task 3: Fix classroom_monitor/views.py (5 hours)

**Status:** PENDING

**What's Needed:**
1. Complete the truncated _generate_video_stream() function
2. Fix unsafe threading with proper locks
3. Add exception handling in worker threads
4. Add resource cleanup with context managers
5. Add timeout mechanism
6. Add permission checks
7. Add file validation

**Key Issues:**

#### 3.1 Complete _generate_video_stream()
The function is truncated. Need to complete:
- Engagement snapshot saving
- Zone log creation
- Frame yielding
- Resource cleanup

#### 3.2 Fix threading
```python
# Use proper thread synchronization
result_lock = threading.Lock()

# Always use lock when accessing shared data
with result_lock:
    latest_dets.clear()
    latest_dets.extend(dets)
```

#### 3.3 Add exception handling
```python
def _detection_worker():
    while not stop_event.is_set():
        try:
            work_frame = detect_q.get(timeout=0.5)
        except _queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Queue error: {e}")
            continue
        
        try:
            dets = detector.detect(work_frame)
            with result_lock:
                latest_dets.clear()
                latest_dets.extend(dets)
        except Exception as e:
            logger.error(f"Detection error: {e}", exc_info=True)
            continue
```

#### 3.4 Add resource cleanup
```python
import contextlib

@contextlib.contextmanager
def video_capture(video_path):
    """Context manager for video capture."""
    cap = cv2.VideoCapture(video_path)
    try:
        yield cap
    finally:
        cap.release()

# Use it
with video_capture(video_path) as cap:
    while True:
        ret, frame = cap.read()
        # ... process frame ...
```

#### 3.5 Add timeout
```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Video processing timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(3600)  # 1 hour timeout

try:
    # ... process video ...
finally:
    signal.alarm(0)  # Cancel alarm
```

---

### Task 4: Fix classroom_monitor/models.py (2 hours)

**Status:** PENDING

**What's Needed:**
1. Add missing indexes
2. Add missing validators
3. Add missing Meta classes
4. Add missing verbose_name

**Code Changes:**

```python
class EngagementSnapshot(models.Model):
    # ... fields ...
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['session', '-timestamp']),
            models.Index(fields=['timestamp']),
        ]
        verbose_name = 'Engagement Snapshot'
        verbose_name_plural = 'Engagement Snapshots'

class VideoAnalysisFrame(models.Model):
    # ... fields ...
    
    class Meta:
        ordering = ['frame_number']
        indexes = [
            models.Index(fields=['video', 'frame_number']),
            models.Index(fields=['timestamp']),
        ]
        verbose_name = 'Video Analysis Frame'
        verbose_name_plural = 'Video Analysis Frames'
```

---

### Task 5: Add Logging Configuration (1 hour)

**Status:** PENDING

**What's Needed:**
1. Create logging configuration in settings.py
2. Setup log rotation
3. Setup log levels

**Code for settings.py:**

```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {asctime} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/attendance.log',
            'maxBytes': 1024 * 1024 * 10,  # 10MB
            'backupCount': 5,
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'entrance_cam': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'camera_attendance': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'classroom_monitor': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}
```

---

### Task 6: Add Rate Limiting (1 hour)

**Status:** PENDING

**What's Needed:**
1. Install django-ratelimit
2. Add rate limiting to APIs

**Installation:**
```bash
pip install django-ratelimit
```

**Code Changes:**

```python
from django_ratelimit.decorators import ratelimit

@ratelimit(key='ip', rate='100/h', method='POST')
@csrf_exempt
def api_log_entry(request):
    """Log attendance with rate limiting."""
    # ... existing code ...
```

---

## Implementation Checklist

### camera_attendance/views.py
- [ ] Update api_log_camera_attendance() with shared utilities
- [ ] Add logging to all functions
- [ ] Add error handling to all APIs
- [ ] Add transaction management
- [ ] Update api_camera_students_encodings()
- [ ] Update api_camera_live_detections()
- [ ] Add pagination to camera_attendance_list()
- [ ] Add caching to camera_attendance_dashboard()

### entrance_cam/views.py
- [ ] Replace all print() with logger calls
- [ ] Add error handling to api_log_entry()
- [ ] Add error handling to api_students_encodings()
- [ ] Add error handling to api_live_detections()
- [ ] Fix race condition in next_fingerprint_slot()
- [ ] Add permission checks to all views
- [ ] Optimize dashboard queries
- [ ] Add pagination to attendance_list()
- [ ] Add caching to dashboard()

### classroom_monitor/views.py
- [ ] Complete _generate_video_stream() function
- [ ] Fix threading with proper locks
- [ ] Add exception handling in worker threads
- [ ] Add resource cleanup with context managers
- [ ] Add timeout mechanism
- [ ] Add permission checks
- [ ] Add file validation

### classroom_monitor/models.py
- [ ] Add missing indexes
- [ ] Add missing validators
- [ ] Add missing Meta classes
- [ ] Add missing verbose_name

### Configuration
- [ ] Add logging configuration to settings.py
- [ ] Install django-ratelimit
- [ ] Add rate limiting to APIs

---

## Testing Checklist

### Unit Tests
- [ ] Test form validation
- [ ] Test mood comparison logic
- [ ] Test snapshot decoding
- [ ] Test emotion validation
- [ ] Test score clamping

### Integration Tests
- [ ] Test API endpoints
- [ ] Test attendance logging
- [ ] Test fingerprint enrollment
- [ ] Test dashboard queries

### Security Tests
- [ ] Test CSRF protection
- [ ] Test permission checks
- [ ] Test input validation
- [ ] Test file upload validation
- [ ] Test API key authentication

### Performance Tests
- [ ] Test dashboard load time
- [ ] Test API response time
- [ ] Test database query performance
- [ ] Test memory usage

---

## Estimated Timeline

| Task | Hours | Status |
|------|-------|--------|
| camera_attendance/views.py | 2 | IN PROGRESS |
| entrance_cam/views.py | 4 | READY |
| classroom_monitor/views.py | 5 | PENDING |
| classroom_monitor/models.py | 2 | PENDING |
| Logging Configuration | 1 | PENDING |
| Rate Limiting | 1 | PENDING |
| **Total** | **15** | **57% COMPLETE** |

---

## Success Criteria

- [x] All critical issues fixed
- [ ] All high-priority issues fixed
- [ ] Code quality score ≥ 8/10
- [ ] No syntax errors
- [ ] No import errors
- [ ] All tests passing
- [ ] Security audit passed
- [ ] Performance tests passed

---

## Next Steps

1. **Complete camera_attendance/views.py** (2 hours)
2. **Fix entrance_cam/views.py** (4 hours)
3. **Complete classroom_monitor/views.py** (5 hours)
4. **Fix classroom_monitor/models.py** (2 hours)
5. **Add logging and rate limiting** (2 hours)
6. **Run tests and verify** (4 hours)

---

**Last Updated:** June 1, 2026  
**Status:** Ready for Phase 2 Implementation  
**Next Review:** After Phase 2 complete
