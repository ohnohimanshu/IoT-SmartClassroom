from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.http import JsonResponse, StreamingHttpResponse
from django.db.models import Count, Q
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.core.cache import cache
from django.core.files.base import ContentFile
from django.db import transaction
from datetime import date, timedelta, datetime
from functools import wraps
import json
import base64
import struct
import zlib
import os
import logging

from .models import Student, Camera, AttendanceLog, ESP32Device, FingerprintAttendance
from .forms import StudentForm, CameraForm, ESP32DeviceForm
from attendance_utils import mood_comparison, decode_snapshot, validate_emotion, clamp_score

logger = logging.getLogger(__name__)

ESP32_API_KEY = os.environ.get("ESP32_API_KEY", "")
if not ESP32_API_KEY:
    import warnings
    warnings.warn("ESP32_API_KEY env var is not set — all ESP32 API requests will be rejected", stacklevel=1)


# ── Auth ──────────────────────────────────────────────────────────────────────

def csrf_failure(request, reason=""):
    messages.error(request, 'Session expired or invalid. Please login again.')
    return redirect('login')


def login_view(request):
    if request.user.is_authenticated:
        try:
            request.user.student_profile  # noqa
            return redirect('student_dashboard')
        except Student.DoesNotExist:
            pass
        return redirect('dashboard')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            try:
                user.student_profile  # noqa
                return redirect('student_dashboard')
            except Student.DoesNotExist:
                pass
            return redirect('dashboard')
        messages.error(request, 'Invalid username or password.')
    return render(request, 'entrance_cam/login.html')


def logout_view(request):
    logout(request)
    return redirect('login')


# ── Dashboard ─────────────────────────────────────────────────────────────────

@login_required
def dashboard(request):
    try:
        request.user.student_profile  # noqa
        return redirect('student_dashboard')
    except Student.DoesNotExist:
        pass

    today = date.today()
    total_students    = Student.objects.filter(is_active=True).count()
    total_cameras     = Camera.objects.filter(is_active=True).count()
    total_fingerprint_devices = ESP32Device.objects.filter(is_active=True).count()
    today_attendance  = AttendanceLog.objects.filter(date=today).count()
    today_fingerprint_attendance = FingerprintAttendance.objects.filter(date=today).count()
    currently_inside  = AttendanceLog.objects.filter(
        date=today, entry_time__isnull=False, exit_time__isnull=True
    ).count()
    # Count students with face encoding (for camera detection)
    enrolled_students = Student.objects.filter(is_active=True).exclude(face_encoding__isnull=True).exclude(face_encoding='').count()

    emotions_today = (AttendanceLog.objects
                      .filter(date=today)
                      .values('entry_emotion')
                      .annotate(count=Count('id')))

    week_data = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        count = AttendanceLog.objects.filter(date=d).count() + FingerprintAttendance.objects.filter(date=d).count()
        week_data.append({'date': d.strftime('%d %b'), 'count': count})

    # Get recent logs from both camera and fingerprint
    recent_camera_logs = (AttendanceLog.objects
                         .select_related('student', 'camera')
                         .order_by('-entry_time')[:5])
    recent_fingerprint_logs = (FingerprintAttendance.objects
                               .select_related('student', 'device')
                               .order_by('-timestamp')[:5])
    
    # Combine and sort recent logs
    recent_logs = []
    for log in recent_camera_logs:
        recent_logs.append({
            'type': 'camera',
            'student': log.student,
            'device': log.camera,
            'time': log.entry_time or log.exit_time,
            'action': 'entry' if log.entry_time else 'exit',
            'emotion': log.entry_emotion if log.entry_time else log.exit_emotion,
            'confidence': None
        })
    for log in recent_fingerprint_logs:
        recent_logs.append({
            'type': 'fingerprint',
            'student': log.student,
            'device': log.device,
            'time': log.timestamp,
            'action': log.attendance_type,
            'emotion': None,
            'confidence': log.confidence
        })
    # Sort by time descending
    recent_logs.sort(key=lambda x: x['time'] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    recent_logs = recent_logs[:10]

    context = {
        'total_students':   total_students,
        'total_cameras':    total_cameras,
        'total_fingerprint_devices': total_fingerprint_devices,
        'today_attendance': today_attendance + today_fingerprint_attendance,
        'today_camera_attendance': today_attendance,
        'today_fingerprint_attendance': today_fingerprint_attendance,
        'currently_inside': currently_inside,
        'enrolled_students': enrolled_students,
        'emotions_today':   list(emotions_today),
        'week_data':        json.dumps(week_data),
        'recent_logs':      recent_logs,
        'today':            today,
    }
    return render(request, 'entrance_cam/dashboard.html', context)


# ── Students ──────────────────────────────────────────────────────────────────

@login_required
def student_list(request):
    q = request.GET.get('q', '').strip()
    students = Student.objects.all()
    if q:
        students = students.filter(
            Q(name__icontains=q) | Q(roll_no__icontains=q) | Q(email__icontains=q)
        )
    return render(request, 'entrance_cam/student_list.html',
                  {'students': students, 'q': q})


@login_required
def student_add(request):
    if request.method == 'POST':
        form = StudentForm(request.POST, request.FILES)
        if form.is_valid():
            student = form.save()
            messages.success(request, f'Student "{student.name}" added. Face encoding will be generated automatically.')
            return redirect('student_list')
    else:
        form = StudentForm()
    return render(request, 'entrance_cam/student_form.html',
                  {'form': form, 'action': 'Add Student'})


@login_required
def student_edit(request, pk):
    student = get_object_or_404(Student, pk=pk)
    if request.method == 'POST':
        form = StudentForm(request.POST, request.FILES, instance=student)
        if form.is_valid():
            form.save()
            messages.success(request, 'Student updated successfully.')
            return redirect('student_list')
    else:
        form = StudentForm(instance=student)
    return render(request, 'entrance_cam/student_form.html',
                  {'form': form, 'action': 'Edit Student', 'student': student})


@login_required
def student_delete(request, pk):
    student = get_object_or_404(Student, pk=pk)
    if request.method == 'POST':
        student.delete()
        messages.success(request, 'Student deleted.')
        return redirect('student_list')
    return render(request, 'entrance_cam/confirm_delete.html',
                  {'obj': student, 'type': 'Student'})


@login_required
def student_detail(request, pk):
    student = get_object_or_404(Student, pk=pk)
    
    # Get both fingerprint and camera attendance logs
    from camera_attendance.models import CameraAttendanceLog
    from datetime import timedelta
    from django.utils import timezone
    
    # Get last 30 days of logs
    thirty_days_ago = timezone.now() - timedelta(days=30)
    
    # Combine both types of logs
    fingerprint_logs = (AttendanceLog.objects
                       .filter(student=student)
                       .order_by('-date', '-entry_time')[:30])
    
    camera_logs = (CameraAttendanceLog.objects
                  .filter(student=student, entry_time__gte=thirty_days_ago)
                  .order_by('-entry_time')[:30])
    
    # Combine and sort by date
    all_logs = list(fingerprint_logs) + list(camera_logs)
    all_logs.sort(key=lambda x: x.entry_time if hasattr(x, 'entry_time') else x.date, reverse=True)
    
    return render(request, 'entrance_cam/student_detail.html',
                  {'student': student, 'logs': all_logs, 'camera_logs': camera_logs})


# ── Cameras ───────────────────────────────────────────────────────────────────

@login_required
def camera_list(request):
    cameras = Camera.objects.all()
    return render(request, 'entrance_cam/camera_list.html', {'cameras': cameras})


@login_required
def camera_add(request):
    if request.method == 'POST':
        form = CameraForm(request.POST)
        if form.is_valid():
            cam = form.save()
            messages.success(request, f'Camera "{cam.name}" added.')
            return redirect('camera_list')
    else:
        form = CameraForm()
    return render(request, 'entrance_cam/camera_form.html',
                  {'form': form, 'action': 'Add Camera'})


@login_required
def camera_edit(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    if request.method == 'POST':
        form = CameraForm(request.POST, instance=camera)
        if form.is_valid():
            form.save()
            messages.success(request, 'Camera updated.')
            return redirect('camera_list')
    else:
        form = CameraForm(instance=camera)
    return render(request, 'entrance_cam/camera_form.html',
                  {'form': form, 'action': 'Edit Camera', 'camera': camera})


@login_required
def camera_proxy_stream(request, pk):
    """
    MJPEG proxy: fetches the camera's HTTP stream server-side and re-serves it
    to the browser over HTTPS.

    WHY THIS IS NEEDED:
    The Django server runs on HTTPS (runsslserver). Browsers enforce Mixed
    Content rules and silently block any http:// resource (images, video)
    loaded from an https:// page. The ESP32-CAM serves on plain HTTP/port 80
    and cannot do TLS natively. This proxy is the correct fix:

        Browser (HTTPS) ←──── Django proxy ────→ ESP32-CAM (HTTP)

    The browser only ever talks to the same HTTPS origin, so no Mixed
    Content warning is raised. Django fetches the HTTP stream internally
    (server-to-server on the same LAN) and streams it straight through.

    Performance note: Django's dev server is single-threaded by default, but
    this view uses a generator + StreamingHttpResponse so it doesn't buffer
    the whole stream into RAM. For production, use gunicorn/uvicorn.
    """
    import requests as req_lib
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    camera = get_object_or_404(Camera, pk=pk)
    url = camera.url.strip()

    # Local webcam — cannot proxy
    if url.isdigit():
        return JsonResponse({'error': 'Local webcam cannot be proxied'}, status=400)

    # Auto-append /stream for bare ESP32-CAM URLs
    if url.lower().startswith('http://') and not any(
        url.lower().endswith(p) for p in ['/stream', '/video', '/mjpeg']
    ):
        import re
        if re.match(r'^https?://[\d.]+/?$', url):
            url = url.rstrip('/') + '/stream'

    def stream_generator(upstream_resp):
        """Yield chunks from the upstream MJPEG response."""
        try:
            for chunk in upstream_resp.iter_content(chunk_size=4096):
                if chunk:
                    yield chunk
        except Exception:
            pass
        finally:
            upstream_resp.close()

    try:
        upstream = req_lib.get(
            url,
            stream=True,
            timeout=(5, None),   # (connect timeout, read timeout=None → stream forever)
            verify=False,
        )
        upstream.raise_for_status()

        content_type = upstream.headers.get('Content-Type', 'multipart/x-mixed-replace;boundary=gc0p4Jq0M2Yt08jU534c0p')
        response = StreamingHttpResponse(
            stream_generator(upstream),
            content_type=content_type,
        )
        response['Cache-Control'] = 'no-store, no-cache, must-revalidate'
        response['Access-Control-Allow-Origin'] = '*'
        return response

    except req_lib.exceptions.ConnectTimeout:
        return JsonResponse({'error': 'Camera connection timed out'}, status=504)
    except req_lib.exceptions.ConnectionError as e:
        return JsonResponse({'error': f'Cannot reach camera: {str(e)[:120]}'}, status=502)
    except Exception as e:
        return JsonResponse({'error': str(e)[:200]}, status=502)


@login_required
def camera_delete(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    if request.method == 'POST':
        camera.delete()
        messages.success(request, 'Camera deleted.')
        return redirect('camera_list')
    return render(request, 'entrance_cam/confirm_delete.html',
                  {'obj': camera, 'type': 'Camera'})


@login_required
def camera_test(request, pk):
    """
    Quick connectivity check for a camera URL.

    FIX: urllib.request.urlopen() hangs forever on MJPEG streams because
    MJPEG is an infinite multipart response — it never "finishes".
    We now use requests with stream=True and just check the HTTP status code
    (and the first few bytes) without consuming the body.

    FIX 2: SSL verification disabled so the Django HTTPS server can reach
    plain HTTP ESP32-CAM streams without SSL errors.

    FIX 3: For webcam indexes (numeric strings like "0"), return a special
    'local' status instead of trying to open them as URLs.
    """
    import requests as req_lib
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    camera = get_object_or_404(Camera, pk=pk)
    url = camera.url.strip()

    # Local webcam index — can't test over HTTP
    if url.isdigit():
        return JsonResponse({'status': 'local', 'message': 'Local webcam — cannot test remotely'})

    try:
        # stream=True: only fetch headers + first chunk, don't download the whole stream
        resp = req_lib.get(
            url,
            timeout=5,
            verify=False,           # allow HTTP cameras from HTTPS Django server
            stream=True,            # CRITICAL: don't try to read infinite MJPEG body
            allow_redirects=True,
        )
        # Read just a tiny bit to confirm data is actually flowing
        first_chunk = next(resp.iter_content(chunk_size=512), b'')
        resp.close()

        if resp.status_code < 400 and len(first_chunk) > 0:
            content_type = resp.headers.get('Content-Type', '')
            return JsonResponse({
                'status': 'online',
                'code': resp.status_code,
                'content_type': content_type,
                'bytes_received': len(first_chunk),
            })
        else:
            return JsonResponse({
                'status': 'offline',
                'error': f'HTTP {resp.status_code} — no data received',
            })
    except req_lib.exceptions.ConnectTimeout:
        return JsonResponse({'status': 'offline', 'error': 'Connection timed out — check IP and WiFi'})
    except req_lib.exceptions.ConnectionError as e:
        return JsonResponse({'status': 'offline', 'error': f'Cannot reach camera: {str(e)[:120]}'})
    except Exception as e:
        return JsonResponse({'status': 'offline', 'error': str(e)[:200]})


# ── Attendance ────────────────────────────────────────────────────────────────

# ── Internal helpers ──────────────────────────────────────────────────────────
# Note: Using shared utilities from attendance_utils.py instead of local helpers


# ── API: log entry / exit (called by detection_script.py) ────────────────────

@csrf_exempt
@csrf_exempt
@transaction.atomic
def api_log_entry(request):
    """
    POST  /api/log/

    Expected JSON body:
        student_id  : int       — Student PK
        camera_id   : int       — Camera PK
        emotion     : str       — e.g. 'happy', 'neutral'  (default 'unknown')
        score       : float     — confidence 0.0–1.0        (default 0.0)
        snapshot    : str|null  — base64 JPEG               (optional)

    Logic:
        • If today already has an open log (entry present, exit NULL)
          for this student  →  record EXIT.
        • Otherwise  →  record ENTRY.

    This correctly handles multiple in/out visits per day.
    """
    if request.method != 'POST':
        logger.warning(f"Invalid method: {request.method}")
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Invalid JSON body: {e}")
        return JsonResponse({'error': 'Invalid JSON body'}, status=400)

    # ── Validate required fields ──────────────────────────────────────────────
    student_id = data.get('student_id')
    camera_id  = data.get('camera_id')
    if student_id is None or camera_id is None:
        logger.warning(f"Missing required fields: student_id={student_id}, camera_id={camera_id}")
        return JsonResponse(
            {'error': 'student_id and camera_id are required'}, status=400
        )

    try:
        student = Student.objects.get(pk=student_id, is_active=True)
    except Student.DoesNotExist:
        logger.warning(f"Student {student_id} not found or inactive")
        return JsonResponse({'error': f'Student {student_id} not found or inactive'}, status=404)

    try:
        camera = Camera.objects.get(pk=camera_id, is_active=True)
    except Camera.DoesNotExist:
        logger.warning(f"Camera {camera_id} not found or inactive")
        return JsonResponse({'error': f'Camera {camera_id} not found or inactive'}, status=404)

    # ── Parse optional fields ─────────────────────────────────────────────────
    try:
        today        = date.today()
        emotion      = validate_emotion(data.get('emotion'))
        score        = clamp_score(float(data.get('score') or 0.0))
        snapshot_b64 = data.get('snapshot')

        # ── Decide ENTRY or EXIT ──────────────────────────────────────────────────
        open_log = (AttendanceLog.objects
                    .filter(student=student, date=today,
                            entry_time__isnull=False,
                            exit_time__isnull=True)
                    .order_by('-entry_time')
                    .first())

        if open_log:
            # ── EXIT ──────────────────────────────────────────────────────────────
            open_log.exit_time          = timezone.now()
            open_log.exit_emotion       = emotion
            open_log.exit_emotion_score = score
            open_log.mood_comparison    = mood_comparison(open_log.entry_emotion, emotion)

            snapshot_file = decode_snapshot(snapshot_b64, student.roll_no, 'exit')
            if snapshot_file:
                open_log.exit_snapshot = snapshot_file

            # Calculate duration
            if open_log.entry_time:
                delta = open_log.exit_time - open_log.entry_time
                open_log.duration_minutes = int(delta.total_seconds() // 60)

            open_log.save()

            logger.info(f"EXIT: {student.name} | emotion={emotion} | "
                       f"duration={open_log.duration_minutes} min | "
                       f"mood={open_log.mood_comparison}")

            return JsonResponse({
                'status':          'exit_logged',
                'student':         student.name,
                'duration':        open_log.duration_minutes,
                'mood_comparison': open_log.mood_comparison,
                'exit_emotion':    emotion,
            })

        else:
            # ── ENTRY ─────────────────────────────────────────────────────────────
            log = AttendanceLog(
                student             = student,
                camera              = camera,
                date                = today,
                entry_time          = timezone.now(),
                entry_emotion       = emotion,
                entry_emotion_score = score,
            )
            snapshot_file = decode_snapshot(snapshot_b64, student.roll_no, 'entry')
            if snapshot_file:
                log.entry_snapshot = snapshot_file

            log.save()

            logger.info(f"ENTRY: {student.name} | emotion={emotion} | score={score:.2f}")

            return JsonResponse({
                'status':         'entry_logged',
                'student':        student.name,
                'entry_emotion':  emotion,
            })
    
    except Exception as e:
        logger.error(f"Error logging attendance: {e}", exc_info=True)
        return JsonResponse({'error': 'Internal server error'}, status=500)


# ── API: student face encodings (called by detection_script.py) ──────────────

@csrf_exempt
@csrf_exempt
def api_students_encodings(request):
    """
    GET  /api/students/encodings/

    Returns JSON list of active students that have a face encoding:
        [ { "id": 1, "name": "Ravi Kumar", "encoding": "[0.123, ...]" }, ... ]
    """
    if request.method != 'GET':
        logger.warning(f"Invalid method: {request.method}")
        return JsonResponse({'error': 'GET only'}, status=405)

    try:
        students = (Student.objects
                    .filter(is_active=True, is_enrolled=True)
                    .exclude(face_encoding__isnull=True)
                    .exclude(face_encoding=''))

        data = [
            {'id': s.id, 'name': s.name, 'encoding': s.face_encoding}
            for s in students
        ]
        logger.info(f"Retrieved {len(data)} students with face encodings")
        return JsonResponse(data, safe=False)
    except Exception as e:
        logger.error(f"Error retrieving student encodings: {e}", exc_info=True)
        return JsonResponse({'error': 'Internal server error'}, status=500)
@csrf_exempt
def api_live_detections(request):
    """
    GET  /api/live-detections/?camera_id=<pk>

    Returns today's attendance logs for a specific camera so the
    Live modal can populate the right-hand detections panel.

    Response:
    {
      "today_total":  12,
      "inside_now":    3,
      "logs": [
        {
          "student_name":   "Ravi Kumar",
          "entry_time":     "2025-06-01T09:14:00",
          "exit_time":      null,
          "entry_emotion":  "happy",
          "exit_emotion":   "neutral",
          "mood_comparison":"improved",
          "duration":       null
        }, ...
      ]
    }
    """
    if request.method != 'GET':
        return JsonResponse({'error': 'GET only'}, status=405)

    camera_id = request.GET.get('camera_id')
    today     = date.today()

    qs = AttendanceLog.objects.filter(date=today).select_related('student')
    if camera_id:
        qs = qs.filter(camera_id=camera_id)
    qs = qs.order_by('-entry_time')[:50]

    logs = []
    for log in qs:
        logs.append({
            'student_name':    log.student.name,
            'roll_no':         log.student.roll_no,
            'entry_time':      log.entry_time.isoformat()  if log.entry_time  else None,
            'exit_time':       log.exit_time.isoformat()   if log.exit_time   else None,
            'entry_emotion':   log.entry_emotion,
            'exit_emotion':    log.exit_emotion,
            'mood_comparison': log.mood_comparison,
            'duration':        log.duration_minutes,
        })

    inside_now = AttendanceLog.objects.filter(
        date=today,
        entry_time__isnull=False,
        exit_time__isnull=True,
    ).count()

    return JsonResponse({
        'today_total': qs.count(),
        'inside_now':  inside_now,
        'logs':        logs,
    })


# ── ESP32 Devices ─────────────────────────────────────────────────────────────────

@login_required
def esp32_device_list(request):
    devices = ESP32Device.objects.all()
    return render(request, 'entrance_cam/esp32_device_list.html', {'devices': devices})


@login_required
def esp32_device_add(request):
    if request.method == 'POST':
        form = ESP32DeviceForm(request.POST)
        if form.is_valid():
            device = form.save()
            messages.success(request, f'ESP32 Device "{device.name}" added. API Key: {device.api_key}')
            return redirect('esp32_device_list')
    else:
        form = ESP32DeviceForm()
    return render(request, 'entrance_cam/esp32_device_form.html', {'form': form, 'action': 'Add ESP32 Device'})


@login_required
def esp32_device_edit(request, pk):
    device = get_object_or_404(ESP32Device, pk=pk)
    if request.method == 'POST':
        form = ESP32DeviceForm(request.POST, instance=device)
        if form.is_valid():
            form.save()
            messages.success(request, 'ESP32 Device updated.')
            return redirect('esp32_device_list')
    else:
        form = ESP32DeviceForm(instance=device)
    return render(request, 'entrance_cam/esp32_device_form.html', {'form': form, 'action': 'Edit ESP32 Device', 'device': device})


@login_required
def esp32_device_delete(request, pk):
    device = get_object_or_404(ESP32Device, pk=pk)
    if request.method == 'POST':
        device.delete()
        messages.success(request, 'ESP32 Device deleted.')
        return redirect('esp32_device_list')
    return render(request, 'entrance_cam/confirm_delete.html', {'obj': device, 'type': 'ESP32 Device'})


# ── Fingerprint Enrollment ────────────────────────────────────────────────────────

def next_fingerprint_slot():
    used = set(Student.objects.filter(fingerprint_id__isnull=False).values_list('fingerprint_id', flat=True))
    for slot in range(1, 128):
        if slot not in used:
            return slot
    raise RuntimeError("Fingerprint sensor is full (127 max)")


@login_required
@require_POST
def enroll_fingerprint(request, student_id):
    student = get_object_or_404(Student, pk=student_id)
    try:
        fp_slot = next_fingerprint_slot()
    except RuntimeError as e:
        return JsonResponse({'ok': False, 'error': str(e)}, status=400)

    # Only set fingerprint_id — do NOT touch is_enrolled (that's for face recognition)
    student.fingerprint_id = fp_slot
    student.save()

    cache.set('esp32_command', {'command': 'ENROLL', 'fingerprint_id': fp_slot}, timeout=120)
    return JsonResponse({'ok': True, 'slot': fp_slot})   # ← JSON, not redirect

@login_required
def enrollment_status(request, student_id):
    try:
        student = get_object_or_404(Student, pk=student_id)
        # Fingerprint is enrolled if fingerprint_id is set
        is_fingerprint_enrolled = bool(student.fingerprint_id)
        return JsonResponse({'is_enrolled': is_fingerprint_enrolled})
    except Exception as e:
        import traceback
        print(f"ERROR in enrollment_status: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def view_fingerprint_image(request, student_id):
    student = get_object_or_404(Student, pk=student_id)
    if student.fp_image:
        return redirect(student.fp_image.url)
    return JsonResponse({'error': 'No fingerprint image available'}, status=404)


# ── Updated Attendance View ───────────────────────────────────────────────────────

@login_required
def attendance_list(request):
    selected_date = request.GET.get('date', str(date.today()))
    try:
        filter_date = date.fromisoformat(selected_date)
    except ValueError:
        filter_date = date.today()

    # Get both camera and fingerprint attendance
    camera_logs = (AttendanceLog.objects
        .filter(date=filter_date)
        .select_related('student', 'camera')
        .order_by('-entry_time'))

    fingerprint_logs = (FingerprintAttendance.objects
        .filter(date=filter_date)
        .select_related('student', 'device')
        .order_by('-timestamp'))

    # Combine and sort all logs by timestamp
    all_logs = []
    for log in camera_logs:
        all_logs.append({
            'type': 'camera',
            'student': log.student,
            'device': log.camera,
            'timestamp': log.entry_time or log.exit_time,
            'action': 'entry' if log.entry_time else 'exit',
            'emotion': log.entry_emotion if log.entry_time else log.exit_emotion,
        })

    for log in fingerprint_logs:
        all_logs.append({
            'type': 'fingerprint',
            'student': log.student,
            'device': log.device,
            'timestamp': log.timestamp,
            'action': log.attendance_type,
            'confidence': log.confidence,
        })

    # Sort by timestamp descending
    all_logs.sort(key=lambda x: x['timestamp'] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    return render(request, 'entrance_cam/attendance_list.html', {
        'all_logs': all_logs,
        'camera_logs': camera_logs,
        'fingerprint_logs': fingerprint_logs,
        'filter_date': filter_date,
        'selected_date': selected_date,
    })


# ── ESP32 API Endpoints ───────────────────────────────────────────────────────────

def esp32_auth(f):
    @wraps(f)
    def wrapper(request, *args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or (api_key != ESP32_API_KEY and not ESP32Device.objects.filter(api_key=api_key).exists()):
            return JsonResponse({'error': 'unauthorized'}, status=401)
        return f(request, *args, **kwargs)
    return wrapper


@csrf_exempt
@esp32_auth
def api_esp32_command(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'GET only'}, status=405)

    cmd = cache.get('esp32_command')
    if cmd and cmd.get('command'):
        return JsonResponse({
            'command': cmd['command'],
            'fingerprint_id': cmd['fingerprint_id']
        })
    return JsonResponse({'command': None})


@csrf_exempt
@esp32_auth
def api_esp32_enroll_result(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    fp_id = data.get('fingerprint_id')
    success = data.get('success', False)

    student = Student.objects.filter(fingerprint_id=fp_id).first()
    if student:
        # If enrollment failed on ESP32, clear the fingerprint_id
        if not success:
            student.fingerprint_id = None
            student.save()
        # If success, we wait for fp_image to be uploaded before marking as fully enrolled
        # The enrollment_status endpoint will check both fingerprint_id and fp_image

    # Clear the command now that ESP32 has confirmed it received and acted on it
    cache.delete('esp32_command')

    return JsonResponse({'ok': True})


@csrf_exempt
@esp32_auth
def api_esp32_upload_image(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[API] JSON parse error: {e}")
        print(f"[API] Request body length: {len(request.body)}")
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    fp_id = data.get('fingerprint_id')
    b64_image = data.get('image', '')

    if not fp_id:
        print(f"[API] Missing fingerprint_id")
        return JsonResponse({'ok': False, 'error': 'missing fingerprint_id'}, status=400)

    if not b64_image:
        print(f"[API] Missing image data")
        return JsonResponse({'ok': False, 'error': 'missing image'}, status=400)

    print(f"[API] Received fp_id={fp_id}, image length={len(b64_image)}")

    student = Student.objects.filter(fingerprint_id=fp_id).first()
    if not student:
        print(f"[API] Student not found for fp_id={fp_id}")
        return JsonResponse({'ok': False, 'error': 'student not found'}, status=404)

    try:
        raw = base64.b64decode(b64_image)
        print(f"[API] Base64 decoded: {len(raw)} bytes")
    except Exception as e:
        print(f"[API] Base64 decode error: {e}")
        return JsonResponse({'ok': False, 'error': 'invalid base64'}, status=400)

    if len(raw) != 36864:
        print(f"[API] Image size mismatch: got {len(raw)}, expected 36864")
        return JsonResponse({'ok': False, 'error': f'image size mismatch: {len(raw)}'}, status=400)

    # Convert 4-bit grayscale to 8-bit and create PNG
    W, H = 256, 288
    pixels = bytearray(W * H)
    for i, byte in enumerate(raw[:W * H // 2]):
        pixels[i * 2] = (byte >> 4) * 17
        pixels[i * 2 + 1] = (byte & 0x0F) * 17

    # Create PNG with correct IHDR
    def png_chunk(tag, chunk_data):
        crc = zlib.crc32(tag + chunk_data) & 0xFFFFFFFF
        return struct.pack(">I", len(chunk_data)) + tag + chunk_data + struct.pack(">I", crc)

    sig = b'\x89PNG\r\n\x1a\n'
    # IHDR: width(4) height(4) bitdepth(1) colortype(1) compression(1) filter(1) interlace(1)
    ihdr = png_chunk(b'IHDR', struct.pack(">IIBBBBB", W, H, 8, 0, 0, 0, 0))
    raw_rows = b''.join(b'\x00' + bytes(pixels[r*W:(r+1)*W]) for r in range(H))
    idat = png_chunk(b'IDAT', zlib.compress(raw_rows, 6))
    iend = png_chunk(b'IEND', b'')
    png_data = sig + ihdr + idat + iend

    # Save to student's fp_image field
    filename = f'fp_{student.id}_{fp_id}.png'
    student.fp_image.save(filename, ContentFile(png_data), save=True)
    print(f"[API] Image saved: {filename} ({len(png_data)} bytes)")

    return JsonResponse({'ok': True, 'file': filename})


@csrf_exempt
@csrf_exempt
@esp32_auth
def api_mark_fingerprint_attendance(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    fp_id = data.get('fingerprint_id')
    confidence = data.get('confidence', 0)

    # Student must have fingerprint_id set
    student = Student.objects.filter(fingerprint_id=fp_id).first()
    if not student:
        print(f"[API] Student not found for fp_id={fp_id}")
        return JsonResponse({'error': 'not_found'}, status=404)

    # Determine if it's entry or exit
    last_log = FingerprintAttendance.objects.filter(student=student, date=date.today()).order_by('-timestamp').first()
    action = 'exit' if last_log and last_log.attendance_type == 'entry' else 'entry'

    now = timezone.now()
    log = FingerprintAttendance.objects.create(
        student=student,
        attendance_type=action,
        timestamp=now,
        date=date.today(),
        fingerprint_id=fp_id,
        confidence=confidence
    )

    # Update student stats
    student.fp_confidence = confidence
    student.fp_scan_count += 1
    student.fp_last_seen = now
    student.save()

    print(f"[API] Attendance marked: {student.name} ({action}) - confidence {confidence}")

    return JsonResponse({
        'ok': True,
        'name': student.name,
        'roll_no': student.roll_no,
        'action': 'IN' if action == 'entry' else 'OUT',
        'time': now.strftime('%H:%M'),
        'confidence': confidence
    })