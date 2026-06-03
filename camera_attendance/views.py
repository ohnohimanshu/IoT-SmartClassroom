"""
Camera Attendance Views
Handles all camera-based attendance marking
"""
import json
import base64
import logging
from datetime import date
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from django.db.models import Count
from django.db import transaction

from entrance_cam.models import Student, Camera
from .models import CameraAttendanceLog

logger = logging.getLogger(__name__)

# ── Try importing shared utils; provide safe fallbacks if absent ──────────────

try:
    from attendance_utils import mood_comparison as _mood_comparison
except ImportError:
    def _mood_comparison(entry_emotion, exit_emotion):
        """
        Compare entry vs exit emotion.
        Returns 'improved', 'declined', or 'stable'.
        """
        POSITIVE  = {'happy', 'surprise'}
        NEGATIVE  = {'sad', 'angry', 'fear', 'disgust'}
        NEUTRAL   = {'neutral'}

        def valence(e):
            if e in POSITIVE: return 1
            if e in NEGATIVE: return -1
            return 0

        diff = valence(exit_emotion) - valence(entry_emotion)
        if diff > 0:  return 'improved'
        if diff < 0:  return 'declined'
        return 'stable'

try:
    from attendance_utils import decode_snapshot as _decode_snapshot
except ImportError:
    import os
    from io import BytesIO
    from django.core.files.base import ContentFile

    def _decode_snapshot(snapshot_b64, roll_no, tag):
        """Decode base64 image and return a Django ContentFile or None."""
        if not snapshot_b64:
            return None
        try:
            img_data = base64.b64decode(snapshot_b64)
            filename = f"cam_{tag}_{roll_no}_{date.today().strftime('%Y%m%d_%H%M%S')}.jpg"
            return ContentFile(img_data, name=filename)
        except Exception as e:
            logger.warning(f"decode_snapshot failed: {e}")
            return None

try:
    from attendance_utils import validate_emotion as _validate_emotion
except ImportError:
    _VALID_EMOTIONS = {'happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust', 'unknown'}

    def _validate_emotion(emotion):
        if emotion and str(emotion).lower() in _VALID_EMOTIONS:
            return str(emotion).lower()
        return 'unknown'

try:
    from attendance_utils import clamp_score as _clamp_score
except ImportError:
    def _clamp_score(score):
        try:
            return max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            return 0.0


# ── API: log entry/exit ───────────────────────────────────────────────────────

@csrf_exempt
@require_POST
@transaction.atomic
def api_log_camera_attendance(request):
    """
    POST /camera-attendance/api/log/

    Body (JSON):
        {
            "student_id": <int>,
            "camera_id":  <int>,
            "emotion":    <str>,   # happy|sad|angry|neutral|surprise|fear|disgust|unknown
            "score":      <float>, # 0.0–1.0
            "snapshot":   <str>    # base64 JPEG (optional)
        }

    Logic:
        • If the student has an OPEN log today (entry without exit) → records EXIT.
        • Otherwise → records ENTRY.
        • Mood comparison is stored on exit.
    """
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Invalid JSON body: {e}")
        return JsonResponse({'error': 'Invalid JSON body'}, status=400)

    student_id = data.get('student_id')
    camera_id  = data.get('camera_id')

    if student_id is None or camera_id is None:
        return JsonResponse(
            {'error': 'student_id and camera_id are required'},
            status=400,
        )

    # Fetch student
    try:
        student = Student.objects.get(pk=student_id, is_active=True)
    except Student.DoesNotExist:
        return JsonResponse(
            {'error': f'Student {student_id} not found or inactive'},
            status=404,
        )

    if not student.face_encoding:
        return JsonResponse(
            {'error': f'Student {student.name} has no face encoding. '
                      'Upload photo and regenerate encodings in admin.'},
            status=400,
        )

    # Fetch camera
    try:
        camera = Camera.objects.get(pk=camera_id, is_active=True)
    except Camera.DoesNotExist:
        return JsonResponse(
            {'error': f'Camera {camera_id} not found or inactive'},
            status=404,
        )

    try:
        today        = date.today()
        emotion      = _validate_emotion(data.get('emotion', 'unknown'))
        score        = _clamp_score(data.get('score', 0.0))
        snapshot_b64 = data.get('snapshot')

        # Look for an open (entry-only) log for this student today
        open_log = (
            CameraAttendanceLog.objects
            .filter(
                student=student,
                date=today,
                entry_time__isnull=False,
                exit_time__isnull=True,
            )
            .order_by('-entry_time')
            .first()
        )

        if open_log:
            # ── EXIT ──────────────────────────────────────────────────────────
            open_log.exit_time          = timezone.now()
            open_log.exit_emotion       = emotion
            open_log.exit_emotion_score = score

            # Compute and STORE mood comparison
            open_log.mood_comparison = _mood_comparison(open_log.entry_emotion, emotion)

            # Duration
            if open_log.entry_time:
                delta = open_log.exit_time - open_log.entry_time
                open_log.duration_minutes = max(0, int(delta.total_seconds() // 60))

            # Exit snapshot
            exit_file = _decode_snapshot(snapshot_b64, student.roll_no, 'exit')
            if exit_file:
                open_log.exit_snapshot = exit_file

            open_log.is_present = True
            open_log.save()

            logger.info(
                f"CAMERA-EXIT  | {student.name} | emotion={emotion} score={score:.2f}"
                f" | duration={open_log.duration_minutes}min | mood={open_log.mood_comparison}"
            )

            return JsonResponse({
                'status':          'exit_logged',
                'student':         student.name,
                'exit_emotion':    emotion,
                'duration':        open_log.duration_minutes,
                'mood_comparison': open_log.mood_comparison,
            })

        else:
            # ── ENTRY ─────────────────────────────────────────────────────────
            log = CameraAttendanceLog(
                student            = student,
                camera             = camera,
                date               = today,
                entry_time         = timezone.now(),
                entry_emotion      = emotion,
                entry_emotion_score= score,
                is_present         = True,
            )

            entry_file = _decode_snapshot(snapshot_b64, student.roll_no, 'entry')
            if entry_file:
                log.entry_snapshot = entry_file

            log.save()

            logger.info(
                f"CAMERA-ENTRY | {student.name} | emotion={emotion} score={score:.2f}"
            )

            return JsonResponse({
                'status':        'entry_logged',
                'student':       student.name,
                'entry_emotion': emotion,
            })

    except Exception as e:
        logger.error(f"Error logging camera attendance: {e}", exc_info=True)
        return JsonResponse({'error': 'Internal server error'}, status=500)


# ── API: student encodings ────────────────────────────────────────────────────

@csrf_exempt
def api_camera_students_encodings(request):
    """
    GET /camera-attendance/api/students/encodings/
    Returns active students that have face encodings.
    """
    if request.method != 'GET':
        return JsonResponse({'error': 'GET only'}, status=405)

    try:
        students = (
            Student.objects
            .filter(is_active=True)
            .exclude(face_encoding__isnull=True)
            .exclude(face_encoding='')
        )
        data = [
            {
                'id':       s.id,
                'name':     s.name,
                'roll_no':  s.roll_no,
                'encoding': s.face_encoding,
            }
            for s in students
        ]
        logger.info(f"Served {len(data)} student encodings")
        return JsonResponse(data, safe=False)
    except Exception as e:
        logger.error(f"Error retrieving student encodings: {e}", exc_info=True)
        return JsonResponse({'error': 'Internal server error'}, status=500)


# ── API: live detections ──────────────────────────────────────────────────────

@csrf_exempt
def api_camera_live_detections(request):
    """
    GET /camera-attendance/api/live-detections/?camera_id=<pk>
    Returns today's attendance logs for the live feed panel.
    """
    if request.method != 'GET':
        return JsonResponse({'error': 'GET only'}, status=405)

    try:
        camera_id = request.GET.get('camera_id')
        today     = date.today()

        qs = CameraAttendanceLog.objects.filter(date=today).select_related('student')
        if camera_id:
            qs = qs.filter(camera_id=camera_id)

        total_today = qs.count()
        inside_now  = qs.filter(exit_time__isnull=True).count()

        logs = []
        for log in qs.order_by('-entry_time')[:50]:
            logs.append({
                'student_id':      log.student.id,       # needed by detection script restart
                'student_name':    log.student.name,
                'roll_no':         log.student.roll_no,
                'entry_time':      log.entry_time.isoformat()  if log.entry_time  else None,
                'exit_time':       log.exit_time.isoformat()   if log.exit_time   else None,
                'entry_emotion':   log.entry_emotion,
                'exit_emotion':    log.exit_emotion,
                'entry_score':     log.entry_emotion_score,
                'exit_score':      log.exit_emotion_score,
                'mood_comparison': log.mood_comparison,
                'duration':        log.duration_minutes,
                'is_present':      log.is_present,
            })

        return JsonResponse({
            'today_total': total_today,
            'inside_now':  inside_now,
            'logs':        logs,
        })
    except Exception as e:
        logger.error(f"Error retrieving live detections: {e}", exc_info=True)
        return JsonResponse({'error': 'Internal server error'}, status=500)


# ── Views ─────────────────────────────────────────────────────────────────────

@login_required
def camera_attendance_list(request):
    """Camera attendance log list with date filter & pagination."""
    from django.core.paginator import Paginator

    try:
        selected_date = request.GET.get('date', str(date.today()))
        try:
            filter_date = timezone.datetime.strptime(selected_date, '%Y-%m-%d').date()
        except (ValueError, TypeError):
            filter_date = date.today()

        logs = (
            CameraAttendanceLog.objects
            .filter(date=filter_date)
            .select_related('student', 'camera')
            .order_by('-entry_time')
        )

        paginator  = Paginator(logs, 50)
        page_obj   = paginator.get_page(request.GET.get('page', 1))

        context = {
            'page_obj':         page_obj,
            'logs':             page_obj.object_list,
            'selected_date':    filter_date,
            'total_entries':    logs.filter(entry_time__isnull=False).count(),
            'total_exits':      logs.filter(exit_time__isnull=False).count(),
            'currently_inside': logs.filter(exit_time__isnull=True).count(),
        }
        return render(request, 'camera_attendance/attendance_list.html', context)

    except Exception as e:
        logger.error(f"Error displaying attendance list: {e}", exc_info=True)
        return render(request, 'camera_attendance/attendance_list.html', {
            'logs': [], 'page_obj': None,
            'selected_date': date.today(),
            'total_entries': 0, 'total_exits': 0, 'currently_inside': 0,
        })


@login_required
def camera_attendance_dashboard(request):
    """Dashboard with live stats and emotion distribution."""
    from django.core.cache import cache

    try:
        cache_key = f'camera_dashboard_{request.user.id}'
        context   = cache.get(cache_key)

        if context is None:
            today = date.today()

            total_students = Student.objects.filter(is_active=True).count()
            students_with_encoding = (
                Student.objects
                .filter(is_active=True)
                .exclude(face_encoding__isnull=True)
                .exclude(face_encoding='')
                .count()
            )

            today_qs          = CameraAttendanceLog.objects.filter(date=today)
            today_entries     = today_qs.filter(entry_time__isnull=False).count()
            currently_inside  = today_qs.filter(entry_time__isnull=False, exit_time__isnull=True).count()

            emotions_today = list(
                today_qs.values('entry_emotion').annotate(count=Count('id'))
            )

            # Mood summary for today
            mood_summary = list(
                today_qs
                .exclude(mood_comparison='unknown')
                .exclude(mood_comparison='')
                .values('mood_comparison')
                .annotate(count=Count('id'))
            )

            recent_logs = list(
                today_qs
                .select_related('student', 'camera')
                .order_by('-entry_time')[:10]
            )

            context = {
                'total_students':        total_students,
                'students_with_encoding': students_with_encoding,
                'today_entries':          today_entries,
                'currently_inside':       currently_inside,
                'emotions_today':         emotions_today,
                'mood_summary':           mood_summary,
                'recent_logs':            recent_logs,
            }
            cache.set(cache_key, context, 60)

        return render(request, 'camera_attendance/dashboard.html', context)

    except Exception as e:
        logger.error(f"Error loading dashboard: {e}", exc_info=True)
        return render(request, 'camera_attendance/dashboard.html', {
            'total_students': 0, 'students_with_encoding': 0,
            'today_entries': 0, 'currently_inside': 0,
            'emotions_today': [], 'mood_summary': [], 'recent_logs': [],
        })