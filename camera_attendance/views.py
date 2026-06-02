"""
Camera Attendance Views
Handles all camera-based attendance marking
"""
import json
import base64
import logging
from datetime import date
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from django.db.models import Count, Q
from django.db import transaction

from entrance_cam.models import Student, Camera
from .models import CameraAttendanceLog
from attendance_utils import mood_comparison, decode_snapshot, validate_emotion, clamp_score

logger = logging.getLogger(__name__)


@csrf_exempt
@transaction.atomic
def api_log_camera_attendance(request):
    """
    POST /camera-attendance/api/log/
    
    Log camera-based attendance (entry/exit).
    
    Expected JSON:
        {
            "student_id": 1,
            "camera_id": 1,
            "emotion": "happy",
            "score": 0.95,
            "snapshot": "base64_encoded_image"
        }
    """
    if request.method != 'POST':
        logger.warning(f"Invalid method: {request.method}")
        return JsonResponse({'error': 'POST only'}, status=405)
    
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Invalid JSON body: {e}")
        return JsonResponse({'error': 'Invalid JSON body'}, status=400)
    
    # Validate required fields
    student_id = data.get('student_id')
    camera_id = data.get('camera_id')
    
    if student_id is None or camera_id is None:
        logger.warning(f"Missing required fields: student_id={student_id}, camera_id={camera_id}")
        return JsonResponse(
            {'error': 'student_id and camera_id are required'},
            status=400
        )
    
    # Get student
    try:
        student = Student.objects.get(pk=student_id, is_active=True)
    except Student.DoesNotExist:
        logger.warning(f"Student {student_id} not found or inactive")
        return JsonResponse(
            {'error': f'Student {student_id} not found or inactive'},
            status=404
        )
    
    # Check if student has face encoding for camera detection
    if not student.face_encoding:
        logger.warning(f"Student {student_id} ({student.name}) has no face encoding")
        return JsonResponse(
            {'error': f'Student {student.name} has no face encoding. Upload photo and regenerate encodings in admin.'},
            status=400
        )
    
    # Get camera
    try:
        camera = Camera.objects.get(pk=camera_id, is_active=True)
    except Camera.DoesNotExist:
        logger.warning(f"Camera {camera_id} not found or inactive")
        return JsonResponse(
            {'error': f'Camera {camera_id} not found or inactive'},
            status=404
        )
    
    try:
        # Parse optional fields
        today = date.today()
        emotion = validate_emotion(data.get('emotion'))
        score = clamp_score(float(data.get('score', 0.0)))
        snapshot_b64 = data.get('snapshot')
        
        # Check for open log (entry without exit)
        open_log = (CameraAttendanceLog.objects
                    .filter(
                        student=student,
                        date=today,
                        entry_time__isnull=False,
                        exit_time__isnull=True
                    )
                    .order_by('-entry_time')
                    .first())
        
        if open_log:
            # ── EXIT ──────────────────────────────────────────────────────────
            open_log.exit_time = timezone.now()
            open_log.exit_emotion = emotion
            open_log.exit_emotion_score = score
            open_log.mood_comparison = mood_comparison(open_log.entry_emotion, emotion)
            
            # Save snapshot
            snapshot_file = decode_snapshot(snapshot_b64, student.roll_no, 'exit')
            if snapshot_file:
                open_log.exit_snapshot = snapshot_file
            
            # Calculate duration
            if open_log.entry_time:
                delta = open_log.exit_time - open_log.entry_time
                open_log.duration_minutes = int(delta.total_seconds() // 60)
            
            open_log.save()
            
            logger.info(f"CAMERA-EXIT: {student.name} | emotion={emotion} | "
                       f"duration={open_log.duration_minutes} min | "
                       f"mood={open_log.mood_comparison}")
            
            return JsonResponse({
                'status': 'exit_logged',
                'student': student.name,
                'duration': open_log.duration_minutes,
                'mood_comparison': open_log.mood_comparison,
                'exit_emotion': emotion,
            })
        
        else:
            # ── ENTRY ─────────────────────────────────────────────────────────
            log = CameraAttendanceLog(
                student=student,
                camera=camera,
                date=today,
                entry_time=timezone.now(),
                entry_emotion=emotion,
                entry_emotion_score=score,
            )
            
            # Save snapshot
            snapshot_file = decode_snapshot(snapshot_b64, student.roll_no, 'entry')
            if snapshot_file:
                log.entry_snapshot = snapshot_file
            
            log.save()
            
            logger.info(f"CAMERA-ENTRY: {student.name} | emotion={emotion} | score={score:.2f}")
            
            return JsonResponse({
                'status': 'entry_logged',
                'student': student.name,
                'entry_emotion': emotion,
            })
    
    except Exception as e:
        logger.error(f"Error logging camera attendance: {e}", exc_info=True)
        return JsonResponse({'error': 'Internal server error'}, status=500)


@csrf_exempt
def api_camera_students_encodings(request):
    """
    GET /camera-attendance/api/students/encodings/
    
    Returns list of students with face encodings for camera detection.
    """
    if request.method != 'GET':
        logger.warning(f"Invalid method: {request.method}")
        return JsonResponse({'error': 'GET only'}, status=405)
    
    try:
        students = (Student.objects
                    .filter(is_active=True)
                    .exclude(face_encoding__isnull=True)
                    .exclude(face_encoding=''))
        
        data = [
            {'id': s.id, 'name': s.name, 'roll_no': s.roll_no, 'encoding': s.face_encoding}
            for s in students
        ]
        logger.info(f"Retrieved {len(data)} student encodings")
        return JsonResponse(data, safe=False)
    except Exception as e:
        logger.error(f"Error retrieving student encodings: {e}", exc_info=True)
        return JsonResponse({'error': 'Internal server error'}, status=500)


@csrf_exempt
def api_camera_live_detections(request):
    """
    GET /camera-attendance/api/live-detections/?camera_id=<pk>
    
    Returns today's camera attendance for a specific camera.
    """
    if request.method != 'GET':
        logger.warning(f"Invalid method: {request.method}")
        return JsonResponse({'error': 'GET only'}, status=405)
    
    try:
        camera_id = request.GET.get('camera_id')
        today = date.today()
        
        qs = CameraAttendanceLog.objects.filter(date=today).select_related('student')
        if camera_id:
            qs = qs.filter(camera_id=camera_id)
        qs = qs.order_by('-entry_time')[:50]
        
        logs = []
        for log in qs:
            logs.append({
                'student_name': log.student.name,
                'roll_no': log.student.roll_no,
                'entry_time': log.entry_time.isoformat() if log.entry_time else None,
                'exit_time': log.exit_time.isoformat() if log.exit_time else None,
                'entry_emotion': log.entry_emotion,
                'exit_emotion': log.exit_emotion,
                'mood_comparison': log.mood_comparison,
                'duration': log.duration_minutes,
            })
        
        logger.info(f"Retrieved {len(logs)} live detections for camera {camera_id}")
        return JsonResponse({
            'today_total': qs.count(),
            'inside_now': qs.filter(exit_time__isnull=True).count(),
            'logs': logs,
        })
    except Exception as e:
        logger.error(f"Error retrieving live detections: {e}", exc_info=True)
        return JsonResponse({'error': 'Internal server error'}, status=500)


@login_required
def camera_attendance_list(request):
    """
    Display camera attendance logs with pagination and filtering.
    """
    from django.core.paginator import Paginator
    
    try:
        selected_date = request.GET.get('date', str(date.today()))
        try:
            filter_date = timezone.datetime.strptime(selected_date, '%Y-%m-%d').date()
        except (ValueError, TypeError):
            filter_date = date.today()
        
        logs = (CameraAttendanceLog.objects
                .filter(date=filter_date)
                .select_related('student', 'camera')
                .order_by('-entry_time'))
        
        # Pagination
        paginator = Paginator(logs, 50)
        page_number = request.GET.get('page', 1)
        page_obj = paginator.get_page(page_number)
        
        context = {
            'page_obj': page_obj,
            'logs': page_obj.object_list,
            'selected_date': filter_date,
            'total_entries': logs.filter(entry_time__isnull=False).count(),
            'total_exits': logs.filter(exit_time__isnull=False).count(),
            'currently_inside': logs.filter(exit_time__isnull=True).count(),
        }
        
        logger.info(f"Displayed camera attendance for {filter_date}")
        return render(request, 'camera_attendance/attendance_list.html', context)
    
    except Exception as e:
        logger.error(f"Error displaying camera attendance list: {e}", exc_info=True)
        return render(request, 'camera_attendance/attendance_list.html', {
            'logs': [],
            'page_obj': None,
            'selected_date': date.today(),
            'total_entries': 0,
            'total_exits': 0,
            'currently_inside': 0,
        })


@login_required
def camera_attendance_dashboard(request):
    """
    Dashboard for camera attendance system with caching.
    """
    from django.core.cache import cache
    
    try:
        # Check if user has permission
        if not request.user.is_staff and not request.user.is_superuser:
            logger.warning(f"Unauthorized dashboard access by {request.user}")
            return render(request, 'camera_attendance/dashboard.html', {
                'total_students': 0,
                'students_with_encoding': 0,
                'today_entries': 0,
                'currently_inside': 0,
                'emotions_today': [],
                'recent_logs': [],
            })
        
        # Try to get cached data
        cache_key = f'camera_dashboard_{request.user.id}'
        context = cache.get(cache_key)
        
        if context is None:
            today = date.today()
            
            total_students = Student.objects.filter(is_active=True).count()
            students_with_encoding = Student.objects.filter(
                is_active=True
            ).exclude(face_encoding__isnull=True).exclude(face_encoding='').count()
            
            today_entries = CameraAttendanceLog.objects.filter(
                date=today,
                entry_time__isnull=False
            ).count()
            
            currently_inside = CameraAttendanceLog.objects.filter(
                date=today,
                entry_time__isnull=False,
                exit_time__isnull=True
            ).count()
            
            # Emotion distribution
            emotions_today = (CameraAttendanceLog.objects
                              .filter(date=today)
                              .values('entry_emotion')
                              .annotate(count=Count('id')))
            
            # Recent logs
            recent_logs = (CameraAttendanceLog.objects
                           .filter(date=today)
                           .select_related('student', 'camera')
                           .order_by('-entry_time')[:10])
            
            context = {
                'total_students': total_students,
                'students_with_encoding': students_with_encoding,
                'today_entries': today_entries,
                'currently_inside': currently_inside,
                'emotions_today': list(emotions_today),
                'recent_logs': recent_logs,
            }
            
            # Cache for 60 seconds
            cache.set(cache_key, context, 60)
            logger.info(f"Camera dashboard data cached for user {request.user}")
        else:
            logger.info(f"Camera dashboard data served from cache for user {request.user}")
        
        return render(request, 'camera_attendance/dashboard.html', context)
    
    except Exception as e:
        logger.error(f"Error loading camera attendance dashboard: {e}", exc_info=True)
        return render(request, 'camera_attendance/dashboard.html', {
            'total_students': 0,
            'students_with_encoding': 0,
            'today_entries': 0,
            'currently_inside': 0,
            'emotions_today': [],
            'recent_logs': [],
        })
