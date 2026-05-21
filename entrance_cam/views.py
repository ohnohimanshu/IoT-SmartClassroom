from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.http import JsonResponse
from django.db.models import Count, Q
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from datetime import date, timedelta
import json
import base64

from .models import Student, Camera, AttendanceLog
from .forms import StudentForm, CameraForm


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
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            try:
                user.student_profile  # noqa
                return redirect('student_dashboard')
            except Student.DoesNotExist:
                pass
            return redirect('dashboard')
        messages.error(request, 'Invalid credentials.')
    return render(request, 'entrance_cam/login.html')


def logout_view(request):
    logout(request)
    return redirect('login')


@login_required
def dashboard(request):
    try:
        request.user.student_profile  # noqa
        return redirect('student_dashboard')
    except Student.DoesNotExist:
        pass

    today = date.today()
    total_students = Student.objects.filter(is_active=True).count()
    total_cameras = Camera.objects.filter(is_active=True).count()
    today_attendance = AttendanceLog.objects.filter(date=today).count()
    currently_inside = AttendanceLog.objects.filter(
        date=today, entry_time__isnull=False, exit_time__isnull=True
    ).count()

    emotions_today = AttendanceLog.objects.filter(date=today).values('entry_emotion').annotate(count=Count('id'))

    week_data = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        count = AttendanceLog.objects.filter(date=d).count()
        week_data.append({'date': d.strftime('%d %b'), 'count': count})

    recent_logs = AttendanceLog.objects.select_related('student', 'camera').order_by('-entry_time')[:10]

    context = {
        'total_students': total_students,
        'total_cameras': total_cameras,
        'today_attendance': today_attendance,
        'currently_inside': currently_inside,
        'emotions_today': list(emotions_today),
        'week_data': json.dumps(week_data),
        'recent_logs': recent_logs,
        'today': today,
    }
    return render(request, 'entrance_cam/dashboard.html', context)


# ── Students ──────────────────────────────────────────────────────────────────
@login_required
def student_list(request):
    q = request.GET.get('q', '')
    students = Student.objects.all()
    if q:
        students = students.filter(
            Q(name__icontains=q) | Q(roll_no__icontains=q) | Q(email__icontains=q)
        )
    return render(request, 'entrance_cam/student_list.html', {'students': students, 'q': q})


@login_required
def student_add(request):
    if request.method == 'POST':
        form = StudentForm(request.POST, request.FILES)
        if form.is_valid():
            student = form.save()
            messages.success(request, f'Student {student.name} added successfully.')
            return redirect('student_list')
    else:
        form = StudentForm()
    return render(request, 'entrance_cam/student_form.html', {'form': form, 'action': 'Add Student'})


@login_required
def student_edit(request, pk):
    student = get_object_or_404(Student, pk=pk)
    if request.method == 'POST':
        form = StudentForm(request.POST, request.FILES, instance=student)
        if form.is_valid():
            form.save()
            messages.success(request, 'Student updated.')
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
    return render(request, 'entrance_cam/confirm_delete.html', {'obj': student, 'type': 'Student'})


@login_required
def student_detail(request, pk):
    student = get_object_or_404(Student, pk=pk)
    logs = AttendanceLog.objects.filter(student=student).order_by('-date')[:30]
    return render(request, 'entrance_cam/student_detail.html', {'student': student, 'logs': logs})


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
    return render(request, 'entrance_cam/camera_form.html', {'form': form, 'action': 'Add Camera'})


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
def camera_delete(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    if request.method == 'POST':
        camera.delete()
        messages.success(request, 'Camera deleted.')
        return redirect('camera_list')
    return render(request, 'entrance_cam/confirm_delete.html', {'obj': camera, 'type': 'Camera'})


@login_required
def camera_test(request, pk):
    """Quick ping to check if camera URL is reachable."""
    import urllib.request
    camera = get_object_or_404(Camera, pk=pk)
    try:
        req = urllib.request.urlopen(camera.url, timeout=3)
        return JsonResponse({'status': 'online', 'code': req.getcode()})
    except Exception as e:
        return JsonResponse({'status': 'offline', 'error': str(e)})


# ── Attendance ────────────────────────────────────────────────────────────────
@login_required
def attendance_list(request):
    selected_date = request.GET.get('date', str(date.today()))
    try:
        filter_date = date.fromisoformat(selected_date)
    except ValueError:
        filter_date = date.today()

    logs = (AttendanceLog.objects
            .filter(date=filter_date)
            .select_related('student', 'camera')
            .order_by('-entry_time'))
    return render(request, 'entrance_cam/attendance_list.html', {
        'logs': logs,
        'filter_date': filter_date,
        'selected_date': selected_date,
    })


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_snapshot(b64_str, roll_no, suffix):
    """Decode a base64 snapshot string into a ContentFile, or return None."""
    if not b64_str:
        return None
    try:
        data_img = base64.b64decode(b64_str)
        filename = f"snapshot_{roll_no}_{suffix}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        return ContentFile(data_img, name=filename)
    except Exception:
        return None


# Emotions considered "positive" for the mood-comparison field
_POSITIVE_EMOTIONS = {'happy', 'surprise'}
_NEGATIVE_EMOTIONS = {'sad', 'angry', 'fear', 'disgust'}


def _mood_comparison(entry_emotion, exit_emotion):
    """
    Returns a simple label comparing entry vs exit mood.
    Values: 'improved', 'declined', 'stable', 'unknown'
    """
    e_in = (entry_emotion or '').lower()
    e_out = (exit_emotion or '').lower()

    if not e_in or not e_out or e_in == 'unknown' or e_out == 'unknown':
        return 'unknown'
    if e_in == e_out:
        return 'stable'

    in_pos = e_in in _POSITIVE_EMOTIONS
    in_neg = e_in in _NEGATIVE_EMOTIONS
    out_pos = e_out in _POSITIVE_EMOTIONS
    out_neg = e_out in _NEGATIVE_EMOTIONS

    if in_neg and out_pos:
        return 'improved'
    if in_pos and out_neg:
        return 'declined'
    if in_neg and out_neg:
        return 'stable'
    if in_pos and out_pos:
        return 'stable'
    return 'stable'


# ── API: Camera feed trigger (called by the detection script) ─────────────────
@csrf_exempt
def api_log_entry(request):
    """
    POST { student_id, camera_id, emotion, score, snapshot }

    Logic:
    - Find the *most recent* log for this student today that has an entry but
      NO exit yet  →  this is an exit event.
    - If no such log exists  →  create a new entry record.

    This correctly handles multiple in/out visits per day.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        data = json.loads(request.body)
        student = Student.objects.get(pk=data['student_id'])
        camera  = Camera.objects.get(pk=data['camera_id'])
        today   = date.today()
        emotion = data.get('emotion', 'unknown')
        score   = float(data.get('score', 0.0))
        snapshot_b64 = data.get('snapshot')

        # Is there an open (entry logged, no exit) record for today?
        open_log = (AttendanceLog.objects
                    .filter(student=student, date=today,
                            entry_time__isnull=False, exit_time__isnull=True)
                    .order_by('-entry_time')
                    .first())

        if open_log:
            # ── EXIT ──────────────────────────────────────────────────────────
            open_log.exit_time         = timezone.now()
            open_log.exit_emotion      = emotion
            open_log.exit_emotion_score = score

            snapshot_file = _decode_snapshot(snapshot_b64, student.roll_no, 'exit')
            if snapshot_file:
                open_log.exit_snapshot = snapshot_file

            # Mood comparison
            open_log.mood_comparison = _mood_comparison(open_log.entry_emotion, emotion)

            # Duration
            if open_log.entry_time:
                delta = open_log.exit_time - open_log.entry_time
                open_log.duration_minutes = int(delta.total_seconds() // 60)

            open_log.save()
            return JsonResponse({
                'status': 'exit_logged',
                'duration': open_log.duration_minutes,
                'mood_comparison': open_log.mood_comparison,
            })

        else:
            # ── ENTRY ─────────────────────────────────────────────────────────
            log = AttendanceLog(
                student=student,
                camera=camera,
                date=today,
                entry_time=timezone.now(),
                entry_emotion=emotion,
                entry_emotion_score=score,
            )
            snapshot_file = _decode_snapshot(snapshot_b64, student.roll_no, 'entry')
            if snapshot_file:
                log.entry_snapshot = snapshot_file
            log.save()
            return JsonResponse({'status': 'entry_logged'})

    except Student.DoesNotExist:
        return JsonResponse({'error': 'Student not found'}, status=404)
    except Camera.DoesNotExist:
        return JsonResponse({'error': 'Camera not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


# ── API: Get student face encodings (called by detection script) ──────────────
@csrf_exempt
def api_students_encodings(request):
    """GET: Return list of students with face encodings for recognition."""
    if request.method != 'GET':
        return JsonResponse({'error': 'GET only'}, status=405)

    students = (Student.objects
                .filter(is_active=True, face_encoding__isnull=False)
                .exclude(face_encoding=''))
    data = [{'id': s.id, 'name': s.name, 'encoding': s.face_encoding} for s in students]
    return JsonResponse(data, safe=False)