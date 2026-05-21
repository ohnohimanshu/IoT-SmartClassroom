from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, StreamingHttpResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from datetime import datetime, timedelta
import json
import base64

from .models import ClassroomCamera, ClassSession, EngagementSnapshot, StudentZoneLog
from .forms import ClassroomCameraForm


@login_required
def dashboard(request):
    active_sessions = ClassSession.objects.filter(is_active=True).select_related('camera')
    cameras = ClassroomCamera.objects.filter(is_active=True)
    context = {
        'active_sessions': active_sessions,
        'cameras': cameras,
    }
    return render(request, 'classroom_monitor/dashboard.html', context)


@login_required
def session_detail(request, session_id):
    session = get_object_or_404(ClassSession, pk=session_id)
    snapshots = EngagementSnapshot.objects.filter(session=session).order_by('-timestamp')[:20]
    latest_snapshot = snapshots.first()
    context = {
        'session': session,
        'snapshots': snapshots,
        'latest_snapshot': latest_snapshot,
    }
    return render(request, 'classroom_monitor/session_detail.html', context)


@login_required
def session_list(request):
    sessions = ClassSession.objects.select_related('camera').order_by('-start_time')
    context = {
        'sessions': sessions,
    }
    return render(request, 'classroom_monitor/session_list.html', context)


@login_required
def camera_list(request):
    cameras = ClassroomCamera.objects.all()
    context = {
        'cameras': cameras,
    }
    return render(request, 'classroom_monitor/camera_list.html', context)


@login_required
def camera_add(request):
    if request.method == 'POST':
        form = ClassroomCameraForm(request.POST)
        if form.is_valid():
            cam = form.save()
            messages.success(request, f'Camera "{cam.name}" added.')
            return redirect('classroom_camera_list')
    else:
        form = ClassroomCameraForm()
    return render(request, 'classroom_monitor/camera_form.html', {'form': form, 'action': 'Add Camera'})


@login_required
def camera_edit(request, pk):
    camera = get_object_or_404(ClassroomCamera, pk=pk)
    if request.method == 'POST':
        form = ClassroomCameraForm(request.POST, instance=camera)
        if form.is_valid():
            form.save()
            messages.success(request, 'Camera updated.')
            return redirect('classroom_camera_list')
    else:
        form = ClassroomCameraForm(instance=camera)
    return render(request, 'classroom_monitor/camera_form.html', {'form': form, 'action': 'Edit Camera', 'camera': camera})


@login_required
def camera_delete(request, pk):
    camera = get_object_or_404(ClassroomCamera, pk=pk)
    if request.method == 'POST':
        camera.delete()
        messages.success(request, 'Camera deleted.')
        return redirect('classroom_camera_list')
    return render(request, 'classroom_monitor/confirm_delete.html', {'obj': camera, 'type': 'Camera'})


@login_required
def camera_test(request, pk):
    import urllib.request
    camera = get_object_or_404(ClassroomCamera, pk=pk)
    try:
        req = urllib.request.urlopen(camera.url, timeout=3)
        return JsonResponse({'status': 'online', 'code': req.getcode()})
    except Exception as e:
        return JsonResponse({'status': 'offline', 'error': str(e)})


@login_required
def session_start(request):
    try:
        from entrance_cam.models import Student
        student = request.user.student_profile
        return JsonResponse({'error': 'Unauthorized'}, status=403)
    except Exception:
        pass
    
    if not request.user.is_staff and not request.user.is_superuser:
        return JsonResponse({'error': 'Unauthorized'}, status=403)
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            camera_id = data.get('camera_id')
            subject = data.get('subject', '')
            teacher = data.get('teacher', '')
            camera = get_object_or_404(ClassroomCamera, pk=camera_id)
            session = ClassSession.objects.create(
                camera=camera,
                subject=subject,
                teacher=teacher
            )
            return JsonResponse({'status': 'created', 'session_id': session.pk})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'POST only'}, status=405)


@login_required
def session_end(request):
    try:
        from entrance_cam.models import Student
        student = request.user.student_profile
        return JsonResponse({'error': 'Unauthorized'}, status=403)
    except Exception:
        pass
    
    if not request.user.is_staff and not request.user.is_superuser:
        return JsonResponse({'error': 'Unauthorized'}, status=403)
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            session_id = data.get('session_id')
            session = get_object_or_404(ClassSession, pk=session_id)
            session.is_active = False
            session.end_time = timezone.now()
            session.save()
            return JsonResponse({'status': 'ended'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'POST only'}, status=405)


@csrf_exempt
def api_snapshot(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    try:
        data = json.loads(request.body)
        camera_id = data.get('camera_id')
        session_id = data.get('session_id')
        session = get_object_or_404(ClassSession, pk=session_id)
        
        frame_snapshot_b64 = data.get('frame_snapshot_b64', None)
        frame_file = None
        
        if frame_snapshot_b64:
            try:
                data_img = base64.b64decode(frame_snapshot_b64)
                frame_file = ContentFile(data_img, name=f"frame_{session.pk}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            except Exception:
                pass
        
        students_data = data.get('students', [])
        
        focused_count = 0
        looking_away_count = 0
        head_down_count = 0
        not_visible_count = 0
        talking_count = 0
        total_detected = 0
        
        snapshot = EngagementSnapshot.objects.create(
            session=session,
            frame_image=frame_file
        )
        
        for student in students_data:
            zone_id = student.get('zone_id')
            pose = student.get('pose', 'not_visible')
            possibly_talking = student.get('possibly_talking', False)
            confidence = student.get('confidence', 0.0)
            
            StudentZoneLog.objects.create(
                snapshot=snapshot,
                zone_id=zone_id,
                pose=pose,
                possibly_talking=possibly_talking,
                confidence=confidence
            )
            
            if pose == 'focused':
                focused_count += 1
            elif pose == 'looking_away':
                looking_away_count += 1
            elif pose == 'head_down':
                head_down_count += 1
            else:
                not_visible_count += 1
            
            if possibly_talking:
                talking_count += 1
            
            if pose != 'not_visible':
                total_detected += 1
        
        engagement_score = 0.0
        if total_detected > 0:
            engagement_score = (focused_count / total_detected) * 100
        
        snapshot.focused_count = focused_count
        snapshot.looking_away_count = looking_away_count
        snapshot.head_down_count = head_down_count
        snapshot.not_visible_count = not_visible_count
        snapshot.talking_count = talking_count
        snapshot.total_detected = total_detected
        snapshot.engagement_score = engagement_score
        snapshot.save()
        
        return JsonResponse({'status': 'saved', 'snapshot_id': snapshot.pk})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@csrf_exempt
def api_active_session(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'GET only'}, status=405)
    camera_id = request.GET.get('camera_id')
    if not camera_id:
        return JsonResponse({'error': 'camera_id required'}, status=400)
    try:
        session = ClassSession.objects.filter(camera_id=camera_id, is_active=True).first()
        if session:
            return JsonResponse({'session_id': session.pk, 'is_active': True})
        return JsonResponse({'session_id': None, 'is_active': False})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@csrf_exempt
def api_stats(request, session_id):
    session = get_object_or_404(ClassSession, pk=session_id)
    snapshots = EngagementSnapshot.objects.filter(session=session).order_by('-timestamp')[:20]
    stats = []
    for snap in reversed(snapshots):
        stats.append({
            'timestamp': snap.timestamp.strftime('%H:%M:%S'),
            'engagement_score': snap.engagement_score,
            'focused_count': snap.focused_count,
            'looking_away_count': snap.looking_away_count,
            'head_down_count': snap.head_down_count,
            'talking_count': snap.talking_count
        })
    
    zone_logs = []
    latest_snap = snapshots.first()
    if latest_snap:
        zone_logs = list(latest_snap.zone_logs.values('zone_id', 'pose', 'possibly_talking'))
    
    return JsonResponse({
        'stats': stats,
        'zone_logs': zone_logs
    })


def generate_frames(camera_url):
    import cv2
    cap = cv2.VideoCapture(camera_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()


@login_required
def live_stream(request, camera_id):
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Unauthorized'}, status=403)
    camera = get_object_or_404(ClassroomCamera, pk=camera_id)
    return StreamingHttpResponse(
        generate_frames(camera.url),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


@login_required
def live_monitor(request):
    try:
        from entrance_cam.models import Student
        student = request.user.student_profile
        return redirect('student_dashboard')
    except Exception:
        pass
    
    if not request.user.is_staff and not request.user.is_superuser:
        try:
            return render(request, 'lab_monitor/error.html', {
                'message': 'You must be an admin to access this page.',
                'current_user': request.user.username,
                'is_staff': request.user.is_staff,
                'is_superuser': request.user.is_superuser
            })
        except:
            return redirect('login')
    
    cameras = ClassroomCamera.objects.filter(is_active=True)
    active_sessions = ClassSession.objects.filter(is_active=True).select_related('camera')
    context = {
        'cameras': cameras,
        'active_sessions': active_sessions,
    }
    return render(request, 'classroom_monitor/live_monitor.html', context)


@login_required
def live_camera_detail(request, camera_id):
    try:
        from entrance_cam.models import Student
        student = request.user.student_profile
        return redirect('student_dashboard')
    except Exception:
        pass
    
    if not request.user.is_staff and not request.user.is_superuser:
        try:
            return render(request, 'lab_monitor/error.html', {
                'message': 'You must be an admin to access this page.',
                'current_user': request.user.username,
                'is_staff': request.user.is_staff,
                'is_superuser': request.user.is_superuser
            })
        except:
            return redirect('login')
    
    camera = get_object_or_404(ClassroomCamera, pk=camera_id)
    session = ClassSession.objects.filter(camera=camera, is_active=True).first()
    context = {
        'camera': camera,
        'session': session,
    }
    return render(request, 'classroom_monitor/live_camera_detail.html', context)
