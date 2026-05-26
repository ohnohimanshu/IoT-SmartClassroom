from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, StreamingHttpResponse
from django.utils import timezone
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from datetime import datetime, timedelta
import json
import base64
import os
import cv2
import numpy as np

from .models import ClassroomCamera, ClassSession, EngagementSnapshot, StudentZoneLog, ClassroomVideo, VideoAnalysisFrame, VideoStudentZone
from .forms import ClassroomCameraForm, ClassroomVideoForm


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
        using_phone_count = 0
        eating_count = 0
        fighting_count = 0
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
            elif pose == 'using_phone':
                using_phone_count += 1
            elif pose == 'eating':
                eating_count += 1
            elif pose == 'fighting':
                fighting_count += 1
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
        snapshot.using_phone_count = using_phone_count
        snapshot.eating_count = eating_count
        snapshot.fighting_count = fighting_count
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
            'using_phone_count': getattr(snap, 'using_phone_count', 0),
            'eating_count': getattr(snap, 'eating_count', 0),
            'fighting_count': getattr(snap, 'fighting_count', 0),
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


@login_required
def video_upload(request):
    """Upload classroom video for engagement analysis."""
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
    
    if request.method == 'POST':
        form = ClassroomVideoForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save(commit=False)
            video.status = 'processing'
            video.save()
            # Kick off analysis in a background thread so the response returns immediately
            import threading
            t = threading.Thread(target=analyze_video, args=(video.pk,), daemon=True)
            t.start()
            messages.success(request, f'Video "{video.title}" uploaded. Analysis running in background.')
            return redirect('classroom_video_list')
    else:
        form = ClassroomVideoForm()
    
    return render(request, 'classroom_monitor/video_upload.html', {'form': form})


@login_required
def video_list(request):
    """List all uploaded videos with their analysis status."""
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
    
    videos = ClassroomVideo.objects.all().order_by('-uploaded_at')
    return render(request, 'classroom_monitor/video_list.html', {'videos': videos})


@login_required
def video_detail(request, pk):
    """View detailed analysis of a video."""
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
    
    video = get_object_or_404(ClassroomVideo, pk=pk)
    frames = VideoAnalysisFrame.objects.filter(video=video).order_by('frame_number')[:50]  # Last 50 frames
    return render(request, 'classroom_monitor/video_detail.html', {
        'video': video,
        'frames': frames,
    })


@login_required
def video_delete(request, pk):
    """Delete a video and its analysis."""
    try:
        from entrance_cam.models import Student
        student = request.user.student_profile
        return redirect('student_dashboard')
    except Exception:
        pass
    
    if not request.user.is_staff and not request.user.is_superuser:
        return JsonResponse({'error': 'Unauthorized'}, status=403)
    
    video = get_object_or_404(ClassroomVideo, pk=pk)
    if request.method == 'POST':
        video.delete()
        messages.success(request, 'Video deleted successfully.')
        return redirect('classroom_video_list')
    return render(request, 'classroom_monitor/confirm_delete_video.html', {'obj': video, 'type': 'Video'})


@csrf_exempt
def api_video_analysis(request, pk):
    """API endpoint to get video analysis status and results."""
    video = get_object_or_404(ClassroomVideo, pk=pk)

    if request.method == 'GET':
        frames = VideoAnalysisFrame.objects.filter(video=video).order_by('frame_number')
        frames_data = []
        for frame in frames:
            frames_data.append({
                'frame_number': frame.frame_number,
                'timestamp': frame.timestamp,
                'engagement_score': frame.engagement_score,
                'focused_count': frame.focused_count,
                'looking_away_count': frame.looking_away_count,
                'head_down_count': frame.head_down_count,
                'total_detected': frame.total_detected,
                'image_url': frame.frame_image.url if frame.frame_image else None,
            })

        return JsonResponse({
            'status': video.status,
            'title': video.title,
            'duration_seconds': video.duration_seconds,
            'total_frames_analyzed': video.total_frames_analyzed,
            'average_engagement_score': video.average_engagement_score,
            'frames': frames_data,
        })

    return JsonResponse({'error': 'GET only'}, status=405)


def _generate_video_stream(video_path):
    """
    Generator: reads every frame of the video, runs YOLO+MediaPipe engagement
    detection, draws green (focused) / red (distracted) rectangles, and yields
    MJPEG chunks. Loaded lazily so the import cost is paid once per stream.
    """
    import sys
    script_dir = os.path.join(settings.BASE_DIR, 'classroom_monitor')
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from detection_script import EngagementDetector

    COLOR_MAP = {
        'focused':      (0, 220, 80),
        'looking_away': (0, 180, 255),
        'head_down':    (30, 30, 220),
        'using_phone':  (0, 0, 255),   # red
        'eating':       (255, 165, 0), # blue
        'fighting':     (0, 0, 180),   # dark red
        'not_visible':  (120, 120, 120),
    }
    LABEL_MAP = {
        'focused':      'Focused',
        'looking_away': 'Looking Away',
        'head_down':    'Head Down',
        'using_phone':  'Using Phone',
        'eating':       'Eating',
        'fighting':     'FIGHT DETECTED',
        'not_visible':  'Not Visible',
    }

    detector = EngagementDetector()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # Detect every N frames; display every frame for smooth playback
    detect_every = max(1, int(fps // 4))   # ~4 detections/sec
    frame_count = 0
    last_detections = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run detection periodically, reuse last result between detections
            if frame_count % detect_every == 0:
                last_detections = detector.detect(frame)

            annotated = frame.copy()
            focused = looking_away = head_down = phone = eating = fighting = 0

            for det in last_detections:
                pose = det['pose']
                if pose == 'focused':
                    focused += 1
                elif pose == 'looking_away':
                    looking_away += 1
                elif pose == 'head_down':
                    head_down += 1
                elif pose == 'using_phone':
                    phone += 1
                elif pose == 'eating':
                    eating += 1
                elif pose == 'fighting':
                    fighting += 1

                x1, y1, x2, y2 = det['bbox']
                color = COLOR_MAP.get(pose, (120, 120, 120))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label_y = y1 - 8 if y1 > 20 else y2 + 18
                cv2.putText(annotated, LABEL_MAP.get(pose, pose),
                            (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if det.get('possibly_talking'):
                    cv2.circle(annotated, (x1 + 14, y1 + 14), 7, (0, 165, 255), -1)

            total = focused + looking_away + head_down + phone + eating + fighting
            score = (focused / total * 100) if total > 0 else 0.0
            bar = f"Focused:{focused}  Away:{looking_away}  Down:{head_down}  Phone:{phone}  Eating:{eating}  Fight:{fighting}  Score:{score:.0f}%"
            bar_w = min(len(bar) * 9 + 12, annotated.shape[1])
            cv2.rectangle(annotated, (0, 0), (bar_w, 26), (0, 0, 0), -1)
            cv2.putText(annotated, bar, (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')
    finally:
        cap.release()


@login_required
def video_live_stream(request, pk):
    """Stream the uploaded video with live engagement annotations as MJPEG."""
    video = get_object_or_404(ClassroomVideo, pk=pk)
    if video.status not in ('completed', 'processing'):
        return JsonResponse({'error': 'Video not ready'}, status=400)
    return StreamingHttpResponse(
        _generate_video_stream(video.video_file.path),
        content_type='multipart/x-mixed-replace; boundary=frame',
    )


def analyze_video(video_pk):
    """
    Analyze a classroom video using the same YOLO + MediaPipe FaceLandmarker
    pipeline as the live detection script (classroom_monitor/detection_script.py).
    Each person is detected with YOLOv8, then MediaPipe estimates head pose
    (yaw/pitch) to classify focused / looking_away / head_down accurately.
    """
    import traceback
    import sys

    try:
        video = ClassroomVideo.objects.get(pk=video_pk)
        video.status = 'processing'
        video.save()

        # Import the EngagementDetector from the sibling detection_script
        script_dir = os.path.join(settings.BASE_DIR, 'classroom_monitor')
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from detection_script import EngagementDetector
        detector = EngagementDetector()
        print(f"[OK] EngagementDetector loaded for video {video_pk}")

        video_path = video.video_file.path
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video.duration_seconds = int(total_frames / fps)
        video.save()

        # Analyze one frame every 2 seconds (balance accuracy vs speed)
        frame_interval = max(1, int(fps * 2))
        frame_count = 0
        analyzed_frames = 0
        total_engagement = 0.0
        total_students_sum = 0

        frames_dir = os.path.join(settings.BASE_DIR, 'media', 'classroom', 'video_frames')
        os.makedirs(frames_dir, exist_ok=True)

        # Color map matching detection_script.py draw_bboxes
        COLOR_MAP = {
            'focused':      (0, 220, 80),    # green
            'looking_away': (0, 200, 255),   # yellow
            'head_down':    (0, 60, 220),    # blue-red
            'using_phone':  (0, 0, 255),     # red
            'eating':       (255, 165, 0),   # blue
            'fighting':     (0, 0, 180),     # dark red
            'not_visible':  (120, 120, 120), # grey
        }
        LABEL_MAP = {
            'focused':      'Focused',
            'looking_away': 'Looking Away',
            'head_down':    'Head Down',
            'using_phone':  'Using Phone',
            'eating':       'Eating',
            'fighting':     'FIGHT DETECTED',
            'not_visible':  'Not Visible',
        }

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            timestamp = frame_count / fps
            analyzed_frames += 1

            # Run YOLO + MediaPipe detection
            detections = detector.detect(frame)

            focused_count = 0
            looking_away_count = 0
            head_down_count = 0
            using_phone_count = 0
            eating_count = 0
            fighting_count = 0
            not_visible_count = 0
            zone_data = []
            annotated = frame.copy()

            for det in detections:
                pose = det['pose']
                if pose == 'focused':
                    focused_count += 1
                elif pose == 'looking_away':
                    looking_away_count += 1
                elif pose == 'head_down':
                    head_down_count += 1
                elif pose == 'using_phone':
                    using_phone_count += 1
                elif pose == 'eating':
                    eating_count += 1
                elif pose == 'fighting':
                    fighting_count += 1
                else:
                    not_visible_count += 1

                zone_data.append({
                    'zone_id': det['zone_id'],
                    'pose': pose,
                    'possibly_talking': det.get('possibly_talking', False),
                    'confidence': det.get('confidence', 0.0),
                })

                # Draw annotated bounding box
                x1, y1, x2, y2 = det['bbox']
                color = COLOR_MAP.get(pose, (120, 120, 120))
                label = LABEL_MAP.get(pose, pose)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label_y = y1 - 8 if y1 - 8 > 12 else y2 + 18
                cv2.putText(annotated, label, (x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if det.get('possibly_talking'):
                    cv2.circle(annotated, (x1 + 14, y1 + 14), 7, (0, 165, 255), -1)

            total_detected = focused_count + looking_away_count + head_down_count + using_phone_count + eating_count + fighting_count
            total_students_sum += total_detected
            engagement_score = (focused_count / total_detected * 100) if total_detected > 0 else 0.0
            total_engagement += engagement_score

            # Overlay summary bar
            bar_text = (f"Focused:{focused_count}  Away:{looking_away_count}"
                        f"  Down:{head_down_count}  Phone:{using_phone_count}  Eating:{eating_count}  Fight:{fighting_count}"
                        f"  Score:{engagement_score:.0f}%")
            bar_w = min(len(bar_text) * 9 + 12, annotated.shape[1])
            cv2.rectangle(annotated, (0, 0), (bar_w, 26), (0, 0, 0), -1)
            cv2.putText(annotated, bar_text, (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Save annotated frame
            frame_img_rel = f'classroom/video_frames/frame_{video.pk}_{frame_count}.jpg'
            frame_img_abs = os.path.join(settings.BASE_DIR, 'media', frame_img_rel)
            cv2.imwrite(frame_img_abs, annotated)

            analysis_frame = VideoAnalysisFrame.objects.create(
                video=video,
                frame_number=frame_count,
                timestamp=timestamp,
                frame_image=frame_img_rel,
                focused_count=focused_count,
                looking_away_count=looking_away_count,
                head_down_count=head_down_count,
                using_phone_count=using_phone_count,
                eating_count=eating_count,
                fighting_count=fighting_count,
                not_visible_count=not_visible_count,
                total_detected=total_detected,
                engagement_score=engagement_score,
            )

            for z in zone_data:
                VideoStudentZone.objects.create(
                    frame=analysis_frame,
                    zone_id=z['zone_id'],
                    pose=z['pose'],
                    possibly_talking=z['possibly_talking'],
                    confidence=z['confidence'],
                )

            video.total_frames_analyzed = analyzed_frames
            video.average_engagement_score = total_engagement / analyzed_frames
            video.save(update_fields=['total_frames_analyzed', 'average_engagement_score'])

            print(f"[Video {video_pk}] Frame {frame_count} ({timestamp:.1f}s) — "
                  f"focused:{focused_count} away:{looking_away_count} down:{head_down_count} "
                  f"score:{engagement_score:.0f}%")

        cap.release()

        video.status = 'completed'
        video.total_students_detected = (total_students_sum // analyzed_frames) if analyzed_frames > 0 else 0
        video.processed_at = timezone.now()
        video.save()
        print(f"[OK] Video {video_pk} complete — {analyzed_frames} frames, "
              f"avg engagement {video.average_engagement_score:.1f}%")

    except Exception as e:
        print(f"[ERROR] analyze_video({video_pk}): {e}")
        traceback.print_exc()
        try:
            video = ClassroomVideo.objects.get(pk=video_pk)
            video.status = 'failed'
            video.save()
        except Exception:
            pass
