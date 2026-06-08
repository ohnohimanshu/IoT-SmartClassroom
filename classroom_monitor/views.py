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
import time
import cv2
import numpy as np
import requests

from .models import ClassroomCamera, ClassSession, EngagementSnapshot, StudentZoneLog, ClassroomVideo, VideoAnalysisFrame, VideoStudentZone, IncidentReport
from .forms import ClassroomCameraForm, ClassroomVideoForm
from django.core.paginator import Paginator
from django.db.models import Q, Count


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
    """Live camera stream with behaviour detection overlays."""
    camera = get_object_or_404(ClassroomCamera, pk=camera_id)
    return StreamingHttpResponse(
        _generate_video_stream(camera.url, camera_id=camera.pk,
                               camera_location=camera.location,
                               request_obj=request),
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


# ── Behavior Incident Views ───────────────────────────────────────────────────

# Module-level shared detector (loaded once, reused by all streams)
_SHARED_DETECTOR = None


def _prewarm_detector():
    """Load YOLO in a background thread at Django startup so the first stream
    request doesn't block for 3-5 seconds."""
    import threading
    def _load():
        global _SHARED_DETECTOR
        try:
            from classroom_monitor.behavior_detection import ClassroomBehaviorDetector
            _SHARED_DETECTOR = ClassroomBehaviorDetector(
                camera_url='', camera_id=0, server_url='')
            print('[OK] Shared detector pre-warmed')
        except Exception as e:
            print(f'[WARN] Pre-warm failed: {e}')
    t = threading.Thread(target=_load, daemon=True)
    t.start()


# Kick off pre-warm immediately when views.py is imported by Django
try:
    _prewarm_detector()
except Exception:
    pass


def _get_yolo_detector():
    """
    Module-level singleton for the YOLO+Haar detector.
    Loaded once when the module is first imported, shared across all streams.
    This prevents the 3-5s YOLO load blocking the HTTP response.
    """
    global _SHARED_DETECTOR
    try:
        if _SHARED_DETECTOR is None:
            from classroom_monitor.behavior_detection import ClassroomBehaviorDetector
            _SHARED_DETECTOR = ClassroomBehaviorDetector(
                camera_url='',
                camera_id=0,
                server_url='',       # no HTTP self-calls inside detector
            )
    except Exception as e:
        print(f'[WARN] Could not init shared detector: {e}')
    return _SHARED_DETECTOR


def _save_incident_direct(det_type, confidence, snapshot_bgr, student,
                          student_name, roll_no, camera_obj, request_obj=None):
    """
    Save IncidentReport to DB and send one alert email per student per incident
    type with a 5-minute cooldown (per student, not global).
    Called directly — no HTTP self-POST, no Twilio. Uses SMTP email.
    """
    try:
        _, buf = cv2.imencode('.jpg', snapshot_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        from django.core.files.base import ContentFile as CF
        ts        = timezone.now().strftime('%Y%m%d_%H%M%S_%f')
        snap_file = CF(buf.tobytes(), name=f'incident_{ts}.jpg')

        severity = ('medium' if det_type in ('using_phone', 'eating_food') else
                    'low')
        tag   = f'{student_name} ({roll_no})' if student else 'Unknown person'
        from classroom_monitor.behavior_detection import LABEL_MAP
        label = LABEL_MAP.get(det_type, det_type)

        incident = IncidentReport.objects.create(
            student=student,
            camera=None,  # ClassroomCamera is not compatible with entrance_cam.Camera FK
            incident_type=det_type,
            severity=severity,
            confidence=float(confidence),
            snapshot=snap_file,
            description=f'{label} — {tag}',
            whatsapp_sent=False,   # field kept for DB compat, unused now
        )

        # Email only for RED alert poses
        if det_type in ('using_phone', 'eating_food'):
            _send_incident_email(
                incident=incident,
                student=student,
                student_name=student_name,
                roll_no=roll_no,
                det_type=det_type,
                snapshot_bytes=buf.tobytes(),
            )

        return incident
    except Exception as e:
        print(f'[ERROR] _save_incident_direct: {e}')
        return None


def _generate_video_stream(video_path, camera_id=0, camera_location='Classroom',
                            request_obj=None):
    """
    MJPEG stream — crash-free on Windows, no HTTP self-calls.

    Thread layout (dlib/face_recognition ONLY on main thread):
      Main thread    — reads frames, draws boxes, runs face-recog when needed,
                       yields MJPEG bytes
      Detect worker  — YOLO + Haar cascade only (no dlib)
      Incident saver — DB write + WhatsApp (no dlib, receives pre-identified data)

    Why dlib must stay on the main thread on Windows:
      dlib's CNN face detector uses Intel TBB / BLAS internally and crashes
      (0xC0000005 access violation) when called from a non-main Windows thread,
      even with a GIL. We avoid this by keeping all face_recognition calls in
      the main generator loop, throttled to once per FACEREC_INTERVAL seconds.
    """
    import time
    import threading
    import queue as _queue

    from classroom_monitor.behavior_detection import (
        ClassroomBehaviorDetector, COLOR_MAP, LABEL_MAP,
        ALERT_POSES, DISTRACTED_POSES,
    )
    from classroom_monitor.face_recognition_helper import (
        StudentFaceRecognizer, DLIB_LOCK,
    )

    FACEREC_INTERVAL = 5.0   # seconds between face-recognition attempts
    COOLDOWN_S       = 90    # seconds between same incident type alerts
    SNAPSHOT_INTERVAL = 10.0 # seconds between saving engagement snapshots

    # ── Camera object ─────────────────────────────────────────────────────────
    camera_obj = None
    if camera_id:
        try:
            camera_obj = ClassroomCamera.objects.get(pk=camera_id)
        except Exception:
            pass

    # ── Shared detector singleton ─────────────────────────────────────────────
    detector = _get_yolo_detector()
    if detector is None:
        from classroom_monitor.behavior_detection import ClassroomBehaviorDetector
        detector = ClassroomBehaviorDetector(camera_url='', camera_id=0, server_url='')

    # ── Face recognizer — loads DB once, used ONLY on main thread ─────────────
    recognizer = StudentFaceRecognizer()
    recognizer.load_from_db()

    # ── Thread primitives ─────────────────────────────────────────────────────
    result_lock = threading.Lock()
    latest_dets = []                        # list[dict] — latest detections
    detect_q    = _queue.Queue(maxsize=1)   # frames  → detect worker
    save_q      = _queue.Queue(maxsize=50)  # incident dicts → DB/WA saver
    stop_event  = threading.Event()
    cooldown    = {}                        # det_type → last incident timestamp

    # ── Detect worker — YOLO + Haar only, NO dlib ────────────────────────────
    def _detection_worker():
        while not stop_event.is_set():
            try:
                work_frame = detect_q.get(timeout=0.5)
            except _queue.Empty:
                continue
            try:
                dets = detector.detect(work_frame)
                with result_lock:
                    latest_dets.clear()
                    latest_dets.extend(dets)
            except Exception as exc:
                print(f'[DETECT] {exc}')

    # ── Save worker — DB write + email alert, NO dlib ──────────────────────────────
    def _save_worker():
        from django.db import close_old_connections
        while not stop_event.is_set():
            try:
                item = save_q.get(timeout=0.5)
            except _queue.Empty:
                continue
            
            try:
                # Clean up old connections before/after DB operations
                close_old_connections()
                _save_incident_direct(
                    det_type     = item['type'],
                    confidence   = item['confidence'],
                    snapshot_bgr = item['snapshot'],
                    student      = item['student'],
                    student_name = item['name'],
                    roll_no      = item['roll'],
                    camera_obj   = camera_obj,
                    request_obj  = request_obj,
                )
                tag = f"{item['name']} ({item['roll']})" if item['student'] else 'Unknown'
                print(f"[INCIDENT] {LABEL_MAP.get(item['type'], item['type'])} | {tag}")
                close_old_connections()
            except Exception as exc:
                print(f'[SAVE WORKER] {exc}')

    det_thread  = threading.Thread(target=_detection_worker, daemon=True)
    save_thread = threading.Thread(target=_save_worker,      daemon=True)
    det_thread.start()
    save_thread.start()

    # ── Video / camera capture ────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        stop_event.set()
        return

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    target_fps   = min(src_fps, 25.0)
    frame_delay  = 1.0 / target_fps
    detect_every = max(1, int(src_fps))     # YOLO once per source-second
    frame_count  = 0
    last_yield   = time.monotonic()
    last_facerec = 0.0                      # time of last face-rec attempt
    last_snapshot_save = 0.0                # time of last engagement snapshot save

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # ── Feed detect worker ────────────────────────────────────────────
            if frame_count % detect_every == 0:
                try:
                    detect_q.put_nowait(frame.copy())
                except _queue.Full:
                    pass

            # ── Save engagement snapshot periodically ──────────────────────────
            now = time.time()
            if (now - last_snapshot_save) >= SNAPSHOT_INTERVAL:
                last_snapshot_save = now
                with result_lock:
                    snap_dets = list(latest_dets)
                
                # Count behaviors
                focused = 0
                looking_away = 0
                head_down = 0
                using_phone = 0
                eating = 0
                not_visible = 0
                
                for det in snap_dets:
                    dt = det.get('type', 'not_visible')
                    if dt == 'focused':
                        focused += 1
                    elif dt == 'looking_away':
                        looking_away += 1
                    elif dt == 'head_down':
                        head_down += 1
                    elif dt == 'using_phone':
                        using_phone += 1
                    elif dt == 'eating_food':
                        eating += 1
                    else:
                        not_visible += 1
                
                total_detected = focused + looking_away + head_down + using_phone + eating
                engagement_score = (focused / total_detected * 100) if total_detected > 0 else 0.0
                
                # Save engagement snapshot if there's an active session
                try:
                    session = ClassSession.objects.filter(camera_id=camera_id, is_active=True).first()
                    if session:
                        # Encode frame
                        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        frame_file = ContentFile(buf.tobytes(), name=f"frame_{session.pk}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                        
                        # Create snapshot
                        snapshot = EngagementSnapshot.objects.create(
                            session=session,
                            frame_image=frame_file,
                            focused_count=focused,
                            looking_away_count=looking_away,
                            head_down_count=head_down,
                            using_phone_count=using_phone,
                            eating_count=eating,
                            not_visible_count=not_visible,
                            total_detected=total_detected,
                            engagement_score=engagement_score,
                        )
                except Exception as e:
                    print(f'[SNAPSHOT] Error saving engagement snapshot: {e}')

            # ── Face recognition + incident queueing (MAIN THREAD, throttled) ─
            now = time.time()
            if (now - last_facerec) >= FACEREC_INTERVAL:
                last_facerec = now
                with result_lock:
                    snap_dets = list(latest_dets)

                for det in snap_dets:
                    if not (det['is_alert'] or det['is_distracted']):
                        continue
                    key = (det['type'], det.get('track_id'))
                    if (now - cooldown.get(key, 0)) < COOLDOWN_S:
                        continue

                    x1, y1, x2, y2 = det['bbox']
                    mid_y = y1 + int((y2 - y1) * 0.55)
                    crop  = frame[y1:mid_y, x1:x2]
                    if crop.size == 0:
                        crop = frame[y1:y2, x1:x2]

                    # dlib call — main thread only, DLIB_LOCK acquired inside match()
                    sid, name, roll, _ = (recognizer.match(crop)
                                          if crop.size > 0
                                          else (None, 'Unknown', '', 1.0))

                    student = None
                    if sid:
                        try:
                            from entrance_cam.models import Student
                            student = Student.objects.get(pk=sid)
                        except Exception:
                            pass

                    cooldown[key] = now
                    try:
                        # Draw rectangles on snapshot before saving
                        snap_with_rects = frame.copy()
                        for d in snap_dets:
                            x1, y1, x2, y2 = d['bbox']
                            color = d['color']
                            label = d['label']
                            conf = d['confidence']
                            
                            # Draw rectangle
                            cv2.rectangle(snap_with_rects, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw label with background
                            text = f"{label} ({conf:.2f})"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.5
                            thickness = 1
                            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                            
                            # Background for text
                            bg_x1 = x1
                            bg_y1 = max(0, y1 - text_size[1] - 4)
                            bg_x2 = x1 + text_size[0] + 4
                            bg_y2 = y1
                            cv2.rectangle(snap_with_rects, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
                            
                            # Text
                            cv2.putText(snap_with_rects, text, (x1 + 2, y1 - 2), font, 
                                       font_scale, (255, 255, 255), thickness)
                        
                        save_q.put_nowait({
                            'type':       key,
                            'confidence': det['confidence'],
                            'snapshot':   snap_with_rects,
                            'student':    student,
                            'name':       name,
                            'roll':       roll,
                        })
                    except _queue.Full:
                        pass

            # ── Draw annotations ──────────────────────────────────────────────
            annotated  = frame.copy()
            focused = distracted = phone = eating = 0

            with result_lock:
                current_dets = list(latest_dets)

            for det in current_dets:
                dt = det.get('type', 'not_visible')
                if   dt == 'focused':                         focused   += 1
                elif dt in ('looking_away','head_down',
                             'distracted'):                   distracted += 1
                elif dt == 'using_phone':                     phone     += 1
                elif dt == 'eating_food':                     eating    += 1

                x1, y1, x2, y2 = det['bbox']
                color     = det.get('color', COLOR_MAP.get(dt, (120,120,120)))
                label     = det.get('label', LABEL_MAP.get(dt, dt))
                thickness = 2
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, thickness)
                cv2.putText(annotated, label, (x1+4, max(y1-8,18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

            # Summary bar
            total = focused + distracted + phone + eating
            score = (focused / total * 100) if total > 0 else 0.0
            bar   = (f'Focused:{focused}  Distracted:{distracted}'
                     f'  Phone:{phone}  Eating:{eating}  Score:{score:.0f}%')
            bar_w = min(len(bar)*9+14, annotated.shape[1])
            cv2.rectangle(annotated, (0,0), (bar_w,26), (20,20,20), -1)
            cv2.putText(annotated, bar, (6,18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)

            # Colour legend — top right
            fh, fw = annotated.shape[:2]
            for li, (ltxt, lclr) in enumerate([
                ('Focused',    (0,200,60)),
                ('Distracted', (0,165,255)),
                ('Alert',      (0,0,220)),
            ]):
                lx = fw-130; ly = 12+li*20
                cv2.rectangle(annotated, (lx,ly-10), (lx+14,ly+4), lclr, -1)
                cv2.putText(annotated, ltxt, (lx+18,ly+3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, lclr, 1)

            _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')

            elapsed = time.monotonic() - last_yield
            wait    = frame_delay - elapsed
            if wait > 0:
                time.sleep(wait)
            last_yield = time.monotonic()

    finally:
        stop_event.set()
        cap.release()
        det_thread.join(timeout=3)
        save_thread.join(timeout=5)


def _get_port():
    """Return the Django dev-server port (default 8000). Override via DJANGO_PORT env var."""
    return int(os.environ.get('DJANGO_PORT', 8000))


# ── Behavior Incident Views ───────────────────────────────────────────────────

@login_required
def incidents_dashboard(request):
    """View all behavior incidents with filtering."""
    incidents = IncidentReport.objects.select_related('student', 'camera').order_by('-detected_at')
    
    # Filters
    incident_type = request.GET.get('type')
    reviewed = request.GET.get('reviewed')
    
    if incident_type:
        incidents = incidents.filter(incident_type=incident_type)
    
    if reviewed is not None:
        incidents = incidents.filter(is_reviewed=bool(int(reviewed)))
    
    # Stats
    phone_count = IncidentReport.objects.filter(incident_type='using_phone').count()
    eating_count = IncidentReport.objects.filter(incident_type='eating_food').count()
    distracted_count = IncidentReport.objects.filter(incident_type='distracted').count()
    
    # Pagination
    paginator = Paginator(incidents, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'incidents': page_obj,
        'page_obj': page_obj,
        'is_paginated': page_obj.has_other_pages(),
        'phone_count': phone_count,
        'eating_count': eating_count,
        'distracted_count': distracted_count,
    }
    return render(request, 'classroom_monitor/incidents_dashboard.html', context)


@login_required
def incident_detail(request, incident_id):
    """View incident details."""
    incident = get_object_or_404(IncidentReport, pk=incident_id)
    context = {'incident': incident}
    return render(request, 'classroom_monitor/incident_detail.html', context)


@login_required
def update_incident(request, incident_id):
    """Update incident notes and review status."""
    incident = get_object_or_404(IncidentReport, pk=incident_id)
    
    if request.method == 'POST':
        incident.admin_notes = request.POST.get('admin_notes', '')
        incident.is_reviewed = 'is_reviewed' in request.POST
        incident.save()
        messages.success(request, 'Incident updated successfully.')
        return redirect('incident_detail', incident_id=incident.id)
    
    return redirect('incident_detail', incident_id=incident.id)


@login_required
def mark_reviewed(request, incident_id):
    """Mark incident as reviewed."""
    incident = get_object_or_404(IncidentReport, pk=incident_id)
    
    if request.method == 'POST':
        incident.is_reviewed = True
        incident.save()
        messages.success(request, 'Incident marked as reviewed.')
    
    return redirect('incident_detail', incident_id=incident.id)


@csrf_exempt
def api_incidents_report(request):
    """
    API endpoint called by the detection script for every incident:
      - RED   (phone/eating/fighting):  saved to DB  +  email alert
      - ORANGE (distracted/looking_away/head_down): saved to DB only
      - Unknown student: snapshot sent to ALERT_EMAIL_TO (admin)

    Expected JSON body:
      student_id    int|null
      camera_id     int
      incident_type str   (e.g. 'using_phone', 'looking_away')
      confidence    float
      snapshot      str   (base64 JPEG)
      student_name  str
      roll_no       str
      description   str
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        data = json.loads(request.body)

        # ── 1. Resolve student ────────────────────────────────────────────────
        student = None
        if data.get('student_id'):
            try:
                from entrance_cam.models import Student
                student = Student.objects.get(id=data['student_id'])
            except Exception:
                pass

        # ── 2. Resolve camera (camera_id 0 means "no camera row") ────────────
        camera = None
        cam_id = data.get('camera_id')
        if cam_id:
            try:
                from camera_attendance.models import Camera
                camera = Camera.objects.get(pk=cam_id)
            except Exception:
                # Try ClassroomCamera as fallback
                try:
                    camera_obj = ClassroomCamera.objects.get(pk=cam_id)
                except Exception:
                    pass

        # ── 3. Decode snapshot ────────────────────────────────────────────────
        snapshot_file = None
        snapshot_bytes = None
        if data.get('snapshot'):
            try:
                snapshot_bytes = base64.b64decode(data['snapshot'])
                ts = timezone.now().strftime('%Y%m%d_%H%M%S_%f')
                snapshot_file = ContentFile(
                    snapshot_bytes,
                    name=f"incident_{ts}.jpg"
                )
            except Exception:
                pass

        # ── 4. Determine severity ─────────────────────────────────────────────
        alert_types = {'using_phone', 'eating_food', 'fighting'}
        inc_type    = data.get('incident_type', 'other')
        
        # Fighting is CRITICAL severity
        if inc_type == 'fighting':
            severity = 'critical'
        elif inc_type in alert_types:
            severity = 'high'
        else:
            severity = 'medium' if inc_type in {'looking_away', 'head_down', 'distracted'} else 'low'

        # ── 5. Save to DB ─────────────────────────────────────────────────────
        incident = IncidentReport.objects.create(
            student=student,
            camera=camera,
            incident_type=inc_type,
            severity=severity,
            confidence=float(data.get('confidence', 0.0)),
            snapshot=snapshot_file,
            description=data.get('description', ''),
            whatsapp_sent=False,
        )

        # ── 6. Email alert for RED incidents (including fighting) ─────────────
        if inc_type in alert_types and snapshot_bytes:
            _send_incident_email(
                incident=incident,
                student=student,
                student_name=data.get('student_name', 'Unknown'),
                roll_no=data.get('roll_no', ''),
                det_type=inc_type,
                snapshot_bytes=snapshot_bytes,
            )

        return JsonResponse({
            'status':      'created',
            'incident_id': incident.id,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=400)


def _load_env_vars():
    """Load .env file into os.environ (fallback when Django doesn't use decouple)."""
    if os.environ.get('SMTP_HOST') or os.environ.get('EMAIL_HOST'):
        return
    for candidate in [
        os.path.join(settings.BASE_DIR, '.env'),
        os.path.join(settings.BASE_DIR, '..', '.env'),
        '.env',
    ]:
        path = os.path.abspath(candidate)
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, _, v = line.partition('=')
                        k = k.strip(); v = v.strip().strip('"').strip("'")
                        os.environ.setdefault(k, v)
            print(f'[ENV] Loaded {path}')
            return


# ── Per-student email cooldown: (student_id_or_"unknown", det_type) → timestamp
# Keeps 5-minute silence per student per incident type.
_EMAIL_COOLDOWN: dict = {}
_EMAIL_COOLDOWN_SECS  = 300   # 5 minutes


def _send_incident_email(incident, student, student_name, roll_no,
                         det_type, snapshot_bytes):
    """
    Send ONE alert email per student per incident-type with a 5-minute cooldown.

    .env / environment variables required:
      SMTP_HOST       — e.g. smtp.gmail.com
      SMTP_PORT       — e.g. 587
      SMTP_USER       — sender address, e.g. school@gmail.com
      SMTP_PASSWORD   — app password or SMTP password
      SMTP_USE_TLS    — True / False  (default True)
      ALERT_EMAIL_TO  — recipient address (teacher / admin)

    Optional (for known-student parent emails):
      The Student model should have a  parent_email  field.
      If present, the email goes to parent AND ALERT_EMAIL_TO.
    """
    import smtplib
    import ssl
    from email.mime.multipart import MIMEMultipart
    from email.mime.text      import MIMEText
    from email.mime.image     import MIMEImage

    _load_env_vars()

    # ── Cooldown check: one email per (student_key, incident_type) per 5 min ──
    student_key = str(student.id) if student else 'unknown'
    cooldown_key = (student_key, det_type)
    now = time.time()
    if (now - _EMAIL_COOLDOWN.get(cooldown_key, 0)) < _EMAIL_COOLDOWN_SECS:
        print(f'[EMAIL] Cooldown active for {student_name} / {det_type} — skipping')
        return False
    _EMAIL_COOLDOWN[cooldown_key] = now

    # ── SMTP credentials from env ─────────────────────────────────────────────
    smtp_host = (os.environ.get('SMTP_HOST') or
                 os.environ.get('EMAIL_HOST', '')).strip()
    smtp_port = int(os.environ.get('SMTP_PORT') or
                    os.environ.get('EMAIL_PORT', 587))
    smtp_user = (os.environ.get('SMTP_USER') or
                 os.environ.get('EMAIL_HOST_USER', '')).strip()
    smtp_pass = (os.environ.get('SMTP_PASSWORD') or
                 os.environ.get('EMAIL_HOST_PASSWORD', '')).strip()
    use_tls   = str(os.environ.get('SMTP_USE_TLS',
                    os.environ.get('EMAIL_USE_TLS', 'True'))).strip().lower() != 'false'
    admin_to  = (os.environ.get('ALERT_EMAIL_TO') or
                 os.environ.get('ADMIN_EMAIL', '')).strip()

    if not smtp_host or not smtp_user or not smtp_pass:
        print('[EMAIL] SMTP credentials missing in .env — '
              'set SMTP_HOST, SMTP_USER, SMTP_PASSWORD')
        return False

    if not admin_to:
        print('[EMAIL] ALERT_EMAIL_TO not set in .env')
        return False

    # ── Build recipient list ──────────────────────────────────────────────────
    from classroom_monitor.behavior_detection import LABEL_MAP
    label = LABEL_MAP.get(det_type, det_type.replace('_', ' ').title())

    recipients = [admin_to]
    parent_email = ''
    if student:
        parent_email = (getattr(student, 'parent_email', '')
                        or getattr(student, 'guardian_email', '') or '').strip()
        if parent_email and parent_email not in recipients:
            recipients.append(parent_email)

    # ── Build email ───────────────────────────────────────────────────────────
    cam_name  = incident.camera.name if incident.camera else 'Unknown'
    timestamp = timezone.now().strftime('%d %b %Y %H:%M:%S')

    if student:
        subject = f'⚠️ Classroom Alert: {label} — {student_name} ({roll_no})'
        body_html = f"""
<html><body style="font-family:Arial,sans-serif;max-width:600px;margin:auto">
  <div style="background:#d32f2f;color:white;padding:16px;border-radius:6px 6px 0 0">
    <h2 style="margin:0">⚠️ Classroom Behaviour Alert</h2>
  </div>
  <div style="border:1px solid #ddd;border-top:none;padding:20px;border-radius:0 0 6px 6px">
    <table style="width:100%;border-collapse:collapse">
      <tr><td style="padding:8px;color:#555;width:130px"><b>Student</b></td>
          <td style="padding:8px">{student_name}</td></tr>
      <tr style="background:#f9f9f9">
          <td style="padding:8px;color:#555"><b>Roll No.</b></td>
          <td style="padding:8px">{roll_no}</td></tr>
      <tr><td style="padding:8px;color:#555"><b>Incident</b></td>
          <td style="padding:8px;color:#d32f2f"><b>{label}</b></td></tr>
      <tr style="background:#f9f9f9">
          <td style="padding:8px;color:#555"><b>Time</b></td>
          <td style="padding:8px">{timestamp}</td></tr>
      <tr><td style="padding:8px;color:#555"><b>Camera</b></td>
          <td style="padding:8px">{cam_name}</td></tr>
    </table>
    <p style="margin-top:16px;color:#555">Snapshot attached below.</p>
  </div>
</body></html>"""
    else:
        subject = f'⚠️ Classroom Alert: {label} — Unknown Person'
        body_html = f"""
<html><body style="font-family:Arial,sans-serif;max-width:600px;margin:auto">
  <div style="background:#e65100;color:white;padding:16px;border-radius:6px 6px 0 0">
    <h2 style="margin:0">⚠️ Classroom Behaviour Alert — Unknown Person</h2>
  </div>
  <div style="border:1px solid #ddd;border-top:none;padding:20px;border-radius:0 0 6px 6px">
    <table style="width:100%;border-collapse:collapse">
      <tr><td style="padding:8px;color:#555;width:130px"><b>Incident</b></td>
          <td style="padding:8px;color:#e65100"><b>{label}</b></td></tr>
      <tr style="background:#f9f9f9">
          <td style="padding:8px;color:#555"><b>Time</b></td>
          <td style="padding:8px">{timestamp}</td></tr>
      <tr><td style="padding:8px;color:#555"><b>Camera</b></td>
          <td style="padding:8px">{cam_name}</td></tr>
    </table>
    <p style="margin-top:16px;color:#555">Snapshot attached. Face not recognised.</p>
  </div>
</body></html>"""

    # ── Assemble MIME ─────────────────────────────────────────────────────────
    msg             = MIMEMultipart('related')
    msg['Subject']  = subject
    msg['From']     = smtp_user
    msg['To']       = ', '.join(recipients)

    msg.attach(MIMEText(body_html, 'html'))

    if snapshot_bytes:
        img_part = MIMEImage(snapshot_bytes, _subtype='jpeg')
        img_part.add_header('Content-Disposition', 'attachment',
                            filename=f'incident_{timezone.now().strftime("%Y%m%d_%H%M%S")}.jpg')
        msg.attach(img_part)

    # ── Send ──────────────────────────────────────────────────────────────────
    try:
        context = ssl.create_default_context()
        if use_tls:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as srv:
                srv.ehlo()
                srv.starttls(context=context)
                srv.login(smtp_user, smtp_pass)
                srv.sendmail(smtp_user, recipients, msg.as_string())
        else:
            with smtplib.SMTP_SSL(smtp_host, smtp_port,
                                  context=context, timeout=15) as srv:
                srv.login(smtp_user, smtp_pass)
                srv.sendmail(smtp_user, recipients, msg.as_string())

        print(f'[EMAIL] Sent "{subject}" → {recipients}')
        return True

    except Exception as e:
        print(f'[EMAIL] Send failed: {e}')
        return False