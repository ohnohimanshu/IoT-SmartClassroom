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
import logging
import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

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
    from classroom_monitor.api_auth import check_detection_api_key
    auth_err = check_detection_api_key(request)
    if auth_err:
        return auth_err
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
    try:
        return StreamingHttpResponse(
            _generate_video_stream(camera.url, camera_id=camera.pk,
                                   camera_location=camera.location,
                                   request_obj=request),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        logger.error(f'live_stream error for camera {camera_id}: {e}', exc_info=True)
        return StreamingHttpResponse(
            _error_frame('Stream error — check server logs'),
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

# Per-stream detector registry (keyed by camera / video job — NOT a single
# global instance). Each ClassroomBehaviorDetector owns its own
# TemporalBehaviorEngine + HeadPoseDetector + PhoneDetector + FoodDetector +
# HandRaiseDetector, all of which key their internal state by ByteTrack
# track_id. Track IDs are assigned independently per tracking session, so two
# different cameras (or a live camera running at the same time as an
# uploaded-video analysis job) WILL eventually produce the same numeric
# track_id for two completely different people. Sharing one detector across
# streams meant one student's keypoint/behavior history, confidence
# smoothing, and incident cooldown could silently get overwritten by an
# unrelated student on a different camera or video. Each key below gets its
# own isolated detector instance instead.
#
# The underlying YOLO model *weights* are still shared process-wide via
# _SharedYOLOModels inside behavior_detection.py, so creating a new
# ClassroomBehaviorDetector per key does NOT reload the model from disk —
# it's cheap after the very first load.
_DETECTOR_REGISTRY: dict = {}
_DETECTOR_REGISTRY_LOCK = None  # set below once threading is imported


def _prewarm_detector():
    """Load YOLO model weights in a background thread at Django startup so
    the first stream request doesn't block for 3-5 seconds. This constructs
    one throwaway detector purely to trigger the shared model load in
    _SharedYOLOModels — it is not kept or reused itself, since real streams
    get their own per-key detector via _get_yolo_detector(key)."""
    import threading
    def _load():
        try:
            from classroom_monitor.behavior_detection import ClassroomBehaviorDetector
            ClassroomBehaviorDetector(camera_url='', camera_id=0, server_url='')
            print('[OK] YOLO models pre-warmed')
        except Exception as e:
            print(f'[WARN] Pre-warm failed: {e}')
    t = threading.Thread(target=_load, daemon=True)
    t.start()


# Kick off pre-warm immediately when views.py is imported by Django
try:
    import threading as _threading_bootstrap
    _DETECTOR_REGISTRY_LOCK = _threading_bootstrap.Lock()
    _prewarm_detector()
except Exception:
    pass


def _get_yolo_detector(key: str = '_default'):
    """
    Per-stream detector registry. `key` should be something stable and
    unique per live camera (e.g. f"camera:{camera_id}") or per uploaded-video
    analysis job (e.g. f"video:{video_pk}") — never shared between two
    different streams/jobs. Lazily creates and caches one
    ClassroomBehaviorDetector per key so tracked-person state never leaks
    across cameras or videos.
    """
    global _DETECTOR_REGISTRY, _DETECTOR_REGISTRY_LOCK
    if _DETECTOR_REGISTRY_LOCK is None:
        import threading as _threading_lazy
        _DETECTOR_REGISTRY_LOCK = _threading_lazy.Lock()
    with _DETECTOR_REGISTRY_LOCK:
        det = _DETECTOR_REGISTRY.get(key)
        if det is None:
            try:
                from classroom_monitor.behavior_detection import ClassroomBehaviorDetector
                det = ClassroomBehaviorDetector(
                    camera_url='',
                    camera_id=0,
                    server_url='',       # no HTTP self-calls inside detector
                )
                _DETECTOR_REGISTRY[key] = det
            except Exception as e:
                print(f'[WARN] Could not init detector for key={key}: {e}')
                det = None
    return det


def _release_yolo_detector(key: str):
    """Remove a per-camera/per-video detector from the registry once its
    stream or job ends, so tracked-person state and memory don't accumulate
    across repeated open/close cycles of the same camera page."""
    global _DETECTOR_REGISTRY, _DETECTOR_REGISTRY_LOCK
    if _DETECTOR_REGISTRY_LOCK is None:
        return
    with _DETECTOR_REGISTRY_LOCK:
        _DETECTOR_REGISTRY.pop(key, None)


def _incident_severity(det_type: str) -> str:
    if det_type in ('using_phone', 'eating_food'):
        return 'high'
    if det_type in ('distracted', 'looking_away', 'head_down'):
        return 'medium'
    return 'low'


def _save_incident_direct(det_type, confidence, snapshot_bgr, student,
                         student_name, roll_no, camera_obj, request_obj=None,
                         description_extra=''):
    """
    Save IncidentReport to DB and send one alert email per student per incident
    type with a 5-minute cooldown (per student, not global).
    Called directly — no HTTP self-POST, no Twilio. Uses SMTP email.
    Returns the IncidentReport on success, None on failure.
    """
    try:
        _, buf = cv2.imencode('.jpg', snapshot_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        from django.core.files.base import ContentFile as CF
        ts        = timezone.now().strftime('%Y%m%d_%H%M%S_%f')
        snap_file = CF(buf.tobytes(), name=f'incident_{ts}.jpg')

        from classroom_monitor.constants import LABEL_MAP, EMAIL_ALERT_TYPES
        severity = _incident_severity(det_type)
        tag   = f'{student_name} ({roll_no})' if student else 'Unknown person'
        label = LABEL_MAP.get(det_type, det_type)

        desc = description_extra or f'{label} — {tag}'

        incident = IncidentReport.objects.create(
            student=student,
            camera=None,  # ClassroomCamera is not compatible with entrance_cam.Camera FK
            incident_type=det_type,
            severity=severity,
            confidence=float(confidence),
            snapshot=snap_file,
            description=desc,
            whatsapp_sent=False,
        )

        if det_type in EMAIL_ALERT_TYPES:
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

def _error_frame(message='Stream unavailable'):
    """Yield a single MJPEG frame containing an error message."""
    try:
        err = np.zeros((240, 480, 3), dtype=np.uint8)
        cv2.putText(err, message[:60], (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2)
        _, buf = cv2.imencode('.jpg', err)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')
    except Exception:
        pass


def _generate_video_stream(video_path, camera_id=0, camera_location='Classroom',
                            request_obj=None):
    """
    MJPEG stream — crash-free on Windows, no HTTP self-calls.
    Fully isolated against dictionary/dataclass structural type mismatches.
    """
    import time
    import threading
    import queue as _queue

    # Guard: if required libraries are missing, stream an error frame instead
    # of raising an unhandled exception (which causes a 500 before any bytes
    # are sent to the browser).
    try:
        from classroom_monitor.behavior_detection import (
            ClassroomBehaviorDetector, COLOR_MAP, LABEL_MAP,
            ALERT_POSES, DISTRACTED_POSES,
        )
        from classroom_monitor.constants import EMAIL_ALERT_TYPES
        from classroom_monitor.face_recognition_helper import (
            StudentFaceRecognizer, DLIB_LOCK,
        )
    except Exception as import_err:
        print(f'[STREAM] Import failed: {import_err}')
        yield from _error_frame(f'Import error: {str(import_err)[:50]}')
        return

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

    # ── Per-camera detector (own tracker/behavior-engine state) ───────────────
    # camera_id defaults to 0 for ad-hoc/URL-only streams; make that a unique
    # key too (keyed by video_path) so two zero-id ad-hoc streams don't share
    # state either.
    stream_key = f'camera:{camera_id}' if camera_id else f'adhoc:{video_path}'
    detector = _get_yolo_detector(stream_key)
    if detector is None:
        from classroom_monitor.behavior_detection import ClassroomBehaviorDetector
        detector = ClassroomBehaviorDetector(camera_url='', camera_id=0, server_url='')

    # ── Face recognizer — loads DB once, used ONLY on main thread ─────────────
    recognizer = StudentFaceRecognizer()
    recognizer.load_from_db()

    # ── Thread primitives ─────────────────────────────────────────────────────
    result_lock = threading.Lock()
    latest_dets = []                        # list[dict] — latest detections
    detect_q    = _queue.Queue(maxsize=1)   # frames  → full detect worker (heavy)
    pose_q      = _queue.Queue(maxsize=2)   # frames  → pose-only worker (cheap)
    save_q      = _queue.Queue(maxsize=50)  # incident dicts → DB/WA saver
    stop_event  = threading.Event()
    cooldown    = {}                        # (type, track_id) → last saved timestamp
    pending_keys = set()

    # `detector.processor.yolo_model.track(..., persist=True)` keeps mutable
    # internal ByteTrack state on the model object. _detection_worker and
    # _pose_worker run as separate threads and both call into that same
    # model — without a lock, two threads can enter .track() at the same
    # time and corrupt tracker state (ID churn, garbage/misaligned
    # keypoints, occasional exceptions). This lock makes every call into the
    # tracker mutually exclusive, regardless of which worker makes it.
    yolo_track_lock = threading.Lock()

    # ── Detect worker — YOLO + Haar only, NO dlib ────────────────────────────
    def _detection_worker():
        while not stop_event.is_set():
            try:
                work_frame = detect_q.get(timeout=0.5)
            except _queue.Empty:
                continue
            try:
                with yolo_track_lock:
                    dets = detector.detect(work_frame)
                with result_lock:
                    latest_dets.clear()
                    latest_dets.extend(dets)
            except Exception as exc:
                print(f'[DETECT] {exc}')

    # ── Pose-only worker — keeps ByteTrack IDs alive between heavy detections ─
    # Calls _parse_pose_detections on every pose_q frame so the tracker sees
    # consistent motion and doesn't churn IDs. Does NOT run object detection
    # or fight detection (cheap path only). It DOES feed each track into
    # behavior_engine.update_person() so keypoint_history actually gets
    # samples at this worker's higher cadence (~8fps) instead of only at the
    # heavy-detection cadence (~2fps) — the wrist-motion-variance ("is this
    # writing or phone-scrolling") heuristic in behavior_detection_core.py
    # needs that higher sample rate to be a meaningful signal. Previously this
    # worker discarded its results entirely, so it had zero effect on
    # tracked-person state.
    def _pose_worker():
        while not stop_event.is_set():
            try:
                work_frame = pose_q.get(timeout=0.5)
            except _queue.Empty:
                continue
            try:
                if detector is not None and detector.processor.yolo_model is not None:
                    with yolo_track_lock:
                        tracks = detector.processor._parse_pose_detections(work_frame)
                    ts = time.time()
                    for tid, x1, y1, x2, y2, conf, kp in tracks:
                        detector.processor.behavior_engine.update_person(
                            tid, (x1, y1, x2, y2), kp, ts)
            except Exception as exc:
                print(f'[POSE] {exc}')

    # ── Save worker — DB write + email alert, NO dlib ──────────────────────────────
    def _save_worker():
        from django.db import close_old_connections
        while not stop_event.is_set():
            try:
                item = save_q.get(timeout=0.5)
            except _queue.Empty:
                continue
            
            try:
                close_old_connections()
                incident = _save_incident_direct(
                    det_type     = item['type'],
                    confidence   = item['confidence'],
                    snapshot_bgr = item['snapshot'],
                    student      = item['student'],
                    student_name = item['name'],
                    roll_no      = item['roll'],
                    camera_obj   = camera_obj,
                    request_obj  = request_obj,
                )
                if incident is not None and item.get('cooldown_key') is not None:
                    cooldown[item['cooldown_key']] = time.time()
                tag = f"{item['name']} ({item['roll']})" if item['student'] else 'Unknown'
                print(f"[INCIDENT] {LABEL_MAP.get(item['type'], item['type'])} | {tag}")
                close_old_connections()
            except Exception as exc:
                print(f'[SAVE WORKER] {exc}')
            finally:
                pending_keys.discard(item.get('cooldown_key'))

    det_thread  = threading.Thread(target=_detection_worker, daemon=True)
    pose_thread = threading.Thread(target=_pose_worker,      daemon=True)
    save_thread = threading.Thread(target=_save_worker,      daemon=True)
    det_thread.start()
    pose_thread.start()
    save_thread.start()

    # ── Video / camera capture ────────────────────────────────────────────────
    cap = None
    reconnect_attempts = 0
    max_reconnect_attempts = 5
    reconnect_delay = 2
    
    def _open_camera():
        try:
            path = int(video_path) if str(video_path).isdigit() else video_path
            cv2_cap = cv2.VideoCapture(path)
            if cv2_cap.isOpened():
                cv2_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cv2_cap
        except Exception as e:
            print(f'[STREAM] Error opening camera: {e}')
        return None
    
    cap = _open_camera()
    if cap is None:
        print(f'[STREAM] Failed to open camera: {video_path}')
        stop_event.set()
        # Yield a placeholder error frame so the browser shows something
        # instead of a broken-image icon.
        try:
            err_frame = np.zeros((240, 480, 3), dtype=np.uint8)
            cv2.putText(err_frame, 'Camera unavailable', (60, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255), 2)
            cv2.putText(err_frame, str(video_path)[:60], (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            _, buf = cv2.imencode('.jpg', err_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')
        except Exception:
            pass
        return

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    target_fps   = min(src_fps, 25.0)
    frame_delay  = 1.0 / target_fps
    # Run pose tracking every 3 frames (~8fps at 25fps source) to keep
    # ByteTrack IDs stable. Heavy object/fight detection runs every ~0.5s.
    # Previously both ran at 1fps (every src_fps frames), which caused
    # track IDs to churn and broke all temporal smoothing logic.
    pose_every   = max(1, int(src_fps // 8))    # ~3 frames at 25fps
    heavy_every  = max(1, int(src_fps // 2))    # ~12 frames at 25fps
    frame_count  = 0
    last_yield   = time.monotonic()
    last_facerec = 0.0
    last_snapshot_save = 0.0
    consecutive_errors = 0
    max_consecutive_errors = 10

    try:
        while True:
            try:
                if cap is None or not cap.isOpened():
                    raise RuntimeError("Camera not connected")
                    
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors >= 5:
                        if cap:
                            cap.release()
                        cap = _open_camera()
                        if cap is None:
                            reconnect_attempts += 1
                            if reconnect_attempts >= max_reconnect_attempts:
                                break
                            time.sleep(reconnect_delay)
                        else:
                            consecutive_errors = 0
                            reconnect_attempts = 0
                    time.sleep(0.1) # Prevent CPU spinning on EOF
                    continue
                else:
                    consecutive_errors = 0
                    reconnect_attempts = 0
                    
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    if cap:
                        cap.release()
                    time.sleep(reconnect_delay)
                    cap = _open_camera()
                    consecutive_errors = 0
                    if cap is None:
                        reconnect_attempts += 1
                        if reconnect_attempts >= max_reconnect_attempts:
                            break
                else:
                    time.sleep(0.1)
                continue
                
            # ── PROTECTED INNER LOOP PROCESSING (Prevents all 500 stream crashes) ──
            try:
                frame_count += 1

                # Two-tier dispatch:
                # pose_q  — every pose_every frames (~8fps): keeps ByteTrack IDs stable
                # detect_q — every heavy_every frames (~2fps): full object+fight pipeline
                if frame_count % pose_every == 0:
                    try:
                        pose_q.put_nowait(frame.copy())
                    except _queue.Full:
                        pass

                if frame_count % heavy_every == 0:
                    try:
                        detect_q.put_nowait(frame.copy())
                    except _queue.Full:
                        pass

                # Thread safe extraction & absolute dictionary normalization
                with result_lock:
                    raw_dets = list(latest_dets)
                
                current_dets = []
                for rd in raw_dets:
                    if isinstance(rd, dict):
                        current_dets.append(rd)
                    else:
                        try:
                            current_dets.append({
                                'type': getattr(rd, 'type', 'focused'),
                                'bbox': getattr(rd, 'bbox', (0,0,0,0)),
                                'confidence': getattr(rd, 'confidence', 0.0),
                                'color': getattr(rd, 'color', (0,200,60)),
                                'label': getattr(rd, 'label', 'Focused'),
                                'is_alert': getattr(rd, 'is_alert', False),
                                'is_distracted': getattr(rd, 'is_distracted', False),
                                'track_id': getattr(rd, 'track_id', None),
                            })
                        except Exception:
                            pass

                # ── Save engagement snapshot periodically ─────────────────────
                now = time.time()
                if (now - last_snapshot_save) >= SNAPSHOT_INTERVAL:
                    last_snapshot_save = now
                    
                    focused = looking_away = head_down = using_phone = eating = hand_raised = not_visible = 0
                    for det in current_dets:
                        dt = det.get('type', 'not_visible')
                        if dt == 'focused':
                            focused += 1
                        elif dt in ('distracted', 'looking_away'):
                            looking_away += 1
                        elif dt == 'head_down':
                            head_down += 1
                        elif dt == 'using_phone':
                            using_phone += 1
                        elif dt == 'eating_food':
                            eating += 1
                        elif dt == 'hand_raised':
                            hand_raised += 1
                        else:
                            not_visible += 1
                    
                    total_detected = focused + looking_away + head_down + using_phone + eating + hand_raised
                    engagement_score = (focused / total_detected * 100) if total_detected > 0 else 0.0
                    
                    def _save_snapshot_bg(f_copy, f_count, l_away, h_down, u_phone, eat, h_raised, n_vis, t_det, e_score):
                        try:
                            from django.db import close_old_connections
                            close_old_connections()
                            session = ClassSession.objects.filter(camera_id=camera_id, is_active=True).first()
                            if session:
                                ret_b, buf_b = cv2.imencode('.jpg', f_copy, [cv2.IMWRITE_JPEG_QUALITY, 80])
                                if ret_b and buf_b is not None:
                                    frame_file = ContentFile(buf_b.tobytes(), name=f"frame_{session.pk}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                                    EngagementSnapshot.objects.create(
                                        session=session,
                                        frame_image=frame_file,
                                        focused_count=f_count,
                                        looking_away_count=l_away,
                                        head_down_count=h_down,
                                        using_phone_count=u_phone,
                                        eating_count=eat,
                                        not_visible_count=n_vis,
                                        total_detected=t_det,
                                        engagement_score=e_score,
                                    )
                            close_old_connections()
                        except Exception as e:
                            print(f'[SNAPSHOT] Error saving engagement snapshot: {e}')
                    
                    snapshot_thread = threading.Thread(
                        target=_save_snapshot_bg, 
                        args=(frame.copy(), focused, looking_away, head_down, using_phone, eating, hand_raised, not_visible, total_detected, engagement_score),
                        daemon=True
                    )
                    snapshot_thread.start()

                # ── Face recognition + incident queueing ──────────────────────
                now = time.time()
                if (now - last_facerec) >= FACEREC_INTERVAL:
                    last_facerec = now
                    facerec_start = time.time()
                    
                    for det in current_dets:
                        if time.time() - facerec_start > 1.0:
                            break
                            
                        if not (det['is_alert'] or det['is_distracted']):
                            continue
                            
                        key = (det['type'], det.get('track_id'))
                        if key in pending_keys:
                            continue
                        if (now - cooldown.get(key, 0)) < COOLDOWN_S:
                            continue

                        x1, y1, x2, y2 = det['bbox']
                        mid_y = y1 + int((y2 - y1) * 0.55)
                        crop  = frame[y1:mid_y, x1:x2]
                        if crop.size == 0:
                            crop = frame[y1:y2, x1:x2]

                        try:
                            sid, name, roll, _ = (recognizer.match(crop) if crop.size > 0 else (None, 'Unknown', '', float('nan')))
                            student = None
                            if sid:
                                try:
                                    from entrance_cam.models import Student
                                    student = Student.objects.get(pk=sid)
                                except Exception:
                                    pass

                            snap_with_rects = frame.copy()
                            for d in current_dets:
                                dx1, dy1, dx2, dy2 = d['bbox']
                                cv2.rectangle(snap_with_rects, (dx1, dy1), (dx2, dy2), d['color'], 2)
                                text = f"{d['label']} ({d['confidence']:.2f})"
                                cv2.putText(snap_with_rects, text, (dx1 + 2, dy1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                            item = {
                                'type':       det['type'],
                                'confidence': det['confidence'],
                                'snapshot':   snap_with_rects,
                                'student':    student,
                                'name':       name,
                                'roll':       roll,
                                'cooldown_key': key,
                            }
                            pending_keys.add(key)
                            try:
                                save_q.put(item, timeout=2.0)
                            except _queue.Full:
                                pending_keys.discard(key)
                                print('[STREAM] Incident save queue full — dropping alert')
                        except Exception as e:
                            print(f'[STREAM] Error in face recognition: {e}')

                # ── Draw annotations ──────────────────────────────────────────
                annotated  = frame.copy()
                focused = distracted = phone = eating = hand_raised = 0

                max_annotations = min(len(current_dets), 20)
                for det in current_dets[:max_annotations]:
                    dt = det.get('type', 'not_visible')
                    if   dt == 'focused': focused   += 1
                    elif dt in ('looking_away','head_down', 'distracted'): distracted += 1
                    elif dt == 'using_phone': phone     += 1
                    elif dt == 'eating_food': eating    += 1
                    elif dt == 'hand_raised': hand_raised +=1

                    x1, y1, x2, y2 = det['bbox']
                    color     = det.get('color', COLOR_MAP.get(dt, (120,120,120)))
                    label     = det.get('label', LABEL_MAP.get(dt, dt))
                    
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(annotated, label, (x1+4, max(y1-8,18)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

                # Summary bar
                total = focused + distracted + phone + eating + hand_raised
                score = (focused / total * 100) if total > 0 else 0.0
                bar   = f'Focused:{focused}  Distracted:{distracted}  Phone:{phone}  Eating:{eating}  Hand Raised:{hand_raised}  Score:{score:.0f}%'
                bar_w = min(len(bar)*9+14, annotated.shape[1])
                cv2.rectangle(annotated, (0,0), (bar_w,26), (20,20,20), -1)
                cv2.putText(annotated, bar, (6,18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)

                # Colour legend
                fh, fw = annotated.shape[:2]
                for li, (ltxt, lclr) in enumerate([('Focused', (0,200,60)), ('Distracted', (0,165,255)), ('Alert', (0,0,220)), ('Hand Raised', (255,255,0))]):
                    lx = fw-150; ly = 12+li*20
                    cv2.rectangle(annotated, (lx,ly-10), (lx+14,ly+4), lclr, -1)
                    cv2.putText(annotated, ltxt, (lx+18,ly+3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, lclr, 1)

            except Exception as loop_processing_err:
                # If calculations fail, fall back gracefully to the unannotated frame
                print(f'[STREAM LOOP EXCEPTION HANDLED]: {loop_processing_err}')
                annotated = frame

            # ── Encode and Yield Frame ────────────────────────────────────────
            _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if buf is not None and len(buf) > 0:
                try:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break

            elapsed = time.monotonic() - last_yield
            wait    = frame_delay - elapsed
            if wait > 0:
                time.sleep(wait)
            last_yield = time.monotonic()

    finally:
        stop_event.set()
        if cap is not None:
            cap.release()
        det_thread.join(timeout=1)
        pose_thread.join(timeout=1)
        save_thread.join(timeout=1)
        _release_yolo_detector(stream_key)

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

    from classroom_monitor.api_auth import check_detection_api_key
    auth_err = check_detection_api_key(request)
    if auth_err:
        return auth_err

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
        from classroom_monitor.constants import EMAIL_ALERT_TYPES
        inc_type    = data.get('incident_type', 'other')
        severity = _incident_severity(inc_type)

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
        if inc_type in EMAIL_ALERT_TYPES and snapshot_bytes:
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
    from classroom_monitor.constants import LABEL_MAP
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


# ── Video Upload & Analysis Views ─────────────────────────────────────────────

@login_required
def video_list(request):
    videos = ClassroomVideo.objects.all()
    return render(request, 'classroom_monitor/video_list.html', {'videos': videos})


@login_required
def video_upload(request):
    if request.method == 'POST':
        form = ClassroomVideoForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save()
            messages.success(request, f'Video "{video.title}" uploaded. Click Analyze to process it.')
            return redirect('classroom_video_list')
    else:
        form = ClassroomVideoForm()
    return render(request, 'classroom_monitor/video_upload.html', {'form': form})


@login_required
def video_detail(request, pk):
    video = get_object_or_404(ClassroomVideo, pk=pk)
    frames = video.frames.order_by('frame_number')
    return render(request, 'classroom_monitor/video_detail.html', {'video': video, 'frames': frames})


@login_required
def video_delete(request, pk):
    video = get_object_or_404(ClassroomVideo, pk=pk)
    if request.method == 'POST':
        video.delete()
        messages.success(request, 'Video deleted.')
        return redirect('classroom_video_list')
    return render(request, 'classroom_monitor/confirm_delete.html', {'obj': video, 'type': 'Video'})


@login_required
def video_analyze(request, pk):
    """Trigger background analysis of an uploaded video."""
    video = get_object_or_404(ClassroomVideo, pk=pk)
    if video.status in ('processing',):
        return JsonResponse({'status': 'already_processing'})

    video.status = 'processing'
    video.frames.all().delete()
    video.save(update_fields=['status'])

    import threading
    t = threading.Thread(target=_analyze_video_task, args=(video.pk,), daemon=True)
    t.start()

    return JsonResponse({'status': 'started'})


@login_required
def video_analysis_status(request, pk):
    video = get_object_or_404(ClassroomVideo, pk=pk)
    return JsonResponse({
        'status': video.status,
        'frames_analyzed': video.total_frames_analyzed,
        'avg_score': round(video.average_engagement_score, 1),
        'duration': video.duration_seconds,
    })


def _analyze_video_task(video_pk):
    """Run in a background thread — reads the video, samples frames, runs YOLO."""
    import django
    from django.db import close_old_connections
    close_old_connections()

    try:
        video = ClassroomVideo.objects.get(pk=video_pk)
        video_path = video.video_file.path

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            video.status = 'failed'
            video.save(update_fields=['status'])
            return

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / src_fps if src_fps > 0 else 0
        video.duration_seconds = int(duration)
        video.save(update_fields=['duration_seconds'])

        # Sample one frame every 5 seconds
        sample_interval = max(1, int(src_fps * 5))

        # Own detector key per video job — this must NOT be the same
        # detector instance used by any live camera stream (or another video
        # job), since track_id-keyed state (keypoint/behavior history,
        # confidence smoothing) would otherwise get cross-contaminated
        # between unrelated people. 5-second frame sampling also means
        # ByteTrack continuity across samples is weak regardless — that's
        # expected for batch analysis, but it's still important this job's
        # track IDs never collide with a live stream's.
        video_detector_key = f'video:{video_pk}'
        detector = _get_yolo_detector(video_detector_key)

        frame_number = 0
        analyzed_count = 0
        total_score = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % sample_interval == 0:
                timestamp_sec = frame_number / src_fps

                try:
                    dets = detector.detect(frame) if detector else []
                except Exception as e:
                    print(f'[VIDEO ANALYZE] detect error: {e}')
                    dets = []

                focused = looking_away = head_down = using_phone = eating = not_visible = 0
                for det in dets:
                    if isinstance(det, dict):
                        dt = det.get('type', 'not_visible')
                    else:
                        dt = getattr(det, 'type', 'not_visible')
                    if dt == 'focused':
                        focused += 1
                    elif dt in ('distracted', 'looking_away'):
                        looking_away += 1
                    elif dt == 'head_down':
                        head_down += 1
                    elif dt == 'using_phone':
                        using_phone += 1
                    elif dt == 'eating_food':
                        eating += 1
                    else:
                        not_visible += 1

                total_det = focused + looking_away + head_down + using_phone + eating
                score = (focused / total_det * 100) if total_det > 0 else 0.0

                # Save annotated frame image
                annotated = frame.copy()
                for det in dets:
                    if isinstance(det, dict):
                        x1, y1, x2, y2 = det.get('bbox', (0, 0, 0, 0))
                        color = det.get('color', (0, 200, 60))
                        label = det.get('label', det.get('type', ''))
                    else:
                        x1, y1, x2, y2 = getattr(det, 'bbox', (0, 0, 0, 0))
                        color = getattr(det, 'color', (0, 200, 60))
                        label = getattr(det, 'label', '')
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, label, (x1 + 2, max(y1 - 6, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                bar = f'Focused:{focused} Away:{looking_away} Phone:{using_phone} Score:{score:.0f}%'
                cv2.rectangle(annotated, (0, 0), (len(bar) * 9 + 14, 24), (20, 20, 20), -1)
                cv2.putText(annotated, bar, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                ret_enc, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                frame_file = None
                if ret_enc:
                    frame_file = ContentFile(
                        buf.tobytes(),
                        name=f'vf_{video_pk}_{frame_number}.jpg'
                    )

                from django.db import close_old_connections as _cc
                _cc()
                vf = VideoAnalysisFrame.objects.create(
                    video=video,
                    frame_number=frame_number,
                    timestamp=timestamp_sec,
                    frame_image=frame_file,
                    focused_count=focused,
                    looking_away_count=looking_away,
                    head_down_count=head_down,
                    using_phone_count=using_phone,
                    eating_count=eating,
                    not_visible_count=not_visible,
                    total_detected=total_det,
                    engagement_score=score,
                )

                analyzed_count += 1
                total_score += score

                # Update progress periodically
                if analyzed_count % 5 == 0:
                    video.total_frames_analyzed = analyzed_count
                    video.average_engagement_score = total_score / analyzed_count if analyzed_count else 0.0
                    video.save(update_fields=['total_frames_analyzed', 'average_engagement_score'])

            frame_number += 1

        cap.release()

        video.status = 'completed'
        video.processed_at = timezone.now()
        video.total_frames_analyzed = analyzed_count
        video.average_engagement_score = total_score / analyzed_count if analyzed_count else 0.0
        video.save()

    except Exception as e:
        print(f'[VIDEO ANALYZE] Error: {e}')
        import traceback; traceback.print_exc()
        try:
            from django.db import close_old_connections as _cc2
            _cc2()
            ClassroomVideo.objects.filter(pk=video_pk).update(status='failed')
        except Exception:
            pass
    finally:
        try:
            _release_yolo_detector(f'video:{video_pk}')
        except Exception:
            pass