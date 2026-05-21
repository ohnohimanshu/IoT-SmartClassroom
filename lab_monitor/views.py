from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.utils import timezone
from django.db.models import Avg
import json
import base64
from io import BytesIO
from PIL import Image
import os
from datetime import datetime

from .models import LabSession, Screenshot, CameraSnapshot, ActivityLog
from entrance_cam.models import Student, AttendanceLog

DEEPFACE_AVAILABLE = None  # None = not yet checked; True/False after first use


def _get_deepface():
    """Lazy-load DeepFace so TensorFlow doesn't block Django startup."""
    global DEEPFACE_AVAILABLE
    if DEEPFACE_AVAILABLE is None:
        try:
            from deepface import DeepFace as _DF
            globals()['DeepFace'] = _DF
            DEEPFACE_AVAILABLE = True
        except ImportError:
            DEEPFACE_AVAILABLE = False
    return DEEPFACE_AVAILABLE


# cv2 / mediapipe / numpy — lazy so they don't block Django startup
_face_landmarker = None
_cv2 = None
_np = None
_mp = None


def _get_face_mesh():
    """Lazy-load mediapipe Tasks API FaceLandmarker (mediapipe >= 0.10)."""
    global _face_landmarker, _cv2, _np, _mp
    if _cv2 is None:
        try:
            import cv2 as _cv2_mod
            import numpy as _np_mod
            import mediapipe as _mp_mod
            _cv2, _np, _mp = _cv2_mod, _np_mod, _mp_mod
            globals().update({'cv2': _cv2, 'np': _np, 'mp': _mp})
        except ImportError as e:
            print(f"[FACE_MESH] Import failed: {e}")
            return None
    if _face_landmarker is None:
        try:
            from mediapipe.tasks import python as _mp_python
            from mediapipe.tasks.python import vision as _mp_vision
            import os
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'face_landmarker.task'
            )
            base_options = _mp_python.BaseOptions(model_asset_path=model_path)
            options = _mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=_mp_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            _face_landmarker = _mp_vision.FaceLandmarker.create_from_options(options)
            print("[FACE_MESH] FaceLandmarker (Tasks API) initialized successfully")
        except Exception as e:
            print(f"[FACE_MESH] Init failed: {e}")
    return _face_landmarker


@login_required(login_url='login')
def student_dashboard(request):
    try:
        student = request.user.student_profile
    except Student.DoesNotExist:
        return redirect('dashboard')
    
    attendance_logs = AttendanceLog.objects.filter(student=student)[:10]
    past_sessions = LabSession.objects.filter(student=student, is_active=False)[:10]
    
    return render(request, 'lab_monitor/student_dashboard.html', {
        'student': student,
        'attendance_logs': attendance_logs,
        'past_sessions': past_sessions
    })


@login_required(login_url='login')
def student_session(request):
    try:
        student = request.user.student_profile
    except Student.DoesNotExist:
        return redirect('dashboard')
    return render(request, 'lab_monitor/student_session.html', {'student': student})


@login_required
def monitor_dashboard(request):
    print(f"DEBUG: monitor_dashboard - User: {request.user}, is_staff: {request.user.is_staff}, is_superuser: {request.user.is_superuser}")
    if not request.user.is_staff and not request.user.is_superuser:
        return render(request, 'lab_monitor/error.html', {
            'message': 'You must be an admin to access this page.',
            'current_user': request.user.username,
            'is_staff': request.user.is_staff,
            'is_superuser': request.user.is_superuser
        })
    active_sessions = LabSession.objects.filter(is_active=True)
    return render(request, 'lab_monitor/monitor_dashboard.html', {'active_sessions': active_sessions})


@login_required
def monitor_detail(request, session_id):
    print(f"DEBUG: monitor_detail - User: {request.user}, is_staff: {request.user.is_staff}, is_superuser: {request.user.is_superuser}")
    if not request.user.is_staff and not request.user.is_superuser:
        return render(request, 'lab_monitor/error.html', {
            'message': 'You must be an admin to access this page.',
            'current_user': request.user.username,
            'is_staff': request.user.is_staff,
            'is_superuser': request.user.is_superuser
        })
    session = get_object_or_404(LabSession, id=session_id)
    return render(request, 'lab_monitor/monitor_detail.html', {'session': session})


@login_required
def session_list(request):
    print(f"DEBUG: session_list - User: {request.user}, is_staff: {request.user.is_staff}, is_superuser: {request.user.is_superuser}")
    if not request.user.is_staff and not request.user.is_superuser:
        return render(request, 'lab_monitor/error.html', {
            'message': 'You must be an admin to access this page.',
            'current_user': request.user.username,
            'is_staff': request.user.is_staff,
            'is_superuser': request.user.is_superuser
        })
    date_filter = request.GET.get('date')
    student_filter = request.GET.get('student')
    
    sessions = LabSession.objects.all()
    
    if date_filter:
        try:
            date_obj = datetime.strptime(date_filter, '%Y-%m-%d').date()
            sessions = sessions.filter(start_time__date=date_obj)
        except ValueError:
            pass
    
    if student_filter:
        sessions = sessions.filter(student_id=student_filter)
    
    students = Student.objects.all()
    return render(request, 'lab_monitor/session_list.html', {'sessions': sessions, 'students': students})


@require_POST
def session_start(request):
    if not request.user.is_authenticated:
        return JsonResponse({'status': 'error', 'message': 'Not authenticated'}, status=401)
    try:
        student = request.user.student_profile
    except Student.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Not a student'}, status=400)
    
    # End any existing active session first
    active_session = LabSession.objects.filter(student=student, is_active=True).first()
    if active_session:
        active_session.end_time = timezone.now()
        active_session.is_active = False
        active_session.calculate_duration()
        active_session.save()
    
    session = LabSession.objects.create(student=student)
    return JsonResponse({'status': 'success', 'session_id': session.id})


@require_POST
def session_end(request):
    if not request.user.is_authenticated:
        return JsonResponse({'status': 'error', 'message': 'Not authenticated'}, status=401)
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')
        try:
            student = request.user.student_profile
        except Student.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Not a student'}, status=400)
        
        session = get_object_or_404(LabSession, id=session_id, student=student)
        session.end_time = timezone.now()
        session.is_active = False
        session.calculate_duration()
        session.save()
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})


@require_POST
def receive_screenshot(request):
    if not request.user.is_authenticated:
        return JsonResponse({'status': 'error', 'message': 'Not authenticated'}, status=401)
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')
        image_b64 = data.get('image_b64')
        tab_title = data.get('tab_title', '')
        
        try:
            student = request.user.student_profile
        except Student.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Not a student'}, status=400)
        
        session = get_object_or_404(LabSession, id=session_id, student=student)
        
        if image_b64:
            format, imgstr = image_b64.split(';base64,')
            ext = format.split('/')[-1]
            data_img = base64.b64decode(imgstr)
            image = Image.open(BytesIO(data_img))
            
            screenshot = Screenshot(session=session, tab_title=tab_title)
            filename = f"screenshot_{session.id}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
            screenshot.image.save(filename, BytesIO(data_img), save=True)
        
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})


@require_POST
def receive_camera_frame(request):
    if not request.user.is_authenticated:
        return JsonResponse({'status': 'error', 'message': 'Not authenticated'}, status=401)
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')
        image_b64 = data.get('image_b64')
        
        try:
            student = request.user.student_profile
        except Student.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Not a student'}, status=400)
        
        session = get_object_or_404(LabSession, id=session_id, student=student)
        
        emotion = 'unknown'
        emotion_score = 0.0
        pose = 'unknown'
        
        if image_b64:
            try:
                if ';base64,' in image_b64:
                    format, imgstr = image_b64.split(';base64,')
                else:
                    imgstr = image_b64
                    format = 'image/jpeg'
                ext = format.split('/')[-1]
                data_img = base64.b64decode(imgstr)
                
                image = Image.open(BytesIO(data_img)).convert('RGB')
                import numpy as _np_local
                import math as _math
                # ascontiguousarray ensures mediapipe gets a proper C-contiguous buffer
                np_image = _np_local.ascontiguousarray(_np_local.array(image))

                print(f"[CAM] shape={np_image.shape} dtype={np_image.dtype}")

                # ── Run mediapipe ONCE (needs read-only array) ────────────────────
                landmarker = _get_face_mesh()
                if landmarker is not None:
                    np_image.flags.writeable = False
                    try:
                        mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=np_image)
                        mesh_results = landmarker.detect(mp_image)
                    except Exception as e:
                        print(f"[MESH ERROR] {e}")
                        mesh_results = None
                    np_image.flags.writeable = True
                else:
                    mesh_results = None

                # New Tasks API: .face_landmarks is a list of NormalizedLandmark lists
                face_lm = (mesh_results.face_landmarks[0]
                           if mesh_results and mesh_results.face_landmarks else None)
                print(f"[MESH] face detected: {face_lm is not None}")

                # ── Emotion ───────────────────────────────────────────────────────
                if _get_deepface():
                    try:
                        analysis = DeepFace.analyze(np_image, actions=['emotion'],
                                                    enforce_detection=False, silent=True)
                        if isinstance(analysis, list):
                            analysis = analysis[0]
                        emotion = analysis.get('dominant_emotion', 'neutral')
                        emotion_score = float(analysis.get('emotion', {}).get(emotion, 0.5))
                    except Exception:
                        emotion = 'neutral'
                        emotion_score = 0.5
                elif face_lm is not None:
                    upper = face_lm[13]; lower = face_lm[14]
                    left_m = face_lm[78]; right_m = face_lm[308]
                    mar_v = _math.hypot(upper.x - lower.x, upper.y - lower.y)
                    mar_h = _math.hypot(left_m.x - right_m.x, left_m.y - right_m.y) + 1e-6
                    mar = mar_v / mar_h
                    left_brow = face_lm[107]; right_brow = face_lm[336]
                    left_eye_lm = face_lm[33]; right_eye_lm = face_lm[263]
                    brow_raise = ((left_brow.y - left_eye_lm.y) + (right_brow.y - right_eye_lm.y)) / 2
                    if mar > 0.25:
                        emotion = 'happy'
                        emotion_score = min(float(mar * 200), 95.0)
                    elif brow_raise < -0.03:
                        emotion = 'surprise'
                        emotion_score = 60.0
                    else:
                        emotion = 'neutral'
                        emotion_score = 70.0
                else:
                    emotion = 'unknown'
                    emotion_score = 0.0

                # ── Pose (head direction) ─────────────────────────────────────────
                if face_lm is not None:
                    h, w = np_image.shape[:2]

                    def pt(idx):
                        return _np_local.array([face_lm[idx].x * w, face_lm[idx].y * h])

                    nose      = pt(1)
                    left_eye  = pt(33)
                    right_eye = pt(263)
                    chin      = pt(152)
                    forehead  = pt(10)

                    eye_center  = (left_eye + right_eye) / 2
                    eye_width   = _np_local.linalg.norm(right_eye - left_eye) + 1e-6
                    face_height = _np_local.linalg.norm(chin - forehead) + 1e-6

                    yaw   = ((nose[0] - eye_center[0]) / eye_width) * 90
                    pitch = ((nose[1] - eye_center[1]) / face_height) * 180

                    print(f"[POSE] yaw={yaw:.1f} pitch={pitch:.1f} eye_w={eye_width:.1f}")

                    if yaw < -18 or yaw > 18:
                        pose = 'looking_away'
                    elif pitch < -12:
                        pose = 'head_down'
                    else:
                        pose = 'focused'
                else:
                    print("[POSE] No face landmarks detected")
                    pose = 'unknown'

                snapshot = CameraSnapshot(session=session, emotion=emotion, emotion_score=emotion_score, pose=pose)
                filename = f"camera_{session.id}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
                snapshot.image.save(filename, BytesIO(data_img), save=True)

            except Exception as e:
                print(f"[CAMERA FRAME ERROR] {e}")
                import traceback; traceback.print_exc()
        
        return JsonResponse({'status': 'success', 'emotion': emotion, 'emotion_score': emotion_score, 'pose': pose})
    except Exception:
        return JsonResponse({'status': 'success', 'emotion': 'unknown', 'emotion_score': 0.0, 'pose': 'unknown'})


@require_POST
def receive_activity(request):
    if not request.user.is_authenticated:
        return JsonResponse({'status': 'error', 'message': 'Not authenticated'}, status=401)
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')
        tab_title = data.get('tab_title', '')
        activity_type = data.get('activity_type', 'active')
        
        try:
            student = request.user.student_profile
        except Student.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Not a student'}, status=400)
        
        session = get_object_or_404(LabSession, id=session_id, student=student)
        ActivityLog.objects.create(session=session, tab_title=tab_title, activity_type=activity_type)
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})


@login_required
def api_active_sessions(request):
    if not request.user.is_staff and not request.user.is_superuser:
        return JsonResponse({'sessions': []})
    
    active_sessions = LabSession.objects.filter(is_active=True)
    sessions_data = []
    for session in active_sessions:
        latest_screenshot = session.screenshots.first()
        latest_camera = session.camera_snapshots.first()
        
        duration = 0
        if session.start_time:
            duration = int((timezone.now() - session.start_time).total_seconds() / 60)
        
        sessions_data.append({
            'id': session.id,
            'student_name': session.student.name,
            'student_roll_no': session.student.roll_no,
            'student_photo': session.student.photo.url if session.student.photo else None,
            'duration': duration,
            'latest_screenshot': latest_screenshot.image.url if latest_screenshot else None,
            'latest_emotion': latest_camera.emotion if latest_camera else 'unknown',
            'latest_pose': latest_camera.pose if latest_camera else 'unknown',
        })
    return JsonResponse({'sessions': sessions_data})


@login_required
def api_session_detail(request, session_id):
    if not request.user.is_staff and not request.user.is_superuser:
        return JsonResponse({'error': 'Unauthorized'}, status=403)
    
    session = get_object_or_404(LabSession, id=session_id)
    latest_screenshot = session.screenshots.first()
    latest_camera = session.camera_snapshots.first()
    
    duration = 0
    if session.start_time:
        duration = int((timezone.now() - session.start_time).total_seconds() / 60)
    
    return JsonResponse({
        'id': session.id,
        'duration': duration,
        'latest_screenshot': latest_screenshot.image.url if latest_screenshot else None,
        'latest_screenshot_timestamp': latest_screenshot.timestamp.isoformat() if latest_screenshot else None,
        'latest_camera': latest_camera.image.url if latest_camera else None,
        'latest_camera_timestamp': latest_camera.timestamp.isoformat() if latest_camera else None,
        'latest_emotion': latest_camera.emotion if latest_camera else 'unknown',
        'latest_pose': latest_camera.pose if latest_camera else 'unknown',
        'latest_emotion_score': latest_camera.emotion_score if latest_camera else 0,
        'is_active': session.is_active,
        'webrtc_offer': session.webrtc_offer,
        'webrtc_answer': session.webrtc_answer,
        'webrtc_ice_candidates_student': session.webrtc_ice_candidates_student,
        'webrtc_ice_candidates_admin': session.webrtc_ice_candidates_admin,
        'webrtc_screen_stream_id': session.webrtc_screen_stream_id,
        'webrtc_camera_stream_id': session.webrtc_camera_stream_id,
    })


@require_POST
def api_webrtc_offer(request, session_id):
    if not request.user.is_authenticated:
        return JsonResponse({'status': 'error', 'message': 'Not authenticated'}, status=401)
    try:
        data = json.loads(request.body)
        try:
            student = request.user.student_profile
        except Student.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Not a student'}, status=400)
        
        session = get_object_or_404(LabSession, id=session_id, student=student)
        session.webrtc_offer = data.get('offer')
        session.webrtc_ice_candidates_student = []
        session.webrtc_screen_stream_id = data.get('screen_stream_id', '')
        session.webrtc_camera_stream_id = data.get('camera_stream_id', '')
        session.save()
        
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


@require_POST
def api_webrtc_answer(request, session_id):
    if not request.user.is_authenticated:
        return JsonResponse({'status': 'error', 'message': 'Not authenticated'}, status=401)
    if not request.user.is_staff and not request.user.is_superuser:
        return JsonResponse({'error': 'Unauthorized'}, status=403)
    
    try:
        data = json.loads(request.body)
        session = get_object_or_404(LabSession, id=session_id)
        session.webrtc_answer = data.get('answer')
        session.save()
        
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


@require_POST
def api_webrtc_ice_candidate(request, session_id):
    if not request.user.is_authenticated:
        return JsonResponse({'status': 'error', 'message': 'Not authenticated'}, status=401)
    try:
        data = json.loads(request.body)
        candidate = data.get('candidate')
        role = data.get('role')  # 'student' or 'admin'
        
        try:
            student = request.user.student_profile
            is_student = True
        except Student.DoesNotExist:
            is_student = False
        
        if is_student:
            session = get_object_or_404(LabSession, id=session_id, student=student)
            if role == 'student':
                candidates = session.webrtc_ice_candidates_student or []
                candidates.append(candidate)
                session.webrtc_ice_candidates_student = candidates
                session.save()
        else:
            if not (request.user.is_staff or request.user.is_superuser):
                return JsonResponse({'error': 'Unauthorized'}, status=403)
            
            session = get_object_or_404(LabSession, id=session_id)
            if role == 'admin':
                candidates = session.webrtc_ice_candidates_admin or []
                candidates.append(candidate)
                session.webrtc_ice_candidates_admin = candidates
                session.save()
        
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


@login_required
def api_student_session_webrtc(request, session_id):
    try:
        student = request.user.student_profile
    except Student.DoesNotExist:
        return JsonResponse({'error': 'Not a student'}, status=400)
    
    session = get_object_or_404(LabSession, id=session_id, student=student)
    return JsonResponse({
        'webrtc_answer': session.webrtc_answer,
        'webrtc_ice_candidates_admin': session.webrtc_ice_candidates_admin,
    })
