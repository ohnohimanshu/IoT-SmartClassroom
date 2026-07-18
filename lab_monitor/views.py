from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.utils import timezone
from django.db.models import Avg
from django.core.cache import cache
import json
import base64
from io import BytesIO
from PIL import Image
import os
from collections import Counter
from datetime import datetime

from .models import LabSession, Screenshot, CameraSnapshot, ActivityLog
from entrance_cam.models import Student
from camera_attendance.models import CameraAttendanceLog

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
    global _face_landmarker, _cv2, _np, _mp, _mp_vision
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
            from mediapipe.tasks.python import vision as _mp_vision_mod
            import os
            _mp_vision = _mp_vision_mod

            here = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(os.path.dirname(here), 'face_landmarker.task'),       # app's parent dir
                os.path.join(os.path.dirname(os.path.dirname(here)), 'face_landmarker.task'),  # project root
                os.path.join(here, 'face_landmarker.task'),                        # app dir itself
            ]
            try:
                from django.conf import settings
                base_dir = getattr(settings, 'BASE_DIR', None)
                if base_dir:
                    candidates.insert(0, os.path.join(str(base_dir), 'face_landmarker.task'))
            except Exception:
                pass

            model_path = None
            for c in candidates:
                exists = os.path.exists(c)
                print(f"[FACE_MESH] Checking: {c} -> exists={exists}")
                if exists and model_path is None:
                    model_path = c
            if model_path is None:
                print("[FACE_MESH] face_landmarker.task NOT FOUND in any candidate location. "
                      "Download it from https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                      "face_landmarker/float16/1/face_landmarker.task and place it at one of the paths above. "
                      "Pose detection will stay 'unknown' until this file is present.")
                return None

            print(f"[FACE_MESH] Loading model from: {model_path}")
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
            import traceback
            traceback.print_exc()
    return _face_landmarker


@login_required(login_url='login')
def student_dashboard(request):
    try:
        student = request.user.student_profile
    except Student.DoesNotExist:
        return redirect('dashboard')
    
    attendance_logs = CameraAttendanceLog.objects.filter(student=student)[:10]
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
    # Get snapshots in order
    camera_snapshots = session.camera_snapshots.all().order_by('timestamp')
    screenshots = session.screenshots.all().order_by('timestamp')
    activity_logs = session.activity_logs.all().order_by('-timestamp')

    # ── Build a professional session summary (KPIs + distributions) ───────
    summary = None
    total_frames = camera_snapshots.count()
    if total_frames:
        emotion_counts = {}
        pose_counts = {}
        total_score = 0.0
        for snap in camera_snapshots:
            emotion_counts[snap.emotion] = emotion_counts.get(snap.emotion, 0) + 1
            pose_counts[snap.pose] = pose_counts.get(snap.pose, 0) + 1
            total_score += snap.emotion_score or 0

        focused_frames = pose_counts.get('focused', 0)
        distracted_frames = pose_counts.get('looking_away', 0) + pose_counts.get('head_down', 0)
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        dominant_pose = max(pose_counts, key=pose_counts.get)

        duration_seconds = 0
        if session.start_time:
            end_ref = session.end_time or timezone.now()
            duration_seconds = max(int((end_ref - session.start_time).total_seconds()), 0)

        summary = {
            'total_frames': total_frames,
            'dominant_emotion': dominant_emotion,
            'dominant_pose': dominant_pose,
            'focus_percent': round((focused_frames / total_frames) * 100, 1),
            'distraction_percent': round((distracted_frames / total_frames) * 100, 1),
            'avg_emotion_score': round((total_score / total_frames) * 100, 1),
            'emotion_counts': emotion_counts,
            'pose_counts': pose_counts,
            'alert_count': distracted_frames,
            'duration_display': '%02d:%02d:%02d' % (
                duration_seconds // 3600, (duration_seconds % 3600) // 60, duration_seconds % 60
            ),
        }

    return render(request, 'lab_monitor/monitor_detail.html', {
        'session': session,
        'camera_snapshots': camera_snapshots,
        'screenshots': screenshots,
        'activity_logs': activity_logs,
        'summary': summary,
    })


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
                h, w = np_image.shape[:2]

                print(f"[CAM] shape={np_image.shape} dtype={np_image.dtype}")

                # ── Run mediapipe ONCE (needs read-only array) ────────────────────
                landmarker = _get_face_mesh()
                mesh_results = None
                if landmarker is not None and _mp is not None:
                    np_image.flags.writeable = False
                    try:
                        mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=np_image)
                        mesh_results = landmarker.detect(mp_image)
                        print(f"[MESH] mesh_results={mesh_results}")
                    except Exception as e:
                        print(f"[MESH ERROR] {e}")
                        import traceback
                        traceback.print_exc()
                        mesh_results = None
                    np_image.flags.writeable = True
                else:
                    mesh_results = None

                # New Tasks API: .face_landmarks is a list of NormalizedLandmark lists
                if mesh_results and hasattr(mesh_results, 'face_landmarks') and mesh_results.face_landmarks:
                    face_lm = mesh_results.face_landmarks[0]
                else:
                    face_lm = None
                print(f"[MESH] face detected: {face_lm is not None}")

                # Tightly crop around the face (when mediapipe found one) before
                # handing the frame to DeepFace. Analyzing the whole frame lets
                # background/hair/clothing dilute the emotion signal, which was
                # the main cause of inaccurate emotion readings.
                face_crop = None
                if face_lm is not None:
                    xs = [p.x * w for p in face_lm]
                    ys = [p.y * h for p in face_lm]
                    x_min, x_max = max(min(xs), 0), min(max(xs), w)
                    y_min, y_max = max(min(ys), 0), min(max(ys), h)
                    pad_x = (x_max - x_min) * 0.25
                    pad_y = (y_max - y_min) * 0.35
                    cx0, cy0 = int(max(x_min - pad_x, 0)), int(max(y_min - pad_y, 0))
                    cx1, cy1 = int(min(x_max + pad_x, w)), int(min(y_max + pad_y, h))
                    if cx1 - cx0 > 20 and cy1 - cy0 > 20:
                        face_crop = _np_local.ascontiguousarray(np_image[cy0:cy1, cx0:cx1])

                # ── Emotion ───────────────────────────────────────────────────────
                analysis = None
                deepface_found_face = False
                if _get_deepface():
                    if face_crop is not None:
                        # Face already located precisely by mediapipe — skip
                        # DeepFace's own detector and analyze the crop directly.
                        try:
                            analysis = DeepFace.analyze(face_crop, actions=['emotion'],
                                                        enforce_detection=False,
                                                        detector_backend='skip', silent=True)
                            if isinstance(analysis, list):
                                analysis = analysis[0]
                            deepface_found_face = True
                        except Exception as e:
                            print(f"[DEEPFACE skip ERROR] {e}")
                            analysis = None
                    else:
                        # Mediapipe didn't find a face. IMPORTANT: previously this used
                        # enforce_detection=False, which silently ran the emotion model
                        # on the *entire* frame (background/desk/etc.) whenever DeepFace's
                        # own detector also failed to find a face — a full frame fed to a
                        # face-emotion model reliably collapses to the same wrong label
                        # (this is why "sad" kept showing up even while smiling).
                        # Instead, try progressively stronger real detectors and only
                        # accept a result when one of them actually finds a face.
                        for backend in ('yunet', 'opencv'):
                            try:
                                result = DeepFace.analyze(np_image, actions=['emotion'],
                                                          enforce_detection=True,
                                                          detector_backend=backend, silent=True)
                                analysis = result[0] if isinstance(result, list) else result
                                deepface_found_face = True
                                print(f"[DEEPFACE] face found via backend={backend}")
                                break
                            except Exception as e:
                                print(f"[DEEPFACE {backend}] no face: {e}")
                                continue

                    if deepface_found_face and analysis:
                        emotion = analysis.get('dominant_emotion', 'neutral')
                        # DeepFace reports 0-100 confidence per emotion. Store it as a
                        # 0-1 fraction so it matches how the dashboard chart scales
                        # (it previously stored the raw 0-100 value, which the chart
                        # then multiplied by 100 again — pinning the graph at its max).
                        emotion_score = float(analysis.get('emotion', {}).get(emotion, 50.0)) / 100.0
                    elif face_lm is not None:
                        # Mediapipe saw a face but DeepFace couldn't confirm one —
                        # don't report a fabricated emotion, just say we're unsure.
                        emotion = 'neutral'
                        emotion_score = 0.5
                    else:
                        # No face found by either method this frame — say so honestly
                        # instead of guessing.
                        emotion = 'unknown'
                        emotion_score = 0.0
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
                        emotion_score = min(mar * 2.0, 0.95)
                    elif brow_raise < -0.03:
                        emotion = 'surprise'
                        emotion_score = 0.6
                    else:
                        emotion = 'neutral'
                        emotion_score = 0.7
                else:
                    emotion = 'unknown'
                    emotion_score = 0.0

                # ── Pose (head direction) ─────────────────────────────────────────
                if face_lm is not None:
                    def pt(idx):
                        return _np_local.array([face_lm[idx].x * w, face_lm[idx].y * h])

                    nose      = pt(1)
                    left_eye  = pt(33)
                    right_eye = pt(263)
                    chin      = pt(152)
                    forehead  = pt(10)

                    # Distance-ratio approach instead of raw pixel offsets normalized
                    # by eye-width. Eye-width itself shrinks from perspective
                    # foreshortening as the head turns, which was partly cancelling
                    # out the very signal we were trying to measure — that's why
                    # yaw/pitch rarely crossed the old thresholds. Comparing
                    # nose-to-eye and nose-to-chin/forehead distances against each
                    # other is scale-invariant and swings much more clearly.
                    d_left    = _np_local.linalg.norm(nose - left_eye) + 1e-6
                    d_right   = _np_local.linalg.norm(nose - right_eye) + 1e-6
                    d_top     = _np_local.linalg.norm(nose - forehead) + 1e-6
                    d_bottom  = _np_local.linalg.norm(nose - chin) + 1e-6

                    yaw_ratio   = (d_right - d_left) / (d_right + d_left)
                    # Nose gets relatively closer to the chin (chin tucks toward
                    # the nose in the 2D image) when looking down, and relatively
                    # closer to the forehead when looking up.
                    pitch_ratio = (d_top - d_bottom) / (d_top + d_bottom)

                    print(f"[POSE] yaw_ratio={yaw_ratio:.3f} pitch_ratio={pitch_ratio:.3f}")

                    if abs(yaw_ratio) > 0.12:
                        pose = 'looking_away'
                    elif pitch_ratio < -0.10:   # nose pulled toward chin: head down
                        pose = 'head_down'
                    elif pitch_ratio > 0.18:    # nose pulled toward forehead: looking up/away
                        pose = 'looking_away'
                    else:
                        pose = 'focused'
                elif deepface_found_face and analysis:
                    # Mediapipe missed the face this frame, but DeepFace's own
                    # detector may still have located it — use its bounding box
                    # and eye positions for a coarse pose estimate instead of
                    # always giving up with "unknown".
                    region = analysis.get('region') or {}
                    le, re = region.get('left_eye'), region.get('right_eye')
                    rw, rh, rx, ry = region.get('w'), region.get('h'), region.get('x'), region.get('y')
                    if le and re and rw and rh:
                        eye_cx, eye_cy = (le[0] + re[0]) / 2, (le[1] + re[1]) / 2
                        bbox_cx, bbox_cy = rx + rw / 2, ry + rh / 2
                        yaw_ratio = (eye_cx - bbox_cx) / (rw / 2 + 1e-6)
                        pitch_ratio = (eye_cy - bbox_cy) / (rh / 2 + 1e-6)
                        print(f"[POSE-fallback] yaw_ratio={yaw_ratio:.2f} pitch_ratio={pitch_ratio:.2f}")
                        if abs(yaw_ratio) > 0.18:
                            pose = 'looking_away'
                        elif pitch_ratio > 0.10:
                            pose = 'head_down'
                        elif pitch_ratio < -0.22:
                            pose = 'looking_away'
                        else:
                            pose = 'focused'
                    else:
                        print("[POSE] No usable face region from DeepFace either - unknown")
                        pose = 'unknown'
                else:
                    print("[POSE] No face landmarks detected - defaulting to unknown")
                    pose = 'unknown'  # Fallback to unknown if we can't detect a face

                # ── Smooth over the last few frames ────────────────────────────────
                # A single noisy frame shouldn't flip the badge/graph; vote/average
                # across the last 3 readings for this session for a stable signal.
                smoothing_key = f'lab_smooth_{session_id}'
                recent = cache.get(smoothing_key, [])
                recent.append({'emotion': emotion, 'score': emotion_score, 'pose': pose})
                recent = recent[-3:]
                cache.set(smoothing_key, recent, timeout=30)

                emotion = Counter(r['emotion'] for r in recent).most_common(1)[0][0]
                emotion_score = sum(r['score'] for r in recent) / len(recent)
                known_poses = [r['pose'] for r in recent if r['pose'] != 'unknown']
                if known_poses:
                    pose = Counter(known_poses).most_common(1)[0][0]

                snapshot = CameraSnapshot(session=session, emotion=emotion, emotion_score=emotion_score, pose=pose)
                filename = f"camera_{session.id}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
                snapshot.image.save(filename, BytesIO(data_img), save=True)

            except Exception as e:
                print(f"[CAMERA FRAME ERROR] {e}")
                import traceback
                traceback.print_exc()
        
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