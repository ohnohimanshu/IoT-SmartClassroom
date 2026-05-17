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

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

try:
    import mediapipe as mp
    import cv2
    import numpy as np
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


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
@login_required
def session_start(request):
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
@login_required
def session_end(request):
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
@login_required
def receive_screenshot(request):
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
@login_required
def receive_camera_frame(request):
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
                
                image = Image.open(BytesIO(data_img))
                np_image = np.array(image)
                if len(np_image.shape) == 2:
                    np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
                elif np_image.shape[2] == 4:
                    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2RGB)
                
                if DEEPFACE_AVAILABLE:
                    try:
                        analysis = DeepFace.analyze(np_image, actions=['emotion'], enforce_detection=False, silent=True)
                        if analysis:
                            if isinstance(analysis, list):
                                analysis = analysis[0]
                            emotion = analysis.get('dominant_emotion', 'neutral')
                            emotion_score = analysis.get('emotion', {}).get(emotion, 0.5)
                    except Exception:
                        emotion = 'neutral'
                        emotion_score = 0.5
                else:
                    emotion = 'neutral'
                    emotion_score = 0.5
                
                if MEDIAPIPE_AVAILABLE:
                    try:
                        mp_face_mesh = mp.solutions.face_mesh
                        face_mesh = mp_face_mesh.FaceMesh(
                            static_image_mode=True, 
                            max_num_faces=1, 
                            refine_landmarks=True, 
                            min_detection_confidence=0.5
                        )
                        results = face_mesh.process(np_image)
                        if results.multi_face_landmarks:
                            landmarks = results.multi_face_landmarks[0].landmark
                            
                            nose_tip = landmarks[1]
                            left_eye = landmarks[33]
                            right_eye = landmarks[263]
                            
                            eye_center_x = (left_eye.x + right_eye.x) / 2
                            eye_center_y = (left_eye.y + right_eye.y) / 2
                            
                            yaw = (nose_tip.x - eye_center_x) * 100
                            pitch = (nose_tip.y - eye_center_y) * 100
                            
                            if yaw > 20 or yaw < -20:
                                pose = 'looking_away'
                            elif pitch > 15:
                                pose = 'head_down'
                            else:
                                pose = 'focused'
                    except Exception:
                        pose = 'focused'
                else:
                    pose = 'focused'
                
                snapshot = CameraSnapshot(session=session, emotion=emotion, emotion_score=emotion_score, pose=pose)
                filename = f"camera_{session.id}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
                snapshot.image.save(filename, BytesIO(data_img), save=True)
            
            except Exception:
                pass
        
        return JsonResponse({'status': 'success', 'emotion': emotion, 'emotion_score': emotion_score, 'pose': pose})
    except Exception:
        return JsonResponse({'status': 'success', 'emotion': 'unknown', 'emotion_score': 0.0, 'pose': 'unknown'})


@require_POST
@login_required
def receive_activity(request):
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
        'is_active': session.is_active,
        'webrtc_offer': session.webrtc_offer,
        'webrtc_answer': session.webrtc_answer,
        'webrtc_ice_candidates_student': session.webrtc_ice_candidates_student,
        'webrtc_ice_candidates_admin': session.webrtc_ice_candidates_admin,
    })


@require_POST
@login_required
def api_webrtc_offer(request, session_id):
    try:
        data = json.loads(request.body)
        try:
            student = request.user.student_profile
        except Student.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Not a student'}, status=400)
        
        session = get_object_or_404(LabSession, id=session_id, student=student)
        session.webrtc_offer = data.get('offer')
        session.webrtc_ice_candidates_student = []
        session.save()
        
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


@require_POST
@login_required
def api_webrtc_answer(request, session_id):
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
@login_required
def api_webrtc_ice_candidate(request, session_id):
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
