from django.urls import path
from . import views

urlpatterns = [
    # ── Auth ──────────────────────────────────────────────────────────────────
    path('',        views.dashboard,    name='dashboard'),
    path('login/',  views.login_view,   name='login'),
    path('logout/', views.logout_view,  name='logout'),

    # ── Students ──────────────────────────────────────────────────────────────
    path('students/',                  views.student_list,   name='student_list'),
    path('students/add/',              views.student_add,    name='student_add'),
    path('students/<int:pk>/edit/',    views.student_edit,   name='student_edit'),
    path('students/<int:pk>/delete/',  views.student_delete, name='student_delete'),
    path('students/<int:pk>/',         views.student_detail, name='student_detail'),
    path('students/<int:student_id>/enroll-fingerprint/', views.enroll_fingerprint, name='enroll_fingerprint'),
    path('students/<int:student_id>/enrollment-status/', views.enrollment_status, name='enrollment_status'),
    path('students/<int:student_id>/fingerprint-image/', views.view_fingerprint_image, name='view_fingerprint_image'),

    # ── Cameras ───────────────────────────────────────────────────────────────
    path('cameras/',                   views.camera_list,   name='camera_list'),
    path('cameras/add/',               views.camera_add,    name='camera_add'),
    path('cameras/<int:pk>/edit/',     views.camera_edit,   name='camera_edit'),
    path('cameras/<int:pk>/delete/',   views.camera_delete, name='camera_delete'),
    path('cameras/<int:pk>/test/',     views.camera_test,   name='camera_test'),
    # Proxy: serves ESP32 HTTP stream over HTTPS to avoid browser Mixed Content block
    path('cameras/<int:pk>/proxy-stream/', views.camera_proxy_stream, name='camera_proxy_stream'),

    # ── ESP32 Devices ─────────────────────────────────────────────────────────
    path('esp32-devices/',                  views.esp32_device_list,   name='esp32_device_list'),
    path('esp32-devices/add/',              views.esp32_device_add,    name='esp32_device_add'),
    path('esp32-devices/<int:pk>/edit/',     views.esp32_device_edit,   name='esp32_device_edit'),
    path('esp32-devices/<int:pk>/delete/',   views.esp32_device_delete, name='esp32_device_delete'),

    # ── Attendance ────────────────────────────────────────────────────────────
    path('attendance/', views.attendance_list, name='attendance_list'),

    # ── API ───────────────────────────────────────────────────────────────────
    # Primary endpoint used by detection_script.py
    path('api/log/', views.api_log_entry, name='api_log_entry'),

    # Legacy alias — keeps any old script version working without changes
    path('api/attendance/log/', views.api_log_entry, name='api_log_entry_legacy'),

    # Student face encodings — fetched by detection_script.py on startup
    path('api/students/encodings/', views.api_students_encodings, name='api_students_encodings'),

    # Live modal: today's detections for a camera (polled every 10 s by the modal)
    path('api/live-detections/', views.api_live_detections, name='api_live_detections'),

    # ESP32 API endpoints (matching Flask app)
    path('api/esp32/command/', views.api_esp32_command, name='api_esp32_command'),
    path('api/esp32/enroll-result/', views.api_esp32_enroll_result, name='api_esp32_enroll_result'),
    path('api/esp32/upload-image/', views.api_esp32_upload_image, name='api_esp32_upload_image'),
    path('api/mark-attendance/', views.api_mark_fingerprint_attendance, name='api_mark_fingerprint_attendance'),
]