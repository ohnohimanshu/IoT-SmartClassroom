from django.urls import path
from . import views

urlpatterns = [
    # ── Auth ──────────────────────────────────────────────────────────────────
    path('', views.dashboard, name='dashboard'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # ── Students ──────────────────────────────────────────────────────────────
    path('students/', views.student_list, name='student_list'),
    path('students/add/', views.student_add, name='student_add'),
    path('students/<int:pk>/edit/', views.student_edit, name='student_edit'),
    path('students/<int:pk>/delete/', views.student_delete, name='student_delete'),
    path('students/<int:pk>/', views.student_detail, name='student_detail'),
    path('students/<int:student_id>/enroll-fingerprint/', views.enroll_fingerprint, name='enroll_fingerprint'),
    path('students/<int:student_id>/enrollment-status/', views.enrollment_status, name='enrollment_status'),
    path('students/<int:student_id>/fingerprint-image/', views.view_fingerprint_image, name='view_fingerprint_image'),

    # ── ESP32 Devices ─────────────────────────────────────────────────────────
    path('esp32-devices/', views.esp32_device_list, name='esp32_device_list'),
    path('esp32-devices/add/', views.esp32_device_add, name='esp32_device_add'),
    path('esp32-devices/<int:pk>/edit/', views.esp32_device_edit, name='esp32_device_edit'),
    path('esp32-devices/<int:pk>/delete/', views.esp32_device_delete, name='esp32_device_delete'),

    # ── Attendance ────────────────────────────────────────────────────────────
    path('attendance/', views.attendance_list, name='attendance_list'),

    # ── Admin Management ──────────────────────────────────────────────────────
    path('admins/', views.manage_admins, name='manage_admins'),
    path('admins/<int:pk>/delete/', views.delete_admin, name='delete_admin'),

    # ── ESP32 API endpoints (matching Flask app) ─────────────────────────────
    path('api/esp32/command/', views.api_esp32_command, name='api_esp32_command'),
    path('api/esp32/enroll-result/', views.api_esp32_enroll_result, name='api_esp32_enroll_result'),
    path('api/esp32/upload-image/', views.api_esp32_upload_image, name='api_esp32_upload_image'),
    path('api/mark-attendance/', views.api_mark_fingerprint_attendance, name='api_mark_fingerprint_attendance'),
]