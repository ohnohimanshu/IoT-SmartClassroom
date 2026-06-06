from django.urls import path
from . import views

app_name = 'camera_attendance'

urlpatterns = [
    # Dashboard
    path('', views.camera_attendance_dashboard, name='dashboard'),

    # Attendance list
    path('attendance/', views.camera_attendance_list, name='attendance_list'),

    # Camera management
    path('cameras/', views.camera_list, name='camera_list'),
    path('cameras/add/', views.camera_add, name='camera_add'),
    path('cameras/<int:pk>/edit/', views.camera_edit, name='camera_edit'),
    path('cameras/<int:pk>/delete/', views.camera_delete, name='camera_delete'),
    path('cameras/<int:pk>/test/', views.camera_test, name='camera_test'),
    path('cameras/<int:pk>/proxy-stream/', views.camera_proxy_stream, name='camera_proxy_stream'),

    # API endpoints
    path('api/log/', views.api_log_camera_attendance, name='api_log'),
    path('api/students/encodings/', views.api_camera_students_encodings, name='api_students_encodings'),
    path('api/live-detections/', views.api_camera_live_detections, name='api_live_detections'),
]
