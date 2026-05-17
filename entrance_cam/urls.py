from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # Students
    path('students/', views.student_list, name='student_list'),
    path('students/add/', views.student_add, name='student_add'),
    path('students/<int:pk>/edit/', views.student_edit, name='student_edit'),
    path('students/<int:pk>/delete/', views.student_delete, name='student_delete'),
    path('students/<int:pk>/', views.student_detail, name='student_detail'),

    # Cameras
    path('cameras/', views.camera_list, name='camera_list'),
    path('cameras/add/', views.camera_add, name='camera_add'),
    path('cameras/<int:pk>/edit/', views.camera_edit, name='camera_edit'),
    path('cameras/<int:pk>/delete/', views.camera_delete, name='camera_delete'),
    path('cameras/<int:pk>/test/', views.camera_test, name='camera_test'),

    # Attendance
    path('attendance/', views.attendance_list, name='attendance_list'),

    # API
    path('api/log/', views.api_log_entry, name='api_log_entry'),
    path('api/students/encodings/', views.api_students_encodings, name='api_students_encodings'),
]
