from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/', views.dashboard, name='classroom_dashboard'),
    path('session/<int:session_id>/', views.session_detail, name='classroom_session_detail'),
    path('sessions/', views.session_list, name='classroom_session_list'),
    path('cameras/', views.camera_list, name='classroom_camera_list'),
    path('cameras/add/', views.camera_add, name='classroom_camera_add'),
    path('cameras/<int:pk>/edit/', views.camera_edit, name='classroom_camera_edit'),
    path('cameras/<int:pk>/delete/', views.camera_delete, name='classroom_camera_delete'),
    path('cameras/<int:pk>/test/', views.camera_test, name='classroom_camera_test'),
    path('cameras/<int:camera_id>/stream/', views.live_stream, name='classroom_live_stream'),

    path('session/start/', views.session_start, name='classroom_session_start'),
    path('session/end/', views.session_end, name='classroom_session_end'),

    path('api/snapshot/', views.api_snapshot, name='classroom_api_snapshot'),
    path('api/session/active/', views.api_active_session, name='classroom_api_active_session'),
    path('api/stats/<int:session_id>/', views.api_stats, name='classroom_api_stats'),

    path('live/', views.live_monitor, name='classroom_live_monitor'),
    path('live/<int:camera_id>/', views.live_camera_detail, name='classroom_live_camera_detail'),
]
