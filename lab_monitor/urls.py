from django.urls import path
from . import views

urlpatterns = [
    path('monitor/', views.monitor_dashboard, name='lab_monitor'),
    path('monitor/<int:session_id>/', views.monitor_detail, name='monitor_detail'),
    path('sessions/', views.session_list, name='session_list'),
    path('session/start/', views.session_start, name='session_start'),
    path('session/end/', views.session_end, name='session_end'),
    path('screenshot/', views.receive_screenshot, name='receive_screenshot'),
    path('camera-frame/', views.receive_camera_frame, name='receive_camera_frame'),
    path('activity/', views.receive_activity, name='receive_activity'),
    path('api/active-sessions/', views.api_active_sessions, name='api_active_sessions'),
    path('api/session/<int:session_id>/', views.api_session_detail, name='api_session_detail'),
    path('api/session/<int:session_id>/webrtc/offer/', views.api_webrtc_offer, name='api_webrtc_offer'),
    path('api/session/<int:session_id>/webrtc/answer/', views.api_webrtc_answer, name='api_webrtc_answer'),
    path('api/session/<int:session_id>/webrtc/ice/', views.api_webrtc_ice_candidate, name='api_webrtc_ice_candidate'),
]
