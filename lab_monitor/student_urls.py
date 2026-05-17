from django.urls import path
from django.shortcuts import redirect
from . import views

def redirect_to_main_login(request):
    return redirect('login')

def redirect_to_main_logout(request):
    return redirect('logout')

urlpatterns = [
    path('login/', redirect_to_main_login, name='student_login'),
    path('logout/', redirect_to_main_logout, name='student_logout'),
    path('dashboard/', views.student_dashboard, name='student_dashboard'),
    path('session/', views.student_session, name='student_session'),
]
