from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('entrance_cam.urls')),
    path('lab/', include('lab_monitor.urls')),
    path('student/', include('lab_monitor.student_urls')),
    path('classroom/', include('classroom_monitor.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) \
  + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
