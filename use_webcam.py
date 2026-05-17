import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'classroom_iot.settings')
django.setup()

from entrance_cam.models import Camera

# Switch to webcam index 0 for testing
for camera in Camera.objects.all():
    camera.url = '0'
    camera.save()
    print(f"Set camera {camera.id} ({camera.name}) to use webcam index 0 for testing")

print("Camera set to webcam! Restart Django server for changes to take effect!")
