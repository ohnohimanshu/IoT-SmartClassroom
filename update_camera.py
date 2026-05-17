import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'classroom_iot.settings')
django.setup()

from entrance_cam.models import Camera

# Update all cameras to use webcam index 0
for camera in Camera.objects.all():
    camera.url = '0'  # Use webcam index 0 instead of IP
    camera.save()
    print(f"Updated camera {camera.id} ({camera.name}) to use webcam index 0")

print("All cameras updated!")
