import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'classroom_iot.settings')
django.setup()

from entrance_cam.models import Camera

# Update camera to use your IP camera URL
for camera in Camera.objects.all():
    camera.url = 'http://192.168.1.2:8080/video'  # Replace with your actual IP camera URL
    camera.save()
    print(f"Updated camera {camera.id} ({camera.name}) to use IP camera URL: {camera.url}")

print("Camera updated!")
