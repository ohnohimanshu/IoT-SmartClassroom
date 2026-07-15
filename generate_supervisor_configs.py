#!/usr/bin/env python3
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'classroom_iot.settings')
sys.path.insert(0, '/app')
django.setup()

from camera_attendance.models import Camera

# Query active cameras from database
try:
    active_cameras = list(Camera.objects.filter(is_active=True).order_by('id'))
except Exception as e:
    print(f"⚠ Could not query cameras from database: {e}")
    print("⚠ Make sure migrations are applied and database is ready.")
    active_cameras = []

if not active_cameras:
    print("ℹ No active cameras found in database.")
    print("ℹ Add cameras in Django admin: http://localhost:8000/admin/camera_attendance/camera/")
    sys.exit(0)

# Generate supervisor config file for each active camera
for idx, camera in enumerate(active_cameras, start=1):
    rebroadcast_port = 8765 + idx - 1
    
    config = f"""[program:detection_camera{camera.id}]
command=python camera_attendance/detection_script_v2.py --camera-url "{camera.url}" --camera-id {camera.id} --server "http://localhost:8000" --cooldown 60 --exit-timeout 90 --confirm-frames 3 --rebroadcast-port {rebroadcast_port}
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/detection_camera{camera.id}.err.log
stdout_logfile=/var/log/supervisor/detection_camera{camera.id}.out.log
priority={998 - idx}
startsecs=15
"""
    
    with open(f'/etc/supervisor/conf.d/detection_camera{camera.id}.conf', 'w') as f:
        f.write(config)
    
    print(f"✓ Generated config for camera {camera.id}: {camera.name} ({camera.url})")

print(f"✓ Total: {len(active_cameras)} camera(s) configured")
