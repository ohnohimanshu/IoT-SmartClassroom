import os
import subprocess
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.apps import apps
from .models import Camera

@receiver(post_save, sender=Camera)
@receiver(post_delete, sender=Camera)
def update_camera_detection_processes(sender, instance, **kwargs):
    """
    Listen for changes to the Camera model (add/update/delete)
    and dynamically reload the background detection scripts.
    """
    # 1. Docker environment (using supervisord)
    if os.path.exists('/etc/supervisor/conf.d'):
        try:
            print(f"[INFO] Camera '{instance.name}' modified. Updating supervisor configs...")
            # Generate new configs
            generate_script = '/app/generate_supervisor_configs.py'
            if os.path.exists(generate_script):
                subprocess.run(['python', generate_script], check=True)
                # Tell supervisor to load new configs, start new processes, and stop removed ones
                subprocess.run(['supervisorctl', 'update'], check=True)
                print("[INFO] Supervisor successfully updated camera detection processes.")
            else:
                print(f"[WARN] {generate_script} not found. Cannot update supervisor.")
        except Exception as e:
            print(f"[ERROR] Failed to update supervisor configs: {e}")
            
    # 2. Local environment (using runserver)
    else:
        app_config = apps.get_app_config('camera_attendance')
        if hasattr(app_config, 'reload_camera_detection'):
            try:
                app_config.reload_camera_detection(instance)
            except Exception as e:
                print(f"[ERROR] Failed to dynamically reload local camera process: {e}")
