"""
Management command to reload camera URLs and restart detection processes.

This is useful when you update a camera URL in the database and want to 
apply it immediately without restarting the Django server.

Usage:
    python manage.py reload_camera_urls              # Reload all cameras
    python manage.py reload_camera_urls --camera-id 8 # Reload specific camera
"""

import os
import sys
import time
import signal
import subprocess
from django.core.management.base import BaseCommand
from django.apps import apps
from entrance_cam.models import Camera


class Command(BaseCommand):
    help = 'Reload camera URLs and restart detection processes'

    def add_arguments(self, parser):
        parser.add_argument(
            '--camera-id',
            type=int,
            help='Reload specific camera by ID (optional)'
        )
        parser.add_argument(
            '--wait',
            type=int,
            default=5,
            help='Seconds to wait before restarting (default: 5)'
        )

    def handle(self, *args, **options):
        camera_id = options.get('camera_id')
        wait_seconds = options.get('wait')

        # Get the app config
        app_config = apps.get_app_config('entrance_cam')

        # Get cameras to reload
        if camera_id:
            try:
                cameras = [Camera.objects.get(pk=camera_id, is_active=True)]
            except Camera.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'Camera {camera_id} not found or inactive')
                )
                return
        else:
            cameras = list(Camera.objects.filter(is_active=True))

        if not cameras:
            self.stdout.write(self.style.WARNING('No active cameras found'))
            return

        self.stdout.write(
            self.style.SUCCESS(f'\n🔄 Reloading {len(cameras)} camera(s)...\n')
        )

        # Stop affected detection processes
        stopped_count = 0
        for camera in cameras:
            stopped = self._stop_camera_process(app_config, camera)
            if stopped:
                stopped_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'✓ Stopped detection for {camera.name} (ID: {camera.id})')
                )
            else:
                self.stdout.write(
                    self.style.WARNING(f'⊘ No process running for {camera.name} (ID: {camera.id})')
                )

        # Wait before restarting
        if stopped_count > 0:
            self.stdout.write(f'\n⏳ Waiting {wait_seconds} seconds before restart...')
            time.sleep(wait_seconds)

            # Restart detection scripts
            self.stdout.write('\n🚀 Restarting detection scripts...\n')
            script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'detection_script.py')
            server_url = app_config._get_server_url()

            restarted_count = 0
            for camera in cameras:
                try:
                    app_config._start_camera_detection(camera, script_path, server_url)
                    restarted_count += 1
                    self.stdout.write(
                        self.style.SUCCESS(f'✓ Restarted detection for {camera.name}')
                    )
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f'✗ Failed to restart {camera.name}: {e}')
                    )

            self.stdout.write(
                self.style.SUCCESS(f'\n✓ Reloaded {restarted_count}/{len(cameras)} cameras\n')
            )
        else:
            self.stdout.write(
                self.style.WARNING('\n⚠ No processes to restart\n')
            )

    def _stop_camera_process(self, app_config, camera):
        """
        Stop the detection process for a specific camera.
        
        Looks through detection_processes list and terminates matching process.
        """
        from entrance_cam.apps import detection_processes
        
        for item in detection_processes[:]:
            if item['camera'].id == camera.id:
                process = item['process']
                try:
                    if sys.platform == 'win32':
                        # Windows: use CTRL_BREAK_EVENT to terminate process group
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        # Unix: use SIGTERM
                        process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        # Force kill if doesn't exit gracefully
                        process.kill()
                        process.wait()
                    
                    detection_processes.remove(item)
                    return True
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f'  Error stopping process: {e}')
                    )
                    return False
        
        return False
