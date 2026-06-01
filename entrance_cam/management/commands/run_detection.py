import subprocess
import os
import sys
from django.core.management.base import BaseCommand
from entrance_cam.models import Camera


class Command(BaseCommand):
    help = 'Run detection script(s) for active cameras'

    def add_arguments(self, parser):
        parser.add_argument(
            '--camera-id',
            type=int,
            help='Specific camera ID to run detection for (optional)'
        )
        parser.add_argument(
            '--server',
            default='http://localhost:8000',
            help='Django server base URL (default: http://localhost:8000)'
        )
        parser.add_argument(
            '--no-gui',
            action='store_true',
            default=True,
            help='Hide OpenCV GUI windows (default: True)'
        )
        parser.add_argument(
            '--gui',
            dest='no_gui',
            action='store_false',
            help='Show OpenCV GUI windows'
        )

    def handle(self, *args, **options):
        camera_id = options.get('camera_id')
        server_url = options.get('server')
        no_gui = options.get('no_gui')

        # Get active cameras
        if camera_id:
            try:
                cameras = [Camera.objects.get(id=camera_id, is_active=True)]
            except Camera.DoesNotExist:
                self.stdout.write(self.style.ERROR(f'Camera with ID {camera_id} not found or inactive'))
                return
        else:
            cameras = Camera.objects.filter(is_active=True)

        if not cameras:
            self.stdout.write(self.style.WARNING('No active cameras found'))
            return

        # Path to detection_script.py
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'detection_script.py'
        )
        script_path = os.path.normpath(script_path)
        if not os.path.exists(script_path):
            self.stdout.write(self.style.ERROR(f'Detection script not found at {script_path}'))
            return

        processes = []
        try:
            for camera in cameras:
                cmd = [
                    sys.executable,
                    script_path,
                    '--camera-url', str(camera.url),
                    '--camera-id', str(camera.id),
                    '--server', server_url
                ]
                if no_gui:
                    cmd.append('--no-gui')
                self.stdout.write(f'Starting detection for camera: {camera.name} (ID: {camera.id})')
                process = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
                processes.append(process)

            self.stdout.write(self.style.SUCCESS('Detection scripts started. Press Ctrl+C to stop.'))
            for process in processes:
                process.wait()
        except KeyboardInterrupt:
            self.stdout.write('\nStopping detection scripts...')
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
            self.stdout.write(self.style.SUCCESS('Detection scripts stopped.'))