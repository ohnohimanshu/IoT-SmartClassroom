"""
Combined server + detection runner that handles multiple cameras.
Usage: python manage.py runall

Features:
- Automatically detects all active cameras from database
- Starts detection script for each camera
- Handles 1 camera now, scales to 3-4 cameras automatically
- Cleans up all processes on exit
"""
import subprocess
import sys
import time
import os
import signal
from django.core.management.base import BaseCommand
from django.core.management import call_command
from entrance_cam.models import Camera


class Command(BaseCommand):
    help = 'Run SSL server with detection scripts for all active cameras'

    def add_arguments(self, parser):
        parser.add_argument(
            '--camera-ids',
            help='Comma-separated camera IDs (default: all active cameras)'
        )
        parser.add_argument(
            '--addrport', default='0.0.0.0:8000',
            help='Address and port for SSL server'
        )
        parser.add_argument(
            '--server-only',
            action='store_true',
            help='Run only server without detection scripts'
        )
        parser.add_argument(
            '--rebroadcast-base-port',
            type=int,
            default=8765,
            help='Base port for MJPEG rebroadcast (default: 8765)'
        )

    def handle(self, *args, **options):
        camera_ids = options['camera_ids']
        addrport = options['addrport']
        server_only = options['server_only']
        rebroadcast_base_port = options['rebroadcast_base_port']

        # Track all detection processes
        detection_processes = []

        if not server_only:
            # Get cameras to run detection for
            if camera_ids:
                camera_id_list = [int(cid.strip()) for cid in camera_ids.split(',')]
                cameras = Camera.objects.filter(id__in=camera_id_list, is_active=True)
            else:
                # Get all active cameras
                cameras = Camera.objects.filter(is_active=True)

            if not cameras:
                self.stdout.write(self.style.WARNING(
                    'No active cameras found in database!'
                ))
                self.stdout.write(self.style.WARNING(
                    'Add cameras in Django admin or use --camera-ids option'
                ))
            else:
                self.stdout.write(self.style.SUCCESS(
                    f'Found {cameras.count()} active camera(s):'
                ))
                for cam in cameras:
                    self.stdout.write(f'  - {cam.name} (ID: {cam.id}): {cam.url}')

            # Start detection script for each camera
            for idx, camera in enumerate(cameras):
                # Assign unique rebroadcast port for each camera
                # Camera 1 → 8765, Camera 2 → 8766, Camera 3 → 8767, etc.
                rebroadcast_port = rebroadcast_base_port + idx

                server_url = f"https://{addrport}" if not addrport.startswith('http') else addrport

                self.stdout.write(self.style.NOTICE(
                    f'\n[{idx+1}/{len(cameras)}] Configuring Camera {camera.id} ({camera.name})'
                ))
                self.stdout.write(f'  Camera URL: {camera.url}')
                self.stdout.write(f'  Rebroadcast Port: {rebroadcast_port}')
                self.stdout.write(f'  Server URL: {server_url}')

                cmd = [
                    sys.executable,
                    'camera_attendance/detection_script_v2.py',
                    '--camera-id', str(camera.id),
                    '--camera-url', camera.url,
                    '--server', server_url,
                    '--rebroadcast-port', str(rebroadcast_port),
                    '--cooldown', '60',
                    '--exit-timeout', '90',
                    '--confirm-frames', '3',
                ]

                self.stdout.write(self.style.SUCCESS(
                    f'\nStarting detection for Camera {camera.id} ({camera.name})...'
                ))
                self.stdout.write(f'Command: {" ".join(cmd)}')
                self.stdout.write(f'Rebroadcast port: {rebroadcast_port}')

                try:
                    # Start detection process
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True
                    )

                    detection_processes.append({
                        'process': proc,
                        'camera': camera,
                        'port': rebroadcast_port
                    })

                    # Wait a moment for startup
                    time.sleep(2)

                    # Check if process is still running
                    if proc.poll() is not None:
                        stdout, _ = proc.communicate()
                        self.stdout.write(self.style.ERROR(
                            f'Detection script for Camera {camera.id} failed to start:\n{stdout}'
                        ))
                    else:
                        self.stdout.write(self.style.SUCCESS(
                            f'Detection running for Camera {camera.id} (PID: {proc.pid}, Port: {rebroadcast_port})'
                        ))

                except Exception as e:
                    self.stdout.write(self.style.ERROR(
                        f'Error starting detection for Camera {camera.id}: {e}'
                    ))

        # Cleanup function
        def cleanup():
            self.stdout.write(self.style.WARNING('\nShutting down detection processes...'))
            for item in detection_processes:
                proc = item['process']
                camera = item['camera']
                if proc.poll() is None:
                    self.stdout.write(f'  Stopping detection for Camera {camera.id} (PID: {proc.pid})...')
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            self.stdout.write(self.style.SUCCESS('All detection processes stopped.'))

        # Register cleanup
        import atexit
        atexit.register(cleanup)

        # Handle signals
        def signal_handler(signum, frame):
            self.stdout.write(self.style.WARNING('\nReceived signal, shutting down...'))
            cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start SSL server (this blocks)
        self.stdout.write(self.style.SUCCESS(f'\nStarting SSL server on {addrport}...'))
        self.stdout.write(self.style.WARNING('Press Ctrl+C to stop everything\n'))

        try:
            call_command('runsslserver', addrport=addrport)
        finally:
            cleanup()
