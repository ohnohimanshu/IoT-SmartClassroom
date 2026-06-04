"""
Custom management command to run SSL server with detection script in background.
Usage: python manage.py runserver_with_detection
"""
import subprocess
import sys
import os
import time
import signal
import atexit
from django.core.management.base import BaseCommand
from django.conf import settings


class Command(BaseCommand):
    help = 'Run SSL server with detection script automatically in background'

    def add_arguments(self, parser):
        parser.add_argument(
            '--addrport', default='0.0.0.0:8000',
            help='Optional port number, or ipaddr:port'
        )
        parser.add_argument(
            '--camera-id', type=int, default=1,
            help='Camera ID to run detection for (default: 1)'
        )
        parser.add_argument(
            '--camera-url',
            help='Camera URL (auto-detected from database if not provided)'
        )
        parser.add_argument(
            '--no-detection',
            action='store_true',
            help='Run server without detection script'
        )

    def handle(self, *args, **options):
        addrport = options['addrport']
        camera_id = options['camera_id']
        camera_url = options['camera_url']
        no_detection = options['no_detection']

        detection_process = None

        if not no_detection:
            # Get camera URL from database if not provided
            if not camera_url:
                try:
                    from entrance_cam.models import Camera
                    camera = Camera.objects.get(id=camera_id, is_active=True)
                    camera_url = camera.url
                    self.stdout.write(self.style.SUCCESS(
                        f'Found camera {camera_id}: {camera.name} @ {camera_url}'
                    ))
                except Exception as e:
                    self.stdout.write(self.style.WARNING(
                        f'Could not get camera from database: {e}'
                    ))
                    camera_url = '0'  # Default to webcam

            # Build detection script command
            server_url = f"http://{addrport}" if not addrport.startswith('http') else f"http://localhost:{addrport.split(':')[-1]}"

            cmd = [
                sys.executable,
                'camera_attendance/detection_script_v2.py',
                '--camera-id', str(camera_id),
                '--camera-url', camera_url,
                '--server', server_url,
                '--cooldown', '60',
                '--exit-timeout', '90',
                '--confirm-frames', '3'
            ]

            self.stdout.write(self.style.SUCCESS(
                f'Starting detection script...'
            ))
            self.stdout.write(f'Command: {" ".join(cmd)}')

            # Start detection process
            try:
                detection_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                # Wait a moment for startup
                time.sleep(2)

                # Check if process is still running
                if detection_process.poll() is not None:
                    stdout, _ = detection_process.communicate()
                    self.stdout.write(self.style.ERROR(
                        f'Detection script failed to start:\n{stdout}'
                    ))
                else:
                    self.stdout.write(self.style.SUCCESS(
                        f'Detection script started successfully (PID: {detection_process.pid})'
                    ))

                    # Register cleanup function
                    def cleanup():
                        if detection_process and detection_process.poll() is None:
                            self.stdout.write(self.style.WARNING(
                                'Stopping detection script...'
                            ))
                            detection_process.terminate()
                            try:
                                detection_process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                detection_process.kill()

                    atexit.register(cleanup)

                    # Handle Ctrl+C
                    def signal_handler(signum, frame):
                        self.stdout.write(self.style.WARNING(
                            '\nReceived interrupt signal, shutting down...'
                        ))
                        cleanup()
                        sys.exit(0)

                    signal.signal(signal.SIGINT, signal_handler)
                    signal.signal(signal.SIGTERM, signal_handler)

            except Exception as e:
                self.stdout.write(self.style.ERROR(
                    f'Failed to start detection script: {e}'
                ))

        # Start SSL server
        self.stdout.write(self.style.SUCCESS(
            f'Starting SSL server on {addrport}...'
        ))

        # Import and run sslserver
        from sslserver.management.commands.runsslserver import Command as SSLCommand
        from django.core.management import call_command

        # Parse addrport
        if ':' in addrport:
            addr, port = addrport.rsplit(':', 1)
            port = int(port)
        else:
            addr = '0.0.0.0'
            port = int(addrport)

        # Call sslserver command
        call_command('runsslserver', addrport=addrport)
