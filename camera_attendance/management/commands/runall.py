"""
Combined server + detection runner that handles multiple cameras.
Usage: python manage.py runall

Features:
- Automatically detects all active cameras from database
- Starts detection script for each camera
- Handles 1 camera now, scales to 3-4 cameras automatically
- Cleans up all processes on exit
- Uses --noreload by default, avoids duplicate detection processes
- Streams detection script output to console
"""
import subprocess
import sys
import time
import os
import signal
import threading
from django.core.management.base import BaseCommand
from django.core.management import call_command
from camera_attendance.models import Camera

# Disable Python output buffering for the entire process
os.environ['PYTHONUNBUFFERED'] = '1'


def stream_output(pipe, prefix):
    """Stream output from a pipe to stdout with a prefix, flushing each line."""
    try:
        for line in iter(pipe.readline, ''):
            line = line.rstrip()
            if line:
                print(f"[{prefix}] {line}", flush=True)
    except Exception:
        pass
    finally:
        pipe.close()


class Command(BaseCommand):
    help = 'Run server with detection scripts for all active cameras'

    def add_arguments(self, parser):
        parser.add_argument(
            '--camera-ids',
            help='Comma-separated camera IDs (default: all active cameras)'
        )
        parser.add_argument(
            '--addrport', default='0.0.0.0:8000',
            help='Address and port for server'
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
        parser.add_argument(
            '--ssl',
            action='store_true',
            help='Use SSL server (default: non-SSL)'
        )
        parser.add_argument(
            '--use-reload',
            action='store_true',
            help='Enable Django reloader (risk of duplicate detection scripts)'
        )

    def handle(self, *args, **options):
        # Set RUNALL_MODE to disable auto-start in camera_attendance.apps
        os.environ['RUNALL_MODE'] = '1'
        
        camera_ids = options.get('camera_ids')
        addrport = options.get('addrport', '0.0.0.0:8000')
        server_only = options.get('server_only', False)
        rebroadcast_base_port = options.get('rebroadcast_base_port', 8765)
        use_ssl = options.get('ssl', False)
        use_reload = options.get('use_reload', False)

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
                self.stdout.flush()
                self.stdout.write(self.style.WARNING(
                    'Add cameras in Django admin or use --camera-ids option'
                ))
                self.stdout.flush()
            else:
                self.stdout.write(self.style.SUCCESS(
                    f'Found {cameras.count()} active camera(s):'
                ))
                self.stdout.flush()
                for cam in cameras:
                    self.stdout.write(f'  - {cam.name} (ID: {cam.id}): {cam.url}')
                    self.stdout.flush()

            # Start detection script for each camera
            for idx, camera in enumerate(cameras):
                # Assign unique rebroadcast port for each camera
                # Camera 1 → 8765, Camera 2 → 8766, Camera 3 → 8767, etc.
                rebroadcast_port = rebroadcast_base_port + idx

                # Use correct protocol for API calls
                server_url = "https://localhost:8000" if use_ssl else "http://localhost:8000"

                self.stdout.write(self.style.NOTICE(
                    f'\n[{idx+1}/{len(cameras)}] Configuring Camera {camera.id} ({camera.name})'
                ))
                self.stdout.flush()
                self.stdout.write(f'  Camera URL: {camera.url}')
                self.stdout.flush()
                self.stdout.write(f'  Rebroadcast Port: {rebroadcast_port}')
                self.stdout.flush()
                self.stdout.write(f'  Server URL: {server_url}')
                self.stdout.flush()

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
                self.stdout.flush()
                self.stdout.write(f'Command: {" ".join(cmd)}')
                self.stdout.flush()

                try:
                    # Disable Python output buffering by setting PYTHONUNBUFFERED=1
                    env = os.environ.copy()
                    env['PYTHONUNBUFFERED'] = '1'
                    
                    # Start detection process with captured output for streaming
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=os.getcwd(),
                        universal_newlines=True,
                        bufsize=1,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0,
                        env=env
                    )

                    detection_processes.append({
                        'process': proc,
                        'camera': camera,
                        'port': rebroadcast_port
                    })

                    # Start output streaming threads
                    prefix = f"DETECT-{camera.id}"
                    threading.Thread(
                        target=stream_output,
                        args=(proc.stdout, prefix),
                        daemon=True
                    ).start()
                    threading.Thread(
                        target=stream_output,
                        args=(proc.stderr, f"{prefix}-ERR"),
                        daemon=True
                    ).start()

                    # Wait a moment for startup
                    time.sleep(2)

                    # Check if process is still running
                    if proc.poll() is not None:
                        self.stdout.write(self.style.ERROR(
                            f'Detection script for Camera {camera.id} failed to start - exit code {proc.poll()}'
                        ))
                        self.stdout.flush()
                    else:
                        self.stdout.write(self.style.SUCCESS(
                            f'Detection running for Camera {camera.id} (PID: {proc.pid}, Port: {rebroadcast_port})'
                        ))
                        self.stdout.flush()

                except Exception as e:
                    self.stdout.write(self.style.ERROR(
                        f'Error starting detection for Camera {camera.id}: {e}'
                    ))
                    self.stdout.flush()
                    import traceback
                    traceback.print_exc()

        # Cleanup function
        def cleanup():
            self.stdout.write(self.style.WARNING('\nShutting down detection processes...'))
            self.stdout.flush()
            for item in detection_processes:
                proc = item['process']
                camera = item['camera']
                if proc.poll() is None:
                    self.stdout.write(f'  Stopping detection for Camera {camera.id} (PID: {proc.pid})...')
                    self.stdout.flush()
                    try:
                        if sys.platform == 'win32':
                            proc.send_signal(signal.CTRL_BREAK_EVENT)
                        else:
                            proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                    except Exception as e:
                        self.stdout.write(f'  Error stopping Camera {camera.id}: {e}')
                        self.stdout.flush()
            self.stdout.write(self.style.SUCCESS('All detection processes stopped.'))
            self.stdout.flush()

        # Register cleanup
        import atexit
        atexit.register(cleanup)

        # Handle signals
        def signal_handler(signum, frame):
            self.stdout.write(self.style.WARNING('\nReceived signal, shutting down...'))
            self.stdout.flush()
            cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start server (this blocks)
        server_type = "SSL" if use_ssl else "HTTP"
        self.stdout.write(self.style.SUCCESS(f'\nStarting {server_type} server on {addrport}...'))
        self.stdout.flush()
        self.stdout.write(self.style.WARNING('Press Ctrl+C to stop everything\n'))
        self.stdout.flush()

        try:
            if use_ssl:
                call_command('runsslserver', addrport=addrport, use_reloader=use_reload)
            else:
                call_command('runserver', addrport=addrport, use_reloader=use_reload)
        finally:
            cleanup()
