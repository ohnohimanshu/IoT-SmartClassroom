"""
Entrance Cam App Configuration

Key fixes applied:
1. Detection scripts start AFTER Django server is fully ready
2. Proper server URL detection (http/https based on runsslserver)
3. Single instance enforcement (prevents multiple starts from reloader)
4. Proper process output handling for debugging
5. Graceful shutdown on exit
"""

from django.apps import AppConfig
import subprocess
import os
import atexit
import sys
import threading
import time
import socket
import signal

# Keep track of running detection script processes
detection_processes = []
_startup_lock = threading.Lock()
_startup_complete = False


class EntranceCamConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'entrance_cam'

    def ready(self):
        """Called when Django app registry is fully populated."""
        # Register post_save signals
        from . import signals  # noqa: F401

        # WAL mode for better SQLite concurrency
        from django.db.backends.signals import connection_created
        
        def _set_wal_mode(sender, connection, **kwargs):
            if connection.vendor == 'sqlite':
                with connection.cursor() as cursor:
                    cursor.execute('PRAGMA journal_mode=WAL;')
                    cursor.execute('PRAGMA synchronous=NORMAL;')
        
        connection_created.connect(_set_wal_mode)

        # Only start detection when running server (not migrations, shell, etc.)
        if 'runserver' in sys.argv or 'runsslserver' in sys.argv:
            print("[INFO] Django runserver/runsslserver detected")
            
            # Prevent double-start from Django's autoreloader.
            # When reloader is active, Django forks:
            #   - Parent process: RUN_MAIN is not set  → this is the watcher, skip
            #   - Child process:  RUN_MAIN == 'true'   → this is the real server, start
            # When --noreload is used, RUN_MAIN is never set, so we start directly.
            is_reloader_child = os.environ.get('RUN_MAIN') == 'true'
            no_reload = '--noreload' in sys.argv
            
            print(f"[DEBUG] RUN_MAIN={os.environ.get('RUN_MAIN')}, no_reload={no_reload}")
            
            if is_reloader_child or no_reload:
                print("[INFO] Starting detection script auto-start...")
                # Register cleanup on exit
                atexit.register(self._cleanup_all_processes)
                
                # Start detection scripts in background thread
                threading.Thread(target=self._delayed_start, daemon=True).start()
            else:
                print("[INFO] Running in reloader parent process (autoreloader watch) - detection will start in child")
    
    def _delayed_start(self):
        """Wait a bit for Django to fully initialize, then start detection."""
        # Wait 3 seconds for Django server to start accepting connections
        print("[INFO] Waiting 3 seconds for Django server to fully start...")
        time.sleep(3)
        
        # Wait for server to actually be ready
        print("[INFO] Checking if server is ready...")
        if not self._wait_for_server():
            print("[ERROR] Django server did not become ready. Detection not started.")
            return
        
        print("[INFO] Django server is ready. Starting detection scripts...")
        self.start_detection_scripts()
    
    def _wait_for_server(self, max_retries=30, delay=1):
        """Wait for Django server to accept connections."""
        server_url = self._get_server_url()
        
        # Parse host and port
        if '://' in server_url:
            _, _, address = server_url.partition('://')
        else:
            address = server_url
        
        if '/' in address:
            address = address.split('/')[0]
        
        if ':' in address:
            host, port_str = address.rsplit(':', 1)
            try:
                port = int(port_str)
            except ValueError:
                host = address
                port = 8000
        else:
            host = address
            port = 8000
        
        # If binding to 0.0.0.0 or ::, try to connect to localhost instead
        if host == '0.0.0.0':
            host = '127.0.0.1'
        elif host == '::':
            host = '::1'
        
        host = host.strip('[]')
        
        print(f"[INFO] Checking server at {host}:{port}...")
        
        for i in range(max_retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    print(f"[INFO] Server ready at {host}:{port} (attempt {i+1})")
                    return True
            except:
                pass
            
            time.sleep(delay)
        
        print(f"[ERROR] Server not ready after {max_retries} attempts")
        return False
    
    def _get_server_url(self):
        """Determine server URL from command line arguments."""
        use_ssl = 'runsslserver' in sys.argv
        
        # Find host:port in args
        for arg in sys.argv:
            if ':' in arg:
                parts = arg.rsplit(':', 1)
                if len(parts) == 2:
                    host, port = parts
                    if port.isdigit():
                        return f"{'https' if use_ssl else 'http'}://{arg}"
        
        return 'https://127.0.0.1:8000' if use_ssl else 'http://127.0.0.1:8000'
    
    def start_detection_scripts(self):
        """Start detection script subprocesses for all active cameras."""
        from .models import Camera
        
        try:
            active_cameras = list(Camera.objects.filter(is_active=True))
        except Exception as e:
            print(f"[ERROR] Could not query cameras: {e}")
            return
        
        if not active_cameras:
            print("[INFO] No active cameras found.")
            return
        
        script_path = os.path.join(os.path.dirname(__file__), 'detection_script.py')
        if not os.path.exists(script_path):
            print(f"[ERROR] Detection script not found: {script_path}")
            return
        
        server_url = self._get_server_url()
        
        for camera in active_cameras:
            self._start_camera_detection(camera, script_path, server_url)
    
    def _start_camera_detection(self, camera, script_path, server_url):
        """Start detection for a single camera."""
        try:
            cmd = [
                sys.executable,
                script_path,
                '--camera-url', str(camera.url),
                '--camera-id', str(camera.id),
                '--server', server_url,
                '--no-gui'
            ]
            
            print(f"[INFO] Starting detection for camera '{camera.name}' (ID: {camera.id})")
            print(f"[INFO]   Camera URL: {camera.url}")
            print(f"[INFO]   Server URL: {server_url}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(os.path.dirname(__file__)),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0,
                bufsize=1,
                universal_newlines=True
            )
            
            detection_processes.append({
                'process': process,
                'camera': camera,
                'start_time': time.time()
            })
            
            print(f"[INFO] Detection started (PID: {process.pid})")
            
            # Start output streaming threads
            threading.Thread(
                target=self._stream_output,
                args=(process.stdout, f"DETECT-{camera.id}", camera.id),
                daemon=True
            ).start()
            threading.Thread(
                target=self._stream_output,
                args=(process.stderr, f"DETECT-{camera.id}-ERR", camera.id),
                daemon=True
            ).start()
            
            # Start process monitor
            threading.Thread(
                target=self._monitor_process,
                args=(process, camera),
                daemon=True
            ).start()
            
        except Exception as e:
            print(f"[ERROR] Failed to start detection for camera '{camera.name}': {e}")
            import traceback
            traceback.print_exc()
    
    def _stream_output(self, pipe, prefix, camera_id):
        """Stream output from subprocess to stdout."""
        try:
            for line in iter(pipe.readline, ''):
                line = line.rstrip()
                if line:
                    print(f"[{prefix}] {line}")
                    if "FATAL" in line or "ERROR" in line or "Could not" in line:
                        print(f"[ALERT] Camera {camera_id}: {line}")
        except Exception as e:
            pass
        finally:
            pipe.close()
    
    def _monitor_process(self, process, camera):
        """Monitor a detection process and restart if it dies."""
        while True:
            try:
                return_code = process.wait()
                
                if return_code != 0:
                    print(f"[WARN] Detection for camera '{camera.name}' exited with code {return_code}")
                    time.sleep(5)
                    
                    # ✓ FIX: Re-fetch camera from database to get updated URL
                    from .models import Camera as CameraModel
                    try:
                        camera = CameraModel.objects.get(pk=camera.id)
                    except CameraModel.DoesNotExist:
                        print(f"[ERROR] Camera {camera.id} no longer exists in database")
                        break
                    
                    server_url = self._get_server_url()
                    script_path = os.path.join(os.path.dirname(__file__), 'detection_script.py')
                    
                    self._start_camera_detection(camera, script_path, server_url)
                    break
                else:
                    print(f"[INFO] Detection for camera '{camera.name}' exited normally")
                    break
                    
            except Exception as e:
                print(f"[ERROR] Error monitoring camera '{camera.name}': {e}")
                time.sleep(5)
    
    def _cleanup_all_processes(self):
        """Kill all running detection script processes."""
        print("\n[INFO] Cleaning up detection processes...")
        
        for proc_info in detection_processes:
            process = proc_info['process']
            camera = proc_info['camera']
            
            if process.poll() is None:
                try:
                    print(f"[INFO] Stopping detection for camera '{camera.name}' (PID: {process.pid})")
                    
                    if sys.platform == 'win32':
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        process.terminate()
                    
                    try:
                        process.wait(timeout=5)
                        print(f"[INFO] Process {process.pid} stopped gracefully")
                    except subprocess.TimeoutExpired:
                        process.kill()
                        print(f"[WARN] Process {process.pid} killed")
                        
                except Exception as e:
                    print(f"[ERROR] Failed to stop process {process.pid}: {e}")