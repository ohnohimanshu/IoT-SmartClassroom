
"""
Camera Attendance App Configuration

Key features:
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


class CameraAttendanceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'camera_attendance'
    verbose_name = 'Camera Attendance System'

    def ready(self):
        """Called when Django app registry is fully populated."""
        # Skip auto-start if we're in RUNALL_MODE (runall.py will handle it)
        if os.environ.get('RUNALL_MODE') == '1':
            print("[INFO] RUNALL_MODE detected - skipping auto-start of detection scripts", flush=True)
            return
            
        # Only start detection when running server (not migrations, shell, etc.)
        if 'runserver' in sys.argv or 'runsslserver' in sys.argv:
            print("[INFO] Django runserver/runsslserver detected", flush=True)
            
            # Prevent double-start from Django's autoreloader.
            # When reloader is active, Django forks:
            #   - Parent process: RUN_MAIN is not set  → this is the watcher, skip
            #   - Child process:  RUN_MAIN == 'true'   → this is the real server, start
            # When --noreload is used, RUN_MAIN is never set, so we start directly.
            is_reloader_child = os.environ.get('RUN_MAIN') == 'true'
            no_reload = '--noreload' in sys.argv
            
            print(f"[DEBUG] RUN_MAIN={os.environ.get('RUN_MAIN')}, no_reload={no_reload}", flush=True)
            
            if is_reloader_child or no_reload:
                print("[INFO] Starting detection script auto-start...", flush=True)
                
                # Import signals to enable dynamic reloading on Camera change
                import camera_attendance.signals
                
                # Register cleanup on exit
                atexit.register(self._cleanup_all_processes)
                
                # Start detection scripts in background thread
                threading.Thread(target=self._delayed_start, daemon=True).start()
            else:
                print("[INFO] Running in reloader parent process (autoreloader watch) - detection will start in child", flush=True)
    
    def _delayed_start(self):
        """Wait a bit for Django to fully initialize, then start detection."""
        # Wait 3 seconds for Django server to start accepting connections
        print("[INFO] Waiting 3 seconds for Django server to fully start...", flush=True)
        time.sleep(3)
        
        # Wait for server to actually be ready
        print("[INFO] Checking if server is ready...", flush=True)
        if not self._wait_for_server():
            print("[ERROR] Django server did not become ready. Detection not started.", flush=True)
            return
        
        print("[INFO] Django server is ready. Starting detection scripts...", flush=True)
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
        
        print(f"[INFO] Checking server at {host}:{port}...", flush=True)
        
        for i in range(max_retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    print(f"[INFO] Server ready at {host}:{port} (attempt {i+1})", flush=True)
                    return True
            except:
                pass
            
            time.sleep(delay)
        
        print(f"[ERROR] Server not ready after {max_retries} attempts", flush=True)
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
                        # Convert 0.0.0.0 to 127.0.0.1 for client connections
                        if host == '0.0.0.0':
                            host = '127.0.0.1'
                        elif host == '::':
                            host = '::1'
                        return f"{'https' if use_ssl else 'http'}://{host}:{port}"
        
        return 'https://127.0.0.1:8000' if use_ssl else 'http://127.0.0.1:8000'
    
    def start_detection_scripts(self):
        """Start detection script subprocesses for all active cameras."""
        from .models import Camera
        
        try:
            active_cameras = list(Camera.objects.filter(is_active=True))
        except Exception as e:
            print(f"[ERROR] Could not query cameras: {e}", flush=True)
            return
        
        if not active_cameras:
            print("[INFO] No active cameras found.", flush=True)
            return
        
        # Use detection_script_v2.py (improved version)
        script_path = os.path.join(os.path.dirname(__file__), 'detection_script_v2.py')
        script_path = os.path.normpath(script_path)
        if not os.path.exists(script_path):
            print(f"[ERROR] Detection script not found: {script_path}", flush=True)
            return
        
        server_url = self._get_server_url()

        for idx, camera in enumerate(active_cameras):
            rebroadcast_port = 8765 + idx
            self._start_camera_detection(camera, script_path, server_url, rebroadcast_port)
    
    def _start_camera_detection(self, camera, script_path, server_url, rebroadcast_port=8765):
        """Start detection for a single camera."""
        try:
            cmd = [
                sys.executable,
                script_path,
                '--camera-url', str(camera.url),
                '--camera-id', str(camera.id),
                '--server', server_url,
                '--rebroadcast-port', str(rebroadcast_port)
            ]
            
            print(f"[INFO] Starting detection for camera '{camera.name}' (ID: {camera.id})", flush=True)
            print(f"[INFO]   Camera URL: {camera.url}", flush=True)
            print(f"[INFO]   Server URL: {server_url}", flush=True)
            
            # Disable Python output buffering for detection scripts
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(os.path.dirname(__file__)),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            detection_processes.append({
                'process': process,
                'camera': camera,
                'start_time': time.time()
            })
            
            print(f"[INFO] Detection started (PID: {process.pid})", flush=True)
            
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
            print(f"[ERROR] Failed to start detection for camera '{camera.name}': {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    def _stream_output(self, pipe, prefix, camera_id):
        """Stream output from subprocess to stdout."""
        try:
            for line in iter(pipe.readline, ''):
                line = line.rstrip()
                if line:
                    print(f"[{prefix}] {line}", flush=True)
                    if "FATAL" in line or "ERROR" in line or "Could not" in line:
                        print(f"[ALERT] Camera {camera_id}: {line}", flush=True)
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
                    print(f"[WARN] Detection for camera '{camera.name}' exited with code {return_code}", flush=True)
                    time.sleep(5)
                    
                    # ✓ FIX: Re-fetch camera from database to get updated URL
                    from .models import Camera as CameraModel
                    try:
                        camera = CameraModel.objects.get(pk=camera.id)
                    except CameraModel.DoesNotExist:
                        print(f"[ERROR] Camera {camera.id} no longer exists in database", flush=True)
                        break
                    
                    server_url = self._get_server_url()
                    script_path = os.path.join(os.path.dirname(__file__), 'detection_script_v2.py')
                    script_path = os.path.normpath(script_path)
                    # Calculate rebroadcast port based on camera index
                    try:
                        from .models import Camera as CameraModel
                        active_cameras = list(CameraModel.objects.filter(is_active=True).order_by('id'))
                        camera_index = active_cameras.index(camera)
                        rebroadcast_port = 8765 + camera_index
                    except ValueError:
                        rebroadcast_port = 8765
                    
                    self._start_camera_detection(camera, script_path, server_url, rebroadcast_port)
                    break
                else:
                    print(f"[INFO] Detection for camera '{camera.name}' exited normally", flush=True)
                    break
                    
            except Exception as e:
                print(f"[ERROR] Error monitoring camera '{camera.name}': {e}", flush=True)
                time.sleep(5)
    
    def _cleanup_all_processes(self):
        """Kill all running detection script processes."""
        print("\n[INFO] Cleaning up detection processes...", flush=True)
        
        for proc_info in detection_processes:
            process = proc_info['process']
            camera = proc_info['camera']
            
            if process.poll() is None:
                try:
                    print(f"[INFO] Stopping detection for camera '{camera.name}' (PID: {process.pid})", flush=True)
                    
                    if sys.platform == 'win32':
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        process.terminate()
                    
                    try:
                        process.wait(timeout=5)
                        print(f"[INFO] Process {process.pid} stopped gracefully", flush=True)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        print(f"[WARN] Process {process.pid} killed", flush=True)
                        
                except Exception as e:
                    print(f"[ERROR] Failed to stop process {process.pid}: {e}", flush=True)

    def reload_camera_detection(self, camera):
        """Called by signals when a camera is created, updated, or deleted."""
        print(f"\n[INFO] Reloading detection process for camera '{camera.name}'...", flush=True)
        import signal
        import subprocess
        
        # 1. Clean up existing process for this camera if any
        for proc_info in list(detection_processes):
            if proc_info['camera'].id == camera.id:
                proc = proc_info['process']
                print(f"[INFO] Stopping existing process {proc.pid} for camera '{camera.name}'", flush=True)
                try:
                    if sys.platform == 'win32':
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        proc.terminate()
                    
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                except Exception as e:
                    print(f"[WARN] Error stopping process: {e}")
                
                detection_processes.remove(proc_info)
        
        # 2. Start new process if the camera is still active
        from .models import Camera
        try:
            cam_db = Camera.objects.get(pk=camera.id)
            if cam_db.is_active:
                script_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'detection_script_v2.py'))
                server_url = self._get_server_url()
                # Calculate rebroadcast port based on camera index
                try:
                    active_cameras = list(Camera.objects.filter(is_active=True).order_by('id'))
                    camera_index = active_cameras.index(cam_db)
                    rebroadcast_port = 8765 + camera_index
                except ValueError:
                    rebroadcast_port = 8765
                print(f"[INFO] Restarting detection for camera '{camera.name}'...", flush=True)
                self._start_camera_detection(cam_db, script_path, server_url, rebroadcast_port)
            else:
                print(f"[INFO] Camera '{camera.name}' is inactive. Process stopped.", flush=True)
        except Camera.DoesNotExist:
            print(f"[INFO] Camera '{camera.name}' was deleted. Process stopped.", flush=True)