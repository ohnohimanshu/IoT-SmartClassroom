"""
Background detection service that runs independently
"""
import threading
import time
import subprocess
import sys
import os
from django.conf import settings

class DetectionService:
    """Manages camera detection processes in background."""
    
    def __init__(self):
        self.processes = {}
        self.running = False
        self.thread = None

    def start(self):
        """Start the detection service."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("[OK] Detection Service started in background")

    def stop(self):
        """Stop the detection service."""
        self.running = False
        for proc in self.processes.values():
            if proc and proc.poll() is None:
                proc.terminate()
        print("[OK] Detection Service stopped")

    def _run(self):
        """Main loop that monitors cameras and starts detection."""
        from entrance_cam.models import Camera
        import django
        
        # Wait for Django to fully initialize
        time.sleep(2)
        
        while self.running:
            try:
                # Get all active cameras
                cameras = Camera.objects.filter(is_active=True)
                
                for camera in cameras:
                    if camera.id not in self.processes or self._process_dead(camera.id):
                        self._start_detection(camera)
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"[ERROR] Detection service error: {e}")
                time.sleep(10)

    def _process_dead(self, camera_id):
        """Check if a process is still running."""
        if camera_id not in self.processes:
            return True
        proc = self.processes[camera_id]
        return proc.poll() is not None

    def _start_detection(self, camera):
        """Start detection for a camera."""
        try:
            # Get Django base directory
            base_dir = settings.BASE_DIR
            script_path = os.path.join(base_dir, 'entrance_cam', 'detection_script.py')
            
            # Get server URL from environment or default to your server
            server_url = os.getenv('DJANGO_SERVER_URL', 'https://192.168.1.5:8000')
            
            # Build command
            cmd = [
                sys.executable,
                script_path,
                '--camera-id', str(camera.id),
                '--camera-url', camera.url,
                '--server', server_url,
                '--cooldown', '30'
            ]
            
            # Start process with unbuffered output
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                cwd=str(base_dir),
                bufsize=1,  # Line buffered
                universal_newlines=True  # Text mode
            )
            
            self.processes[camera.id] = proc
            print(f"[OK] Detection started for Camera {camera.id}: {camera.name}")
            
            # Monitor output in background
            threading.Thread(
                target=self._monitor_process,
                args=(camera.id, proc),
                daemon=True
            ).start()
            
        except Exception as e:
            print(f"[ERROR] Failed to start detection for {camera.name}: {e}")

    def _monitor_process(self, camera_id, proc):
        """Monitor process output and errors in real-time."""
        print(f"[Monitor] Started monitoring Camera {camera_id} process")
        
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    # Process ended
                    exit_code = proc.poll()
                    print(f"[Camera {camera_id}] Process exited with code: {exit_code}")
                    break
                
                line = line.strip()
                if line:
                    print(f"[Camera {camera_id}] {line}")
        except Exception as e:
            print(f"[ERROR] Monitor for Camera {camera_id} failed: {e}")

# Global service instance
detection_service = DetectionService()
