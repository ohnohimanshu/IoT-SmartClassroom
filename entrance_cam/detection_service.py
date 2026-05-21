"""
Background detection service that runs independently.
"""
import threading
import time
import subprocess
import sys
import os
from django.conf import settings


class DetectionService:
    """Manages per-camera detection subprocesses in the background."""

    def __init__(self):
        self.processes: dict = {}   # camera_id -> Popen
        self.running = False
        self.thread = None

    def start(self):
        """Start the detection service manager thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("[OK] Detection Service started in background")

    def stop(self):
        """Terminate all camera subprocesses and stop the manager thread."""
        self.running = False
        for camera_id, proc in list(self.processes.items()):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self.processes.clear()
        print("[OK] Detection Service stopped")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self):
        """Manager loop: keep one subprocess alive per active camera."""
        from entrance_cam.models import Camera

        # Give Django ORM a moment to finish initialising
        time.sleep(2)

        while self.running:
            try:
                active_camera_ids = set(
                    Camera.objects.filter(is_active=True).values_list('id', flat=True)
                )

                # ── Stop processes for cameras that are now inactive ──────────
                for camera_id in list(self.processes.keys()):
                    if camera_id not in active_camera_ids:
                        self._stop_process(camera_id)

                # ── Start / restart processes for active cameras ──────────────
                for camera in Camera.objects.filter(is_active=True, id__in=active_camera_ids):
                    if self._process_dead(camera.id):
                        # Clean up the dead entry before restarting
                        self.processes.pop(camera.id, None)
                        self._start_detection(camera)

                time.sleep(5)

            except Exception as e:
                print(f"[ERROR] Detection service manager error: {e}")
                time.sleep(10)

    def _process_dead(self, camera_id: int) -> bool:
        """Return True if no live process exists for camera_id."""
        proc = self.processes.get(camera_id)
        if proc is None:
            return True
        return proc.poll() is not None   # poll() returns None while running

    def _stop_process(self, camera_id: int):
        """Gracefully terminate the process for a given camera."""
        proc = self.processes.pop(camera_id, None)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"[OK] Stopped detection process for Camera {camera_id}")

    def _start_detection(self, camera):
        """Spawn a new detection subprocess for the given camera."""
        try:
            base_dir = settings.BASE_DIR
            script_path = os.path.join(base_dir, 'entrance_cam', 'detection_script.py')

            # Resolve server URL
            server_url = os.getenv('DJANGO_SERVER_URL')
            if not server_url:
                import socket
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(('8.8.8.8', 80))
                    local_ip = s.getsockname()[0]
                    s.close()
                except Exception:
                    local_ip = '127.0.0.1'
                server_url = f'https://{local_ip}:8000'
                print(f"[INFO] Auto-detected server URL: {server_url}")

            cmd = [
                sys.executable,
                script_path,
                '--camera-id', str(camera.id),
                '--camera-url', camera.url,
                '--server', server_url,
                '--cooldown', '30',
            ]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(base_dir),
                bufsize=1,
                universal_newlines=True,
            )

            self.processes[camera.id] = proc
            print(f"[OK] Detection started for Camera {camera.id}: {camera.name}")

            threading.Thread(
                target=self._monitor_process,
                args=(camera.id, proc),
                daemon=True,
            ).start()

        except Exception as e:
            print(f"[ERROR] Failed to start detection for Camera {camera.id} ({camera.name}): {e}")

    def _monitor_process(self, camera_id: int, proc):
        """Stream subprocess stdout to the Django console."""
        print(f"[Monitor] Watching Camera {camera_id} subprocess (pid {proc.pid})")
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    print(f"[Camera {camera_id}] {line}")
        except Exception as e:
            print(f"[ERROR] Monitor for Camera {camera_id} failed: {e}")
        finally:
            exit_code = proc.wait()
            print(f"[Camera {camera_id}] Process exited with code {exit_code}")


# Global singleton
detection_service = DetectionService()