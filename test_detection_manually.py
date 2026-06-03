"""
Manual test of face detection on Vivo camera.
Run: python camera_attendance/detection_script_v2.py --camera-id 18 --camera-url "https://192.168.1.7:8080/video" --server "https://0.0.0.0:8000"

This will now show:
  [INFO] [Frame XXX] Detected N face(s)
  [INFO] ✓ DETECTED: [Student Name] (distance: X.XXX)
  [DEBUG] ↻ [Student Name] on cooldown (25s remaining)
  [DEBUG] ✗ Face not matched (best distance > 0.6)

Watch for these messages to debug why detection isn't working.
"""

print(__doc__)
