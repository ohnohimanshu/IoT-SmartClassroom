"""
Quick diagnostic script to check why face detection isn't working.
Run: python manage.py shell < diagnose_detection.py
"""

import json
from django.utils import timezone
from datetime import date
from entrance_cam.models import Student, Camera
from camera_attendance.models import CameraAttendanceLog

print("\n" + "="*70)
print("FACE DETECTION DIAGNOSTIC")
print("="*70)

# 1. Check if students have encodings
print("\n1. STUDENTS WITH FACE ENCODINGS:")
students_with_enc = Student.objects.filter(
    is_active=True
).exclude(face_encoding__isnull=True).exclude(face_encoding='')
print(f"   Total: {students_with_enc.count()}")

if students_with_enc.count() == 0:
    print("   ✗ NO STUDENTS HAVE ENCODINGS!")
    print("   → Run: python manage.py generate_encodings --all")
else:
    for s in students_with_enc[:3]:
        try:
            enc = json.loads(s.face_encoding)
            print(f"   ✓ {s.name} ({s.id}) - {len(enc)} floats - enrolled: {s.is_enrolled}")
        except:
            print(f"   ✗ {s.name} ({s.id}) - Invalid encoding")

# 2. Check if cameras are configured
print("\n2. CAMERAS:")
cameras = Camera.objects.filter(is_active=True)
print(f"   Total active: {cameras.count()}")
if cameras.count() == 0:
    print("   ✗ NO ACTIVE CAMERAS!")
    print("   → Go to /admin/entrance_cam/camera/ and add one")
else:
    for c in cameras:
        print(f"   ✓ {c.name} (ID: {c.id}) - URL: {c.url}")

# 3. Check today's logs
print("\n3. TODAY'S CAMERA ATTENDANCE LOGS:")
today = date.today()
today_logs = CameraAttendanceLog.objects.filter(date=today)
print(f"   Total logs: {today_logs.count()}")
print(f"   Entries: {today_logs.filter(entry_time__isnull=False).count()}")
print(f"   Exits: {today_logs.filter(exit_time__isnull=False).count()}")

if today_logs.count() > 0:
    print("   Recent logs:")
    for log in today_logs.order_by('-entry_time')[:5]:
        status = "INSIDE" if log.exit_time is None else f"EXIT {log.exit_time.strftime('%H:%M')}"
        print(f"   • {log.student.name} - Entry: {log.entry_time.strftime('%H:%M')} → {status} - is_present: {log.is_present}")
else:
    print("   ✗ NO LOGS TODAY - Detection hasn't logged anything yet!")

# 4. Check API endpoint
print("\n4. API ENDPOINT CHECK:")
try:
    import requests
    from django.conf import settings
    
    # Get base URL
    base_url = "http://127.0.0.1:8000"
    
    # Test API
    resp = requests.get(
        f"{base_url}/camera-attendance/api/students/encodings/",
        timeout=5,
        verify=False
    )
    
    if resp.status_code == 200:
        data = resp.json()
        print(f"   ✓ API working - {len(data)} students returned")
        if len(data) == 0:
            print("   ✗ API returns EMPTY list!")
            print("   → Check: Do students have is_enrolled=True?")
    else:
        print(f"   ✗ API returned {resp.status_code}")
except Exception as e:
    print(f"   ✗ API error: {e}")

# 5. Check if detection is actually running
print("\n5. DETECTION SCRIPT STATUS:")
print("   Check console for:")
print("   • [INFO] Loaded X students with encodings")
print("   • [INFO] ✓ Detection running")
print("   • [INFO] ✓ [ENTRY_LOGGED] or [EXIT_LOGGED]")
print("   If not seeing these, detection script may have crashed")

print("\n" + "="*70)
print("WHAT TO CHECK:")
print("="*70)
print("""
If NO LOGS today:
  1. Do students have face encodings?
     → Run: python manage.py generate_encodings --all
  
  2. Is detection script actually running?
     → Check console for detection logs
  
  3. Is camera accessible?
     → Check camera IP/URL connectivity
     → Try: curl http://192.168.1.7:8080/video
  
  4. Is there cooldown blocking logs?
     → Default: 30 seconds between same student
     → Try different students

If YES LOGS but seeing errors:
  → Check specific error message
  → Fix accordingly
""")

print("="*70 + "\n")
