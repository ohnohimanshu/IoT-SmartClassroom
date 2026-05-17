#!/usr/bin/env python
"""
Diagnosis script to check if face detection setup is correct
Run: python diagnosis.py
"""

import os
import sys
import django
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'classroom_iot.settings')
sys.path.insert(0, str(Path(__file__).parent))
django.setup()

from entrance_cam.models import Student, Camera
import json

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("\n" + "="*60)
print("CLASSROOM IOT FACE DETECTION DIAGNOSIS")
print("="*60)

# Check 1: Students
print("\n[1] CHECKING STUDENTS")
students = Student.objects.filter(is_active=True)
print(f"    Total active students: {len(students)}")

students_with_photos = students.filter(photo__isnull=False).exclude(photo='')
print(f"    Students with photos: {len(students_with_photos)}")

students_with_encodings = students.filter(face_encoding__isnull=False).exclude(face_encoding='')
print(f"    Students with face encodings: {len(students_with_encodings)}")

if len(students_with_photos) > 0 and len(students_with_encodings) == 0:
    print("\n    WARNING: Photos uploaded but no face encodings generated!")
    print("       This is why faces aren't being detected!")
    print("\n    FIX: Run the face encoding generation:")
    print("       python manage.py generate_face_encodings")

# Check 2: Face encodings quality
print("\n[2] CHECKING FACE ENCODING QUALITY")
if students_with_encodings:
    for student in students_with_encodings[:3]:
        try:
            enc = json.loads(student.face_encoding)
            print(f"    {student.name}: encoding length = {len(enc)} (should be 128)")
            if len(enc) != 128:
                print(f"      Unexpected encoding length!")
        except Exception as e:
            print(f"    {student.name}: Invalid encoding - {e}")
else:
    print("    No students with face encodings!")

# Check 3: Cameras
print("\n[3] CHECKING CAMERAS")
cameras = Camera.objects.filter(is_active=True)
print(f"    Total active cameras: {len(cameras)}")

for camera in cameras:
    print(f"    - {camera.name}: {camera.url}")
    
    # Try to test camera
    if camera.url == '0' or camera.url.isdigit():
        print(f"      (Webcam index: {camera.url})")
    elif camera.url.startswith('http'):
        import urllib.request
        try:
            req = urllib.request.urlopen(camera.url, timeout=2)
            print(f"      Camera reachable (HTTP {req.getcode()})")
        except Exception as e:
            print(f"      Camera NOT reachable: {e}")
    else:
        print(f"      Invalid camera URL format")

# Check 4: Face recognition library
print("\n[4] CHECKING LIBRARIES")
try:
    import face_recognition
    print(f"    face_recognition: installed")
except ImportError:
    print(f"    face_recognition: NOT installed - pip install face-recognition")

try:
    import cv2
    print(f"    OpenCV: installed")
except ImportError:
    print(f"    OpenCV: NOT installed")

try:
    from deepface import DeepFace
    print(f"    DeepFace: installed")
except ImportError:
    print(f"    DeepFace: NOT installed - pip install deepface")

try:
    import requests
    print(f"    requests: installed")
except ImportError:
    print(f"    requests: NOT installed")

# Summary
print("\n" + "="*60)
print("SUMMARY & NEXT STEPS")
print("="*60)

if len(students_with_photos) > len(students_with_encodings):
    print("\nISSUE FOUND: No face encodings!")
    print("\n   ROOT CAUSE: Photos uploaded but encodings not generated")
    print("\n   SOLUTION:")
    print("      1. Activate virtual environment")
    print("      2. Run: python manage.py generate_face_encodings")
    print("      3. Restart Django")

elif len(students_with_encodings) == 0:
    print("\nISSUE FOUND: No students with photos and encodings!")
    print("\n   ROOT CAUSE: Need to add students and upload photos")
    print("\n   SOLUTION:")
    print("      1. Go to http://127.0.0.1:8000/admin/")
    print("      2. Add students with clear face photos")
    print("      3. Run: python manage.py generate_face_encodings")
    print("      4. Restart Django")

elif len(cameras) == 0:
    print("\nNo cameras configured!")
    print("\n   NEXT STEP: Add cameras in admin panel")

else:
    print("\nSetup looks correct!")
    print("\n   Everything configured properly")
    print("   Face detection should work")
    print("   Check Django terminal for detection logs")

print("\n" + "="*60 + "\n")
