import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'classroom_iot.settings')
django.setup()

from entrance_cam.models import Student

students = Student.objects.all()
print("Checking all students...")
for student in students:
    has_photo = student.photo is not None and student.photo != ''
    has_encoding = student.face_encoding is not None and student.face_encoding.strip() != ''
    print(f"  - {student.name}: photo={has_photo}, encoding={has_encoding}")
    if has_encoding:
        print(f"    Encoding length: {len(student.face_encoding)}")
