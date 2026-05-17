import os
import django
import json
import face_recognition

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'classroom_iot.settings')
django.setup()

from entrance_cam.models import Student

print("Starting manual face encoding generation...")
students = Student.objects.filter(photo__isnull=False).exclude(photo='')

print(f"Found {students.count()} students with photos")

for student in students:
    try:
        print(f"Processing {student.name}...")
        image_path = student.photo.path
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            encoding = encodings[0]
            student.face_encoding = json.dumps(encoding.tolist())
            student.save(update_fields=['face_encoding'])
            print(f"✓ Successfully encoded {student.name}")
        else:
            print(f"✗ No face found for {student.name}")
    except Exception as e:
        print(f"✗ Error processing {student.name}: {e}")

print("\nDone!")
