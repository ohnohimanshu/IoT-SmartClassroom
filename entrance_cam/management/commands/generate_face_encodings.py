from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from entrance_cam.models import Student
import cv2
import json
import base64
from io import BytesIO

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    print("[ERROR] face_recognition library not installed. Run: pip install face-recognition")
    FACE_RECOGNITION_AVAILABLE = False


class Command(BaseCommand):
    help = 'Generate face encodings from student photos for facial recognition'

    def handle(self, *args, **options):
        if not FACE_RECOGNITION_AVAILABLE:
            self.stdout.write(self.style.ERROR('face_recognition library not installed!'))
            self.stdout.write('Install it with: pip install face-recognition')
            return

        students = Student.objects.filter(photo__isnull=False).exclude(photo='')
        self.stdout.write(f"Processing {students.count()} students...")

        for idx, student in enumerate(students, 1):
            try:
                # Read image from Django media
                image_path = student.photo.path
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)

                if not face_encodings:
                    self.stdout.write(self.style.WARNING(f"[{idx}] {student.name} - No face detected"))
                    continue

                # Store first encoding as JSON
                encoding = face_encodings[0]
                student.face_encoding = json.dumps(encoding.tolist())
                student.save(update_fields=['face_encoding'])
                self.stdout.write(self.style.SUCCESS(f"[{idx}] {student.name} - ✓ Encoding generated"))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"[{idx}] {student.name} - Error: {str(e)}"))

        self.stdout.write(self.style.SUCCESS('\n✓ Face encoding generation complete!'))
