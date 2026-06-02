"""
Diagnostic command to check camera attendance system status.
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from entrance_cam.models import Student, Camera
from camera_attendance.models import CameraAttendanceLog


class Command(BaseCommand):
    help = 'Check camera attendance system status and diagnose issues'

    def handle(self, *args, **options):
        self.stdout.write(self.style.WARNING('=' * 60))
        self.stdout.write(self.style.WARNING('CAMERA ATTENDANCE SYSTEM DIAGNOSTIC'))
        self.stdout.write(self.style.WARNING('=' * 60))
        
        # 1. Check cameras
        self.stdout.write('\n✓ CAMERAS:')
        cameras = Camera.objects.all()
        if not cameras:
            self.stdout.write(self.style.ERROR('  ✗ NO CAMERAS FOUND - Please add a camera in Django admin'))
        else:
            for cam in cameras:
                status = self.style.SUCCESS('✓ Active') if cam.is_active else self.style.ERROR('✗ Inactive')
                self.stdout.write(f'  {cam.name} (ID: {cam.id}) - {status}')
                self.stdout.write(f'    URL: {cam.url}')
        
        # 2. Check students with face encodings
        self.stdout.write('\n✓ STUDENTS WITH FACE ENCODING:')
        students_with_encoding = Student.objects.exclude(
            face_encoding__isnull=True
        ).exclude(face_encoding='')
        
        if not students_with_encoding:
            self.stdout.write(self.style.ERROR('  ✗ NO STUDENTS WITH FACE ENCODING'))
            self.stdout.write('  TO FIX:')
            self.stdout.write('    1. Go to Django admin → Students')
            self.stdout.write('    2. Upload a photo for each student')
            self.stdout.write('    3. Select all students with photos')
            self.stdout.write('    4. Use action: "🔄 Regenerate Face Encodings for selected students"')
        else:
            for student in students_with_encoding:
                self.stdout.write(f'  ✓ {student.name} (ID: {student.id}) - {len(student.face_encoding) // 100} KB encoding')
        
        total_students = Student.objects.filter(is_active=True).count()
        self.stdout.write(f'\n  Summary: {len(students_with_encoding)} / {total_students} students have encoding')
        
        # 3. Check today's attendance logs
        self.stdout.write('\n✓ TODAY\'S CAMERA ATTENDANCE LOGS:')
        from datetime import date
        today_logs = CameraAttendanceLog.objects.filter(date=date.today()).count()
        if not today_logs:
            self.stdout.write(self.style.WARNING('  ⚠ No attendance logged yet today'))
        else:
            self.stdout.write(self.style.SUCCESS(f'  ✓ {today_logs} logs recorded today'))
        
        # 4. What to do next
        self.stdout.write('\n' + self.style.WARNING('=' * 60))
        self.stdout.write(self.style.WARNING('NEXT STEPS:'))
        self.stdout.write(self.style.WARNING('=' * 60))
        self.stdout.write('\n1. MAKE SURE A CAMERA IS CONFIGURED:')
        self.stdout.write('   Go to Django admin → Cameras → Add Camera')
        self.stdout.write('   URL: Use "0" for webcam, or "http://your-esp32-cam-ip/stream"')
        self.stdout.write('\n2. GENERATE FACE ENCODINGS FOR STUDENTS:')
        self.stdout.write('   a) Upload photos for students')
        self.stdout.write('   b) Use admin action to regenerate encodings')
        self.stdout.write(f'   Currently: {len(students_with_encoding)} / {total_students} students ready')
        self.stdout.write('\n3. RUN THE DETECTION SCRIPT:')
        self.stdout.write('   python manage.py run_detection --server https://192.168.1.9:8000')
        self.stdout.write('\n4. CHECK IF IT\'S WORKING:')
        self.stdout.write('   Run this command again to see if attendance logs appear')
        
        self.stdout.write('\n' + self.style.WARNING('=' * 60))
