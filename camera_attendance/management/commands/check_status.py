"""
Diagnostic command to check camera attendance system status.
"""
from django.management.base import BaseCommand
from django.utils import timezone
from entrance_cam.models import Student
from camera_attendance.models import Camera, CameraAttendanceLog


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
            self.stdout.write(self.style.ERROR('  ✗ NO CAMERAS FOUND'))
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
        else:
            for student in students_with_encoding:
                self.stdout.write(f'  ✓ {student.name} (ID: {student.id})')
        
        total_students = Student.objects.filter(is_active=True).count()
        self.stdout.write(f'\nSummary: {len(students_with_encoding)} / {total_students} students ready')
        
        # 3. Check today's logs
        self.stdout.write('\n✓ TODAY\'S CAMERA ATTENDANCE LOGS:')
        from datetime import date
        today_logs = CameraAttendanceLog.objects.filter(date=date.today()).count()
        if not today_logs:
            self.stdout.write(self.style.WARNING('  ⚠ No attendance logged yet'))
        else:
            self.stdout.write(self.style.SUCCESS(f'  ✓ {today_logs} logs recorded'))
        
        self.stdout.write('\n' + self.style.WARNING('=' * 60))
