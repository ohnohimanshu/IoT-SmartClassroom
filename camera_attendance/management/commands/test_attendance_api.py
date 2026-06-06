"""
Test script to verify camera attendance API works end-to-end.
Usage: python manage.py test_attendance_api
"""
import json
import requests
from django.management.base import BaseCommand
from django.test import Client
from django.utils import timezone
from datetime import date
from entrance_cam.models import Student
from camera_attendance.models import Camera, CameraAttendanceLog


class Command(BaseCommand):
    help = 'Test camera attendance API end-to-end'

    def handle(self, *args, **options):
        self.stdout.write(self.style.WARNING('='*70))
        self.stdout.write(self.style.WARNING('CAMERA ATTENDANCE API TEST'))
        self.stdout.write(self.style.WARNING('='*70))

        # Step 1: Check students
        self.stdout.write('\n✓ Step 1: Checking students with face encodings...')
        students_with_encoding = Student.objects.filter(
            is_active=True
        ).exclude(face_encoding__isnull=True).exclude(face_encoding='')
        
        if not students_with_encoding.exists():
            self.stdout.write(self.style.ERROR('  ✗ NO STUDENTS WITH FACE ENCODINGS FOUND'))
            self.stdout.write('  Run: python manage.py generate_encodings')
            return
        
        student = students_with_encoding.first()
        self.stdout.write(self.style.SUCCESS(f'  ✓ Found student: {student.name} (ID: {student.id})'))

        # Step 2: Check camera
        self.stdout.write('\n✓ Step 2: Checking cameras...')
        camera = Camera.objects.filter(is_active=True).first()
        
        if not camera:
            self.stdout.write(self.style.ERROR('  ✗ NO ACTIVE CAMERA FOUND'))
            self.stdout.write('  Create a camera in admin: /admin/entrance_cam/camera/')
            return
        
        self.stdout.write(self.style.SUCCESS(f'  ✓ Found camera: {camera.name} (ID: {camera.id})'))

        # Step 3: Test API - Entry
        self.stdout.write('\n✓ Step 3: Testing API - Entry log...')
        client = Client()
        
        entry_payload = {
            'student_id': student.id,
            'camera_id': camera.id,
            'emotion': 'happy',
            'score': 0.95,
            'snapshot': None  # No snapshot for testing
        }
        
        response = client.post(
            '/camera-attendance/api/log/',
            data=json.dumps(entry_payload),
            content_type='application/json'
        )
        
        if response.status_code == 200:
            data = response.json()
            status = data.get('status')
            self.stdout.write(self.style.SUCCESS(f'  ✓ API Response: {status}'))
            self.stdout.write(f'     Response: {data}')
        else:
            self.stdout.write(self.style.ERROR(f'  ✗ API Error ({response.status_code}): {response.content}'))
            return

        # Step 4: Check database
        self.stdout.write('\n✓ Step 4: Checking database for entry log...')
        today = date.today()
        entry_logs = CameraAttendanceLog.objects.filter(
            student=student,
            date=today,
            entry_time__isnull=False
        )
        
        if not entry_logs.exists():
            self.stdout.write(self.style.ERROR('  ✗ NO ENTRY LOG FOUND IN DATABASE'))
            return
        
        log = entry_logs.first()
        self.stdout.write(self.style.SUCCESS(f'  ✓ Entry log found:'))
        self.stdout.write(f'     Entry time: {log.entry_time}')
        self.stdout.write(f'     Emotion: {log.entry_emotion}')
        self.stdout.write(f'     is_present: {log.is_present}')

        # Step 5: Test API - Exit
        self.stdout.write('\n✓ Step 5: Testing API - Exit log...')
        
        exit_payload = {
            'student_id': student.id,
            'camera_id': camera.id,
            'emotion': 'neutral',
            'score': 0.92,
            'snapshot': None
        }
        
        response = client.post(
            '/camera-attendance/api/log/',
            data=json.dumps(exit_payload),
            content_type='application/json'
        )
        
        if response.status_code == 200:
            data = response.json()
            status = data.get('status')
            self.stdout.write(self.style.SUCCESS(f'  ✓ API Response: {status}'))
            self.stdout.write(f'     Response: {data}')
        else:
            self.stdout.write(self.style.ERROR(f'  ✗ API Error ({response.status_code}): {response.content}'))
            return

        # Step 6: Check exit log
        self.stdout.write('\n✓ Step 6: Checking database for exit log...')
        log.refresh_from_db()
        
        if log.exit_time is None:
            self.stdout.write(self.style.ERROR('  ✗ NO EXIT LOG FOUND'))
            return
        
        self.stdout.write(self.style.SUCCESS(f'  ✓ Exit log found:'))
        self.stdout.write(f'     Exit time: {log.exit_time}')
        self.stdout.write(f'     Duration: {log.duration_minutes} minutes')
        self.stdout.write(f'     Mood: {log.mood_comparison}')
        self.stdout.write(f'     is_present: {log.is_present}')

        # Step 7: Summary
        self.stdout.write('\n' + self.style.WARNING('='*70))
        self.stdout.write(self.style.SUCCESS('✓ ALL TESTS PASSED!'))
        self.stdout.write('Camera attendance API is working correctly.')
        self.stdout.write(self.style.WARNING('='*70))
