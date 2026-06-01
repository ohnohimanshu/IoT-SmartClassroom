"""
Management command to regenerate face encodings for all students with photos.

Usage:
    python manage.py regenerate_face_encodings                # Regenerate all
    python manage.py regenerate_face_encodings --student-id 5 # Single student
    python manage.py regenerate_face_encodings --force        # Force regenerate even if encoding exists
    python manage.py regenerate_face_encodings --verbose      # Show detailed output
"""

import json
import os
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from entrance_cam.models import Student

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False


class Command(BaseCommand):
    help = 'Regenerate face encodings for students. Marks is_enrolled based on success.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--student-id',
            type=int,
            help='Regenerate encoding for a specific student ID (optional)'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force regenerate even if encoding already exists'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed output for each student'
        )

    def handle(self, *args, **options):
        if not FACE_RECOGNITION_AVAILABLE:
            raise CommandError(
                'face_recognition not installed. '
                'Run: pip install face_recognition'
            )

        student_id = options.get('student_id')
        force = options.get('force')
        verbose = options.get('verbose')

        # ── Get students to process ───────────────────────────────────────────
        if student_id:
            try:
                students = [Student.objects.get(pk=student_id)]
            except Student.DoesNotExist:
                raise CommandError(f'Student with ID {student_id} not found')
        else:
            # Get all students with photos
            students = Student.objects.filter(photo__isnull=False).exclude(photo='')

        if not students:
            self.stdout.write(self.style.WARNING('No students with photos found'))
            return

        self.stdout.write(
            self.style.SUCCESS(f'\n🔄 Regenerating encodings for {len(students)} students...\n')
        )

        # ── Process each student ──────────────────────────────────────────────
        success_count = 0
        failed_count = 0
        skipped_count = 0

        for student in students:
            status = self._process_student(student, force, verbose)

            if status == 'success':
                success_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'✓ {student.name} ({student.roll_no})')
                )
            elif status == 'skip':
                skipped_count += 1
                if verbose:
                    self.stdout.write(
                        self.style.WARNING(f'⊘ {student.name} ({student.roll_no}) — already enrolled')
                    )
            elif status == 'failed':
                failed_count += 1
                self.stdout.write(
                    self.style.ERROR(f'✗ {student.name} ({student.roll_no})')
                )

        # ── Summary ───────────────────────────────────────────────────────────
        self.stdout.write('\n' + '=' * 60)
        self.stdout.write(self.style.SUCCESS(f'✓ Success:  {success_count}'))
        if skipped_count > 0:
            self.stdout.write(self.style.WARNING(f'⊘ Skipped:  {skipped_count}'))
        if failed_count > 0:
            self.stdout.write(self.style.ERROR(f'✗ Failed:   {failed_count}'))
        self.stdout.write('=' * 60 + '\n')

        if failed_count > 0:
            self.stdout.write(
                self.style.WARNING(
                    'Tip: Students with failed encodings cannot be detected by cameras.\n'
                    'Make sure their photos are:\n'
                    '  • Clear and well-lit\n'
                    '  • Front-facing\n'
                    '  • With only ONE face per photo\n'
                    '  • At least 50x50 pixels\n'
                )
            )

    def _process_student(self, student, force=False, verbose=False):
        """
        Process a single student's face encoding.
        Returns: 'success' | 'skip' | 'failed'
        """
        try:
            image_path = student.photo.path
            if not os.path.exists(image_path):
                if verbose:
                    self.stdout.write(
                        self.style.ERROR(f'  Photo file missing: {image_path}')
                    )
                Student.objects.filter(pk=student.pk).update(is_enrolled=False)
                return 'failed'

            # ── Check if we should skip ───────────────────────────────────────
            if not force:
                has_encoding = bool(student.face_encoding and student.face_encoding.strip())
                if has_encoding:
                    return 'skip'

            # ── Load and process image ────────────────────────────────────────
            try:
                image = face_recognition.load_image_file(image_path)
            except Exception as e:
                if verbose:
                    self.stdout.write(self.style.ERROR(f'  Failed to load image: {e}'))
                Student.objects.filter(pk=student.pk).update(is_enrolled=False)
                return 'failed'

            # ── Extract face encodings ────────────────────────────────────────
            encodings = face_recognition.face_encodings(image)

            if not encodings:
                if verbose:
                    self.stdout.write(
                        self.style.WARNING('  No face detected in photo')
                    )
                Student.objects.filter(pk=student.pk).update(
                    face_encoding='',
                    is_enrolled=False
                )
                return 'failed'

            if len(encodings) > 1 and verbose:
                self.stdout.write(
                    self.style.WARNING(f'  Multiple faces detected ({len(encodings)}) — using first')
                )

            # ── Save encoding ─────────────────────────────────────────────────
            encoding_json = json.dumps(encodings[0].tolist())
            Student.objects.filter(pk=student.pk).update(
                face_encoding=encoding_json,
                is_enrolled=True
            )
            return 'success'

        except Exception as e:
            if verbose:
                self.stdout.write(self.style.ERROR(f'  Exception: {e}'))
            Student.objects.filter(pk=student.pk).update(is_enrolled=False)
            return 'failed'
