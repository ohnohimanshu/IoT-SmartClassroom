"""
Django management command to generate face encodings for students.
Usage: python manage.py generate_encodings [--all] [--force]
"""
import os
import json
from django.core.management.base import BaseCommand, CommandError
from entrance_cam.models import Student


class Command(BaseCommand):
    help = 'Generate face encodings for students with photos'

    def add_arguments(self, parser):
        parser.add_argument(
            '--all',
            action='store_true',
            help='Generate for all students (not just missing)',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force regenerate even if encoding exists',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.WARNING('='*70))
        self.stdout.write(self.style.WARNING('FACE ENCODING GENERATION'))
        self.stdout.write(self.style.WARNING('='*70))

        try:
            import face_recognition
        except ImportError:
            self.stdout.write(self.style.ERROR(
                '\n✗ face_recognition not installed!'
            ))
            self.stdout.write('Install with: pip install face-recognition\n')
            raise CommandError('face_recognition library required')

        # Determine which students to process
        if options['all']:
            students = Student.objects.filter(is_active=True)
            self.stdout.write('\n📋 Processing ALL active students...')
        else:
            students = Student.objects.filter(
                is_active=True,
                photo__isnull=False
            ).exclude(photo='')
            
            if not options['force']:
                # Exclude those already with encodings
                students = students.filter(
                    face_encoding__isnull=True
                ) | Student.objects.filter(
                    is_active=True,
                    photo__isnull=False,
                    face_encoding=''
                )
                self.stdout.write('\n📋 Processing students WITHOUT encodings...')
            else:
                self.stdout.write('\n📋 Processing students (FORCE regenerate)...')

        total = students.count()
        self.stdout.write(f'\nFound {total} students to process\n')

        # Process each student
        success_count = 0
        error_count = 0
        skip_count = 0

        for idx, student in enumerate(students, 1):
            # Check if student has photo
            if not student.photo:
                self.stdout.write(f'{idx}/{total} ⊘ {student.name} - No photo')
                skip_count += 1
                continue

            # Check if photo file exists
            try:
                image_path = student.photo.path
                if not os.path.exists(image_path):
                    self.stdout.write(
                        self.style.WARNING(
                            f'{idx}/{total} ✗ {student.name} - Photo file missing'
                        )
                    )
                    error_count += 1
                    continue
            except Exception as e:
                self.stdout.write(
                    self.style.WARNING(f'{idx}/{total} ✗ {student.name} - Error: {e}')
                )
                error_count += 1
                continue

            # Generate encoding
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if not encodings:
                    self.stdout.write(
                        self.style.WARNING(
                            f'{idx}/{total} ⚠ {student.name} - No face detected'
                        )
                    )
                    student.face_encoding = ''
                    student.is_enrolled = False
                    student.save(update_fields=['face_encoding', 'is_enrolled'])
                    error_count += 1
                    continue

                if len(encodings) > 1:
                    self.stdout.write(
                        self.style.WARNING(
                            f'{idx}/{total} ⚠ {student.name} - Multiple faces (using first)'
                        )
                    )

                # Save encoding
                encoding_json = json.dumps(encodings[0].tolist())
                student.face_encoding = encoding_json
                student.is_enrolled = True
                student.save(update_fields=['face_encoding', 'is_enrolled'])

                self.stdout.write(
                    self.style.SUCCESS(
                        f'{idx}/{total} ✓ {student.name} - Encoded (enrolled)'
                    )
                )
                success_count += 1

            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'{idx}/{total} ✗ {student.name} - {str(e)[:50]}')
                )
                error_count += 1

        # Summary
        self.stdout.write('\n' + self.style.WARNING('='*70))
        self.stdout.write(self.style.SUCCESS(f'✓ Success: {success_count}'))
        self.stdout.write(self.style.ERROR(f'✗ Failed: {error_count}'))
        self.stdout.write(self.style.WARNING(f'⊘ Skipped: {skip_count}'))
        self.stdout.write(self.style.SUCCESS(f'📊 Total Processed: {success_count + error_count}'))
        self.stdout.write(self.style.WARNING('='*70))

        # Final stats
        total_with_encoding = Student.objects.filter(
            is_active=True
        ).exclude(face_encoding__isnull=True).exclude(face_encoding='').count()
        
        self.stdout.write(
            self.style.SUCCESS(f'\n✓ Total students ready for face detection: {total_with_encoding}')
        )
        self.stdout.write('')
