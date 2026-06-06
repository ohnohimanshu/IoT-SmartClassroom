from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import datetime

# For backwards compatibility: import Camera from camera_attendance
try:
    from camera_attendance.models import Camera
except ImportError:
    pass

COURSE_CHOICES = [
    ('B.Tech', 'B.Tech'), ('M.Tech', 'M.Tech'), ('BCA', 'BCA'),
    ('MCA', 'MCA'),       ('B.Sc', 'B.Sc'),     ('M.Sc', 'M.Sc'),
    ('MBA', 'MBA'),        ('Other', 'Other'),
]

YEAR_CHOICES = [
    (1, '1st Year'), (2, '2nd Year'), (3, '3rd Year'),
    (4, '4th Year'), (5, '5th Year'),
]

EMOTION_CHOICES = [
    ('happy',    'Happy'),
    ('sad',      'Sad'),
    ('angry',    'Angry'),
    ('neutral',  'Neutral'),
    ('surprise', 'Surprise'),
    ('fear',     'Fear'),
    ('disgust',  'Disgust'),
    ('unknown',  'Unknown'),
]

MOOD_COMPARISON_CHOICES = [
    ('improved', 'Improved'),
    ('declined', 'Declined'),
    ('stable',   'Stable'),
    ('unknown',  'Unknown'),
]


class Student(models.Model):
    name          = models.CharField(max_length=100)
    roll_no       = models.CharField(max_length=30, unique=True)
    email         = models.EmailField(unique=True)
    course        = models.CharField(max_length=20, choices=COURSE_CHOICES)
    branch        = models.CharField(max_length=100)
    year          = models.IntegerField(choices=YEAR_CHOICES)
    photo         = models.ImageField(upload_to='students/photos/')

    # JSON list of 128 floats produced by face_recognition.
    # Generated automatically by signals.generate_face_encoding_on_photo_upload
    face_encoding = models.TextField(blank=True, null=True)

    # Fingerprint fields
    fingerprint_id = models.IntegerField(unique=True, null=True, blank=True)
    is_enrolled = models.BooleanField(
        default=False,
        help_text="True if face encoding generated successfully (for camera detection)"
    )
    fp_confidence = models.IntegerField(null=True, blank=True)
    fp_scan_count = models.IntegerField(default=0)
    fp_last_seen = models.DateTimeField(null=True, blank=True)
    fp_image = models.ImageField(upload_to='fingerprints/', blank=True, null=True)

    user = models.OneToOneField(
        User, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='student_profile',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    is_active  = models.BooleanField(default=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return f"{self.name} ({self.roll_no})"

    def get_enrollment_status(self):
        """Return human-readable enrollment status for camera detection."""
        if self.is_enrolled and self.face_encoding:
            return '✓ Enrolled (Face recognition ready)'
        elif self.face_encoding and not self.is_enrolled:
            return '⚠ Encoding exists but not verified'
        elif self.photo:
            return '✗ Photo uploaded but no face detected'
        else:
            return '⊘ No photo uploaded'


class ESP32Device(models.Model):
    name = models.CharField(max_length=100)
    ip_address = models.CharField(max_length=100, help_text="ESP32 IP address or server URL")
    location = models.CharField(max_length=100, default='Main Entrance')
    is_active = models.BooleanField(default=True)
    last_seen = models.DateTimeField(null=True, blank=True)
    api_key = models.CharField(max_length=64, unique=True)
    enrollment_mode = models.BooleanField(default=False)
    enrollment_student = models.ForeignKey(
        Student, on_delete=models.SET_NULL, null=True, blank=True, related_name='enrollment_device'
    )
    pending_fingerprint_id = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'ESP32 Device'
        verbose_name_plural = 'ESP32 Devices'

    def __str__(self):
        return f"{self.name} ({self.ip_address})"

    def is_online(self):
        if not self.last_seen:
            return False
        return (timezone.now() - self.last_seen).total_seconds() < 120


class FingerprintAttendance(models.Model):
    ATTENDANCE_TYPE_CHOICES = [
        ('entry', 'Entry'),
        ('exit', 'Exit'),
    ]

    student = models.ForeignKey(
        Student, on_delete=models.CASCADE, related_name='fingerprint_attendance_logs'
    )
    device = models.ForeignKey(
        ESP32Device, on_delete=models.SET_NULL, null=True, blank=True, related_name='attendance_records'
    )
    attendance_type = models.CharField(
        max_length=10, choices=ATTENDANCE_TYPE_CHOICES, default='entry'
    )
    timestamp = models.DateTimeField(default=timezone.now)
    date = models.DateField(default=timezone.now)
    fingerprint_id = models.IntegerField()
    confidence = models.IntegerField(default=0, help_text="Fingerprint match confidence (0-255)")
    duration_minutes = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Fingerprint Attendance'
        verbose_name_plural = 'Fingerprint Attendance Records'
        indexes = [
            models.Index(fields=['student', 'date']),
            models.Index(fields=['timestamp']),
        ]

    def __str__(self):
        return f"{self.student.name} - {self.attendance_type} at {self.timestamp.strftime('%H:%M')}"

    def calculate_duration(self):
        if self.attendance_type != 'exit':
            return None

        entry_record = FingerprintAttendance.objects.filter(
            student=self.student,
            attendance_type='entry',
            timestamp__lt=self.timestamp,
            date=self.date
        ).order_by('-timestamp').first()

        if entry_record:
            delta = self.timestamp - entry_record.timestamp
            return int(delta.total_seconds() // 60)
        return None