"""
ESP32 Fingerprint Attendance System Models
Integrated with entrance_cam module
"""

from django.db import models
from django.utils import timezone
import datetime


class ESP32Device(models.Model):
    """ESP32 Fingerprint Device configuration"""
    name = models.CharField(max_length=100)
    ip_address = models.CharField(max_length=15, help_text="ESP32 IP address")
    location = models.CharField(max_length=100, default='Main Entrance')
    
    # Device status
    is_active = models.BooleanField(default=True)
    last_seen = models.DateTimeField(null=True, blank=True)
    
    # API Key for ESP32 authentication
    api_key = models.CharField(max_length=64, unique=True)
    
    # Enrollment state
    enrollment_mode = models.BooleanField(default=False)
    enrollment_student = models.ForeignKey(
        'Student',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='enrollment_device'
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
        """Check if device is online (seen in last 2 minutes)"""
        if not self.last_seen:
            return False
        return (timezone.now() - self.last_seen).total_seconds() < 120


class FingerprintAttendance(models.Model):
    """Fingerprint-based attendance records"""
    ATTENDANCE_TYPE_CHOICES = [
        ('entry', 'Entry'),
        ('exit', 'Exit'),
    ]
    
    student = models.ForeignKey(
        'Student',
        on_delete=models.CASCADE,
        related_name='fingerprint_attendance_logs'
    )
    device = models.ForeignKey(
        ESP32Device,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='attendance_records'
    )
    
    # Attendance type
    attendance_type = models.CharField(
        max_length=10,
        choices=ATTENDANCE_TYPE_CHOICES,
        default='entry'
    )
    
    # Timestamps
    timestamp = models.DateTimeField(default=timezone.now)
    date = models.DateField(default=datetime.date.today)
    
    # Fingerprint data
    fingerprint_id = models.IntegerField()
    confidence = models.IntegerField(default=0, help_text="Fingerprint match confidence (0-255)")
    
    # Duration (for exit records)
    duration_minutes = models.IntegerField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
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
        """Calculate duration from entry to exit"""
        if self.attendance_type != 'exit':
            return None
        
        # Find the matching entry record
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


class FingerprintImage(models.Model):
    """Stores fingerprint images captured during enrollment"""
    student = models.OneToOneField(
        'Student',
        on_delete=models.CASCADE,
        related_name='fingerprint_image'
    )
    fingerprint_id = models.IntegerField(unique=True)
    
    # Image file
    image = models.ImageField(
        upload_to='fingerprints/',
        help_text="PNG fingerprint image from ESP32"
    )
    
    # Metadata
    captured_at = models.DateTimeField(auto_now_add=True)
    device = models.ForeignKey(
        ESP32Device,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    
    class Meta:
        verbose_name = 'Fingerprint Image'
        verbose_name_plural = 'Fingerprint Images'
    
    def __str__(self):
        return f"Fingerprint {self.fingerprint_id} - {self.student.name}"


class ESP32CommandQueue(models.Model):
    """Queue for commands to be sent to ESP32 devices"""
    COMMAND_CHOICES = [
        ('enroll', 'Enroll Fingerprint'),
        ('delete', 'Delete Fingerprint'),
        ('reset', 'Reset Device'),
    ]
    
    device = models.ForeignKey(
        ESP32Device,
        on_delete=models.CASCADE,
        related_name='command_queue'
    )
    command = models.CharField(max_length=20, choices=COMMAND_CHOICES)
    fingerprint_id = models.IntegerField(null=True, blank=True)
    student = models.ForeignKey(
        'Student',
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    
    # Status
    is_processed = models.BooleanField(default=False)
    processed_at = models.DateTimeField(null=True, blank=True)
    result = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
        verbose_name = 'ESP32 Command'
        verbose_name_plural = 'ESP32 Command Queue'
    
    def __str__(self):
        return f"{self.command} for {self.device.name}"
