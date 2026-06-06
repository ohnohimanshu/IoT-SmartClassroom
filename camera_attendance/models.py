"""
Camera-based Attendance System Models
Completely separate from fingerprint enrollment system
"""
from django.db import models
from django.utils import timezone
import datetime
from entrance_cam.models import Student


class Camera(models.Model):
    """
    Camera configuration for attendance detection.

    Fields:
        name: Camera name/identifier
        url: Camera stream URL or webcam index
        location: Physical location of camera
        is_active: Whether camera is active
        created_at: Creation timestamp
    """
    name = models.CharField(max_length=100)
    url = models.CharField(
        max_length=300,
        help_text="Webcam index (0, 1, …) or IP stream URL e.g. http://192.168.1.100:8080/video",
    )
    location = models.CharField(max_length=100, default='Entrance')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['name']
        verbose_name = 'Camera'
        verbose_name_plural = 'Cameras'

    def __str__(self):
        """Return string representation of the camera."""
        return f"{self.name} @ {self.location}"


class CameraAttendanceLog(models.Model):
    """
    Records attendance detected via camera face recognition.
    Completely independent from fingerprint system.
    
    Fields:
        student: ForeignKey to Student
        camera: ForeignKey to Camera
        date: Date of attendance
        entry_time: Time of entry
        exit_time: Time of exit
        entry_emotion: Emotion detected at entry
        exit_emotion: Emotion detected at exit
        mood_comparison: Comparison of entry vs exit emotion
        duration_minutes: Time spent in minutes
    """
    
    student = models.ForeignKey(
        Student,
        on_delete=models.CASCADE,
        related_name='camera_attendance_logs'
    )
    camera = models.ForeignKey(
        Camera,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='camera_attendance_logs'
    )
    
    # Date and time
    date = models.DateField(default=timezone.now)
    entry_time = models.DateTimeField(null=True, blank=True)
    exit_time = models.DateTimeField(null=True, blank=True)
    
    # Emotion detection
    entry_emotion = models.CharField(
        max_length=20,
        choices=[
            ('happy', 'Happy'),
            ('sad', 'Sad'),
            ('angry', 'Angry'),
            ('neutral', 'Neutral'),
            ('surprise', 'Surprise'),
            ('fear', 'Fear'),
            ('disgust', 'Disgust'),
            ('unknown', 'Unknown'),
        ],
        default='unknown'
    )
    exit_emotion = models.CharField(
        max_length=20,
        choices=[
            ('happy', 'Happy'),
            ('sad', 'Sad'),
            ('angry', 'Angry'),
            ('neutral', 'Neutral'),
            ('surprise', 'Surprise'),
            ('fear', 'Fear'),
            ('disgust', 'Disgust'),
            ('unknown', 'Unknown'),
        ],
        default='unknown'
    )
    
    entry_emotion_score = models.FloatField(default=0.0)
    exit_emotion_score = models.FloatField(default=0.0)
    
    # Snapshots
    entry_snapshot = models.ImageField(upload_to='camera/entry/', blank=True, null=True)
    exit_snapshot = models.ImageField(upload_to='camera/exit/', blank=True, null=True)
    
    # Mood comparison
    mood_comparison = models.CharField(
        max_length=20,
        choices=[
            ('improved', 'Improved'),
            ('declined', 'Declined'),
            ('stable', 'Stable'),
            ('unknown', 'Unknown'),
        ],
        default='unknown',
        blank=True,
    )
    
    # Duration
    duration_minutes = models.IntegerField(null=True, blank=True)
    is_present = models.BooleanField(default=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-date', '-entry_time']
        indexes = [
            models.Index(fields=['student', 'date']),
            models.Index(fields=['date', '-entry_time']),
        ]
        verbose_name = 'Camera Attendance Log'
        verbose_name_plural = 'Camera Attendance Logs'

    def __str__(self):
        """Return string representation of the attendance log."""
        return f"{self.student.name} — {self.date} ({self.get_entry_status()})"
    
    def get_entry_status(self):
        """Return entry/exit status."""
        if self.entry_time and self.exit_time:
            return f"Entry: {self.entry_time.strftime('%H:%M')} → Exit: {self.exit_time.strftime('%H:%M')}"
        elif self.entry_time:
            return f"Entry: {self.entry_time.strftime('%H:%M')} (Inside)"
        else:
            return "No entry recorded"
    
    def calculate_duration(self):
        """Calculate duration in memory without saving."""
        if self.entry_time and self.exit_time:
            delta = self.exit_time - self.entry_time
            self.duration_minutes = int(delta.total_seconds() / 60)
            self.save(update_fields=['duration_minutes'])
