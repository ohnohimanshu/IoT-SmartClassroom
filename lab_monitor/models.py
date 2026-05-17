from django.db import models
from entrance_cam.models import Student


class LabSession(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='lab_sessions')
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    duration_minutes = models.IntegerField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    webrtc_offer = models.JSONField(null=True, blank=True)
    webrtc_answer = models.JSONField(null=True, blank=True)
    webrtc_ice_candidates_student = models.JSONField(default=list, blank=True)
    webrtc_ice_candidates_admin = models.JSONField(default=list, blank=True)

    class Meta:
        ordering = ['-start_time']

    def __str__(self):
        return f"{self.student.name} — {self.start_time.strftime('%Y-%m-%d %H:%M')}"

    def calculate_duration(self):
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            self.duration_minutes = int(delta.total_seconds() / 60)
            self.save(update_fields=['duration_minutes'])


class Screenshot(models.Model):
    session = models.ForeignKey(LabSession, on_delete=models.CASCADE, related_name='screenshots')
    image = models.ImageField(upload_to='lab/screenshots/')
    timestamp = models.DateTimeField(auto_now_add=True)
    tab_title = models.CharField(max_length=300, blank=True)

    class Meta:
        ordering = ['-timestamp']


class CameraSnapshot(models.Model):
    EMOTION_CHOICES = [
        ('happy', 'Happy'), ('sad', 'Sad'), ('angry', 'Angry'),
        ('neutral', 'Neutral'), ('surprise', 'Surprise'),
        ('fear', 'Fear'), ('disgust', 'Disgust'), ('unknown', 'Unknown'),
    ]
    POSE_CHOICES = [
        ('focused', 'Focused'), ('looking_away', 'Looking Away'),
        ('head_down', 'Head Down'), ('unknown', 'Unknown'),
    ]
    session = models.ForeignKey(LabSession, on_delete=models.CASCADE, related_name='camera_snapshots')
    image = models.ImageField(upload_to='lab/camera/')
    timestamp = models.DateTimeField(auto_now_add=True)
    emotion = models.CharField(max_length=20, choices=EMOTION_CHOICES, default='unknown')
    emotion_score = models.FloatField(default=0.0)
    pose = models.CharField(max_length=20, choices=POSE_CHOICES, default='unknown')

    class Meta:
        ordering = ['-timestamp']


class ActivityLog(models.Model):
    ACTIVITY_CHOICES = [
        ('active', 'Active'), ('idle', 'Idle'), ('tab_change', 'Tab Change'),
    ]
    session = models.ForeignKey(LabSession, on_delete=models.CASCADE, related_name='activity_logs')
    timestamp = models.DateTimeField(auto_now_add=True)
    tab_title = models.CharField(max_length=300, blank=True)
    activity_type = models.CharField(max_length=20, choices=ACTIVITY_CHOICES, default='active')

    class Meta:
        ordering = ['-timestamp']
