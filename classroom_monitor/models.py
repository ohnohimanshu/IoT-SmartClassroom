from django.db import models
from django.utils import timezone


class ClassroomCamera(models.Model):
    STREAM_TYPE_CHOICES = [
        ('mjpeg', 'MJPEG'),
        ('rtsp', 'RTSP'),
        ('snapshot', 'Snapshot'),
    ]

    name = models.CharField(max_length=100)
    url = models.CharField(max_length=300, help_text="RTSP or HTTP stream URL")
    location = models.CharField(max_length=100, default='Classroom')
    is_active = models.BooleanField(default=True)
    stream_type = models.CharField(
        max_length=10,
        choices=STREAM_TYPE_CHOICES,
        default='mjpeg'
    )
    snapshot_url = models.CharField(
        max_length=300, blank=True,
        help_text="Direct snapshot URL if different from stream URL"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} @ {self.location}"


class ClassSession(models.Model):
    camera = models.ForeignKey(ClassroomCamera, on_delete=models.CASCADE)
    subject = models.CharField(max_length=100, blank=True)
    teacher = models.CharField(max_length=100, blank=True)
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    total_students_detected = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.subject} - {self.camera.name} ({self.start_time.strftime('%Y-%m-%d %H:%M')})"


class EngagementSnapshot(models.Model):
    session = models.ForeignKey(ClassSession, on_delete=models.CASCADE, related_name='snapshots')
    timestamp = models.DateTimeField(auto_now_add=True)
    frame_image = models.ImageField(upload_to='classroom/frames/', blank=True, null=True)
    focused_count = models.IntegerField(default=0)
    looking_away_count = models.IntegerField(default=0)
    head_down_count = models.IntegerField(default=0)
    not_visible_count = models.IntegerField(default=0)
    talking_count = models.IntegerField(default=0)
    total_detected = models.IntegerField(default=0)
    engagement_score = models.FloatField(default=0.0)

    def __str__(self):
        return f"Snapshot {self.pk} - {self.session.subject} - {self.timestamp}"


class StudentZoneLog(models.Model):
    POSE_CHOICES = [
        ('focused', 'Focused'),
        ('looking_away', 'Looking Away'),
        ('head_down', 'Head Down'),
        ('not_visible', 'Not Visible'),
    ]

    snapshot = models.ForeignKey(EngagementSnapshot, on_delete=models.CASCADE, related_name='zone_logs')
    zone_id = models.IntegerField()
    pose = models.CharField(max_length=20, choices=POSE_CHOICES, default='not_visible')
    possibly_talking = models.BooleanField(default=False)
    confidence = models.FloatField(default=0.0)

    def __str__(self):
        return f"Zone {self.zone_id} - {self.pose} ({self.snapshot.pk})"
