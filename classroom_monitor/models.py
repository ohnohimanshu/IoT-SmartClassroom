from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from entrance_cam.models import Student, Camera


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

    class Meta:
        ordering = ['name']
        verbose_name = 'Classroom Camera'
        verbose_name_plural = 'Classroom Cameras'

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

    class Meta:
        ordering = ['-start_time']
        indexes = [
            models.Index(fields=['camera', '-start_time']),
            models.Index(fields=['is_active', '-start_time']),
        ]
        verbose_name = 'Class Session'
        verbose_name_plural = 'Class Sessions'

    def __str__(self):
        return f"{self.subject} - {self.camera.name} ({self.start_time.strftime('%Y-%m-%d %H:%M')})"


class EngagementSnapshot(models.Model):
    session = models.ForeignKey(ClassSession, on_delete=models.CASCADE, related_name='snapshots')
    timestamp = models.DateTimeField(auto_now_add=True)
    frame_image = models.ImageField(upload_to='classroom/frames/', blank=True, null=True)
    focused_count = models.IntegerField(default=0)
    looking_away_count = models.IntegerField(default=0)
    head_down_count = models.IntegerField(default=0)
    using_phone_count = models.IntegerField(default=0)
    eating_count = models.IntegerField(default=0)
    not_visible_count = models.IntegerField(default=0)
    talking_count = models.IntegerField(default=0)
    total_detected = models.IntegerField(default=0)
    engagement_score = models.FloatField(default=0.0)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['session', '-timestamp']),
            models.Index(fields=['timestamp']),
        ]
        verbose_name = 'Engagement Snapshot'
        verbose_name_plural = 'Engagement Snapshots'

    def __str__(self):
        return f"Snapshot {self.pk} - {self.session.subject} - {self.timestamp}"


class StudentZoneLog(models.Model):
    POSE_CHOICES = [
        ('focused', 'Focused'),
        ('looking_away', 'Looking Away'),
        ('head_down', 'Head Down'),
        ('using_phone', 'Using Phone'),
        ('eating', 'Eating'),
        ('not_visible', 'Not Visible'),
    ]

    snapshot = models.ForeignKey(EngagementSnapshot, on_delete=models.CASCADE, related_name='zone_logs')
    zone_id = models.IntegerField()
    pose = models.CharField(max_length=20, choices=POSE_CHOICES, default='not_visible')
    possibly_talking = models.BooleanField(default=False)
    confidence = models.FloatField(default=0.0)

    class Meta:
        ordering = ['snapshot', 'zone_id']
        indexes = [
            models.Index(fields=['snapshot', 'zone_id']),
        ]
        verbose_name = 'Student Zone Log'
        verbose_name_plural = 'Student Zone Logs'

    def __str__(self):
        return f"Zone {self.zone_id} - {self.pose} ({self.snapshot.pk})"


class ClassroomVideo(models.Model):
    STATUS_CHOICES = [
        ('uploading', 'Uploading'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    title = models.CharField(max_length=200)
    video_file = models.FileField(upload_to='classroom/videos/', help_text="Upload MP4 or AVI video")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='uploading'
    )
    processed_at = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.IntegerField(null=True, blank=True)
    total_frames_analyzed = models.IntegerField(default=0)
    total_students_detected = models.IntegerField(default=0)
    average_engagement_score = models.FloatField(default=0.0)
    notes = models.TextField(blank=True, help_text="Optional notes about this video")

    class Meta:
        ordering = ['-uploaded_at']
        indexes = [
            models.Index(fields=['status', '-uploaded_at']),
        ]
        verbose_name = 'Classroom Video'
        verbose_name_plural = 'Classroom Videos'

    def __str__(self):
        return f"{self.title} ({self.uploaded_at.strftime('%Y-%m-%d')})"

    def get_thumbnail_url(self):
        if self.thumbnail:
            return self.thumbnail.url
        return '/static/classroom_monitor/images/video-placeholder.png'


class VideoAnalysisFrame(models.Model):
    video = models.ForeignKey(ClassroomVideo, on_delete=models.CASCADE, related_name='frames')
    frame_number = models.IntegerField()
    timestamp = models.FloatField()  # seconds from start
    frame_image = models.ImageField(upload_to='classroom/video_frames/', blank=True, null=True)
    focused_count = models.IntegerField(default=0)
    looking_away_count = models.IntegerField(default=0)
    head_down_count = models.IntegerField(default=0)
    using_phone_count = models.IntegerField(default=0)
    eating_count = models.IntegerField(default=0)
    not_visible_count = models.IntegerField(default=0)
    total_detected = models.IntegerField(default=0)
    engagement_score = models.FloatField(default=0.0)

    class Meta:
        ordering = ['frame_number']
        indexes = [
            models.Index(fields=['video', 'frame_number']),
            models.Index(fields=['timestamp']),
        ]
        verbose_name = 'Video Analysis Frame'
        verbose_name_plural = 'Video Analysis Frames'

    def __str__(self):
        return f"Frame {self.frame_number} ({self.timestamp:.1f}s) - {self.video.title}"


class VideoStudentZone(models.Model):
    FRAME_POSE_CHOICES = [
        ('focused', 'Focused'),
        ('looking_away', 'Looking Away'),
        ('head_down', 'Head Down'),
        ('using_phone', 'Using Phone'),
        ('eating', 'Eating'),
        ('not_visible', 'Not Visible'),
    ]

    frame = models.ForeignKey(VideoAnalysisFrame, on_delete=models.CASCADE, related_name='zones')
    zone_id = models.IntegerField()
    pose = models.CharField(max_length=20, choices=FRAME_POSE_CHOICES, default='not_visible')
    possibly_talking = models.BooleanField(default=False)
    confidence = models.FloatField(default=0.0)

    class Meta:
        ordering = ['frame', 'zone_id']
        indexes = [
            models.Index(fields=['frame', 'zone_id']),
        ]
        verbose_name = 'Video Student Zone'
        verbose_name_plural = 'Video Student Zones'

    def __str__(self):
        return f"Zone {self.zone_id} - {self.pose} (Frame {self.frame.frame_number})"


# ── Behavior Incident Model ───────────────────────────────────────────────────

BEHAVIOR_INCIDENT_CHOICES = [
    # RED — alert + WhatsApp (critical)
    ('fighting',     'Physical Fight'),
    ('using_phone',  'Using Phone'),
    ('eating_food',  'Eating Food'),
    # ORANGE — stored only
    ('looking_away', 'Looking Away'),
    ('head_down',    'Head Down'),
    ('distracted',   'Distracted'),
    # fallback
    ('other',        'Other'),
]

SEVERITY_CHOICES = [
    ('low', 'Low'),
    ('medium', 'Medium'),
    ('high', 'High'),
    ('critical', 'Critical'),
]


class IncidentReport(models.Model):
    """
    Records behavior incidents detected in classroom:
    - Phone usage
    - Eating food
    - Distracted behavior
    """
    student = models.ForeignKey(
        Student, on_delete=models.CASCADE, related_name='incident_reports',
        null=True, blank=True
    )
    camera = models.ForeignKey(
        Camera, on_delete=models.SET_NULL, null=True, blank=True,
    )
    
    incident_type = models.CharField(
        max_length=20, choices=BEHAVIOR_INCIDENT_CHOICES,
    )
    severity = models.CharField(
        max_length=20, choices=SEVERITY_CHOICES, default='medium',
    )
    
    description = models.TextField(blank=True)
    
    # Snapshot of the incident
    snapshot = models.ImageField(upload_to='incidents/', blank=True, null=True)
    
    # Detection confidence (0.0 - 1.0)
    confidence = models.FloatField(default=0.0)
    
    detected_at = models.DateTimeField(auto_now_add=True)
    
    # WhatsApp alert details
    whatsapp_sent = models.BooleanField(default=False)
    whatsapp_sent_at = models.DateTimeField(null=True, blank=True)
    
    # Admin review
    is_reviewed = models.BooleanField(default=False)
    reviewed_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='incident_reviews',
    )
    reviewed_at = models.DateTimeField(null=True, blank=True)
    admin_notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-detected_at']
        indexes = [
            models.Index(fields=['student', '-detected_at']),
            models.Index(fields=['incident_type', '-detected_at']),
            models.Index(fields=['severity', '-detected_at']),
        ]
        verbose_name = 'Incident Report'
        verbose_name_plural = 'Incident Reports'
    
    def __str__(self):
        student_name = self.student.name if self.student else 'Unknown'
        return f"{student_name} — {self.get_incident_type_display()} @ {self.detected_at}"
    
    def get_incident_emoji(self):
        """Return emoji for incident type."""
        emoji_map = {
            'phone_usage': '📱',
            'eating_food': '🍔',
            'fighting': '🥊',
            'distracted': '😴',
            'other': '⚠️',
        }
        return emoji_map.get(self.incident_type, '⚠️')