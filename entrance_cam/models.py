from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import datetime

COURSE_CHOICES = [
    ('B.Tech', 'B.Tech'), ('M.Tech', 'M.Tech'), ('BCA', 'BCA'),
    ('MCA', 'MCA'), ('B.Sc', 'B.Sc'), ('M.Sc', 'M.Sc'),
    ('MBA', 'MBA'), ('Other', 'Other'),
]
YEAR_CHOICES = [
    (1, '1st Year'), (2, '2nd Year'), (3, '3rd Year'),
    (4, '4th Year'), (5, '5th Year'),
]
EMOTION_CHOICES = [
    ('happy', 'Happy'), ('sad', 'Sad'), ('angry', 'Angry'),
    ('neutral', 'Neutral'), ('surprise', 'Surprise'), ('fear', 'Fear'),
    ('disgust', 'Disgust'), ('unknown', 'Unknown'),
]
MOOD_COMPARISON_CHOICES = [
    ('improved', 'Improved'),
    ('declined', 'Declined'),
    ('stable', 'Stable'),
    ('unknown', 'Unknown'),
]


class Student(models.Model):
    name       = models.CharField(max_length=100)
    roll_no    = models.CharField(max_length=30, unique=True)
    email      = models.EmailField(unique=True)
    course     = models.CharField(max_length=20, choices=COURSE_CHOICES)
    branch     = models.CharField(max_length=100)
    year       = models.IntegerField(choices=YEAR_CHOICES)
    photo      = models.ImageField(upload_to='students/photos/')
    # Stores JSON list of 128 floats produced by face_recognition
    face_encoding = models.TextField(blank=True, null=True)
    user       = models.OneToOneField(
        User, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='student_profile',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    is_active  = models.BooleanField(default=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return f"{self.name} ({self.roll_no})"


class Camera(models.Model):
    name       = models.CharField(max_length=100)
    url        = models.CharField(
        max_length=300,
        help_text="IP camera stream URL e.g. http://192.168.1.100:8080/video",
    )
    location   = models.CharField(max_length=100, default='Entrance')
    is_active  = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} @ {self.location}"


class AttendanceLog(models.Model):
    student    = models.ForeignKey(
        Student, on_delete=models.CASCADE, related_name='attendance_logs',
    )
    camera     = models.ForeignKey(
        Camera, on_delete=models.SET_NULL, null=True, blank=True,
    )

    # FIX 1: Use default=datetime.date.today (callable) instead of auto_now_add=True.
    # auto_now_add makes the field non-writable, which breaks explicit log creation
    # in views.py.  A callable default still auto-fills on creation but stays writable.
    date       = models.DateField(default=datetime.date.today)

    entry_time  = models.DateTimeField(null=True, blank=True)
    exit_time   = models.DateTimeField(null=True, blank=True)

    entry_emotion       = models.CharField(max_length=20, choices=EMOTION_CHOICES, default='unknown')
    exit_emotion        = models.CharField(max_length=20, choices=EMOTION_CHOICES, default='unknown')
    entry_emotion_score = models.FloatField(default=0.0)
    exit_emotion_score  = models.FloatField(default=0.0)

    entry_snapshot = models.ImageField(upload_to='snapshots/entry/', blank=True, null=True)
    exit_snapshot  = models.ImageField(upload_to='snapshots/exit/',  blank=True, null=True)

    # FIX 2: Add the mood_comparison field that views.py writes on every exit.
    # Without this the exit path raises AttributeError every time.
    mood_comparison = models.CharField(
        max_length=20, choices=MOOD_COMPARISON_CHOICES,
        default='unknown', blank=True,
    )

    duration_minutes = models.IntegerField(null=True, blank=True)
    is_present       = models.BooleanField(default=True)

    class Meta:
        ordering = ['-date', '-entry_time']
        # FIX 3: Removed unique_together = ['student', 'date'].
        # That constraint allowed only ONE attendance row per student per day,
        # which crashed with IntegrityError on a student's second entry of the day
        # and made the multi-visit logic in views.py impossible to use.
        # The open-log query in views.py (entry logged, exit NULL) correctly
        # handles multiple visits without needing a DB-level uniqueness constraint.

    def __str__(self):
        return f"{self.student.name} — {self.date}"

    def calculate_duration(self):
        """
        FIX 4: Pure in-memory helper — does NOT call self.save().
        Callers (views.py) are responsible for saving.  The old version called
        self.save() here, which (a) fired post_save signals redundantly and
        (b) could overwrite other unsaved field changes the caller had made.
        """
        if self.entry_time and self.exit_time:
            delta = self.exit_time - self.entry_time
            self.duration_minutes = int(delta.total_seconds() / 60)