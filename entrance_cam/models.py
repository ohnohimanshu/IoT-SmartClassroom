from django.db import models
from django.contrib.auth.models import User

COURSE_CHOICES = [
    ('B.Tech', 'B.Tech'), ('M.Tech', 'M.Tech'), ('BCA', 'BCA'),
    ('MCA', 'MCA'), ('B.Sc', 'B.Sc'), ('M.Sc', 'M.Sc'),
    ('MBA', 'MBA'), ('Other', 'Other'),
]
YEAR_CHOICES = [(1,'1st Year'),(2,'2nd Year'),(3,'3rd Year'),(4,'4th Year'),(5,'5th Year')]
EMOTION_CHOICES = [
    ('happy','Happy'),('sad','Sad'),('angry','Angry'),('neutral','Neutral'),
    ('surprise','Surprise'),('fear','Fear'),('disgust','Disgust'),('unknown','Unknown'),
]

class Student(models.Model):
    name = models.CharField(max_length=100)
    roll_no = models.CharField(max_length=30, unique=True)
    email = models.EmailField(unique=True)
    course = models.CharField(max_length=20, choices=COURSE_CHOICES)
    branch = models.CharField(max_length=100)
    year = models.IntegerField(choices=YEAR_CHOICES)
    photo = models.ImageField(upload_to='students/photos/')
    face_encoding = models.TextField(blank=True, null=True)
    user = models.OneToOneField(User, on_delete=models.SET_NULL,
                                 null=True, blank=True, related_name='student_profile')
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.name} ({self.roll_no})"

    class Meta:
        ordering = ['name']


class Camera(models.Model):
    name = models.CharField(max_length=100)
    url = models.CharField(max_length=300, help_text="IP camera stream URL e.g. http://192.168.1.100:8080/video")
    location = models.CharField(max_length=100, default='Entrance')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} @ {self.location}"


class AttendanceLog(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='attendance_logs')
    camera = models.ForeignKey(Camera, on_delete=models.SET_NULL, null=True, blank=True)
    date = models.DateField(auto_now_add=True)
    entry_time = models.DateTimeField(null=True, blank=True)
    exit_time = models.DateTimeField(null=True, blank=True)
    entry_emotion = models.CharField(max_length=20, choices=EMOTION_CHOICES, default='unknown')
    exit_emotion = models.CharField(max_length=20, choices=EMOTION_CHOICES, default='unknown')
    entry_emotion_score = models.FloatField(default=0.0)
    exit_emotion_score = models.FloatField(default=0.0)
    entry_snapshot = models.ImageField(upload_to='snapshots/entry/', blank=True, null=True)
    exit_snapshot = models.ImageField(upload_to='snapshots/exit/', blank=True, null=True)
    duration_minutes = models.IntegerField(null=True, blank=True)
    is_present = models.BooleanField(default=True)

    class Meta:
        ordering = ['-date', '-entry_time']
        unique_together = ['student', 'date']

    def __str__(self):
        return f"{self.student.name} — {self.date}"

    def calculate_duration(self):
        if self.entry_time and self.exit_time:
            delta = self.exit_time - self.entry_time
            self.duration_minutes = int(delta.total_seconds() / 60)
            self.save(update_fields=['duration_minutes'])
