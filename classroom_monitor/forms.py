from django import forms
from .models import ClassroomCamera, ClassroomVideo


class ClassroomCameraForm(forms.ModelForm):
    class Meta:
        model = ClassroomCamera
        fields = ['name', 'url', 'location', 'is_active', 'stream_type', 'snapshot_url']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Classroom A Camera'}),
            'url': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'http://192.168.1.100:8080/video'}),
            'location': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Classroom A'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'stream_type': forms.Select(attrs={'class': 'form-select'}),
            'snapshot_url': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'http://192.168.1.100/snapshot.jpg'}),
        }


class ClassroomVideoForm(forms.ModelForm):
    class Meta:
        model = ClassroomVideo
        fields = ['title', 'video_file', 'notes']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Monday Morning Lecture'}),
            'video_file': forms.FileInput(attrs={'class': 'form-control', 'accept': 'video/mp4,video/x-m4v,video/*'}),
            'notes': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Optional notes about this video...'}),
        }
