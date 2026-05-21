from django import forms
from .models import ClassroomCamera


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
