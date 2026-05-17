from django import forms
from .models import Student, Camera


class StudentForm(forms.ModelForm):
    class Meta:
        model = Student
        fields = ['name', 'roll_no', 'email', 'course', 'branch', 'year', 'photo']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Full Name'}),
            'roll_no': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 2021CS001'}),
            'email': forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'student@college.edu'}),
            'course': forms.Select(attrs={'class': 'form-select'}),
            'branch': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Computer Science'}),
            'year': forms.Select(attrs={'class': 'form-select'}),
            'photo': forms.FileInput(attrs={'class': 'form-control', 'accept': 'image/*'}),
        }


class CameraForm(forms.ModelForm):
    class Meta:
        model = Camera
        fields = ['name', 'url', 'location', 'is_active']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Main Gate Camera'}),
            'url': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'http://192.168.1.100:8080/video'}),
            'location': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Lab Entrance'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
