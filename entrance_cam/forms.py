from django import forms
from django.core.exceptions import ValidationError
from .models import Student, ESP32Device
import secrets
import re


class StudentForm(forms.ModelForm):
    """Form for creating and editing student records."""
    
    class Meta:
        model = Student
        fields = ['name', 'roll_no', 'email', 'course', 'branch', 'year', 'photo']
        widgets = {
            'name':    forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Full Name'}),
            'roll_no': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 2021CS001'}),
            'email':   forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'student@college.edu'}),
            'course':  forms.Select(attrs={'class': 'form-select'}),
            'branch':  forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Computer Science'}),
            'year':    forms.Select(attrs={'class': 'form-select'}),
            'photo':   forms.FileInput(attrs={'class': 'form-control', 'accept': 'image/*'}),
        }
    
    def clean_name(self):
        """Validate student name."""
        name = self.cleaned_data.get('name', '').strip()
        if not name:
            raise ValidationError('Name is required.')
        if len(name) < 2:
            raise ValidationError('Name must be at least 2 characters.')
        if len(name) > 100:
            raise ValidationError('Name must be less than 100 characters.')
        return name
    
    def clean_roll_no(self):
        """Validate roll number."""
        roll_no = self.cleaned_data.get('roll_no', '').strip()
        if not roll_no:
            raise ValidationError('Roll number is required.')
        if len(roll_no) < 3:
            raise ValidationError('Roll number must be at least 3 characters.')
        if len(roll_no) > 30:
            raise ValidationError('Roll number must be less than 30 characters.')
        # Check if roll_no already exists (excluding current instance)
        if Student.objects.filter(roll_no=roll_no).exclude(pk=self.instance.pk).exists():
            raise ValidationError('This roll number already exists.')
        return roll_no
    
    def clean_email(self):
        """Validate email address."""
        email = self.cleaned_data.get('email', '').strip().lower()
        if not email:
            raise ValidationError('Email is required.')
        # Check if email already exists (excluding current instance)
        if Student.objects.filter(email=email).exclude(pk=self.instance.pk).exists():
            raise ValidationError('This email already exists.')
        return email
    
    def clean_photo(self):
        """Validate photo file."""
        photo = self.cleaned_data.get('photo')
        if photo:
            # Check file size (max 5MB)
            if photo.size > 5 * 1024 * 1024:
                raise ValidationError('Photo file size must be less than 5MB.')
            # Check file type
            allowed_types = ['image/jpeg', 'image/png', 'image/gif']
            if photo.content_type not in allowed_types:
                raise ValidationError('Only JPEG, PNG, and GIF images are allowed.')
        return photo


class ESP32DeviceForm(forms.ModelForm):
    """Form for creating and editing ESP32 device records."""
    
    class Meta:
        model = ESP32Device
        fields = ['name', 'ip_address', 'location', 'is_active']
        widgets = {
            'name':      forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Main Gate Fingerprint Scanner'}),
            'ip_address': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'http://192.168.1.100'}),
            'location':  forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Main Entrance'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.instance.pk:
            # Generate a strong API key for new devices (64 characters)
            self.instance.api_key = secrets.token_urlsafe(48)
    
    def clean_name(self):
        """Validate device name."""
        name = self.cleaned_data.get('name', '').strip()
        if not name:
            raise ValidationError('Device name is required.')
        if len(name) < 2:
            raise ValidationError('Device name must be at least 2 characters.')
        return name
    
    def clean_ip_address(self):
        """Validate IP address or URL."""
        ip_address = self.cleaned_data.get('ip_address', '').strip()
        if not ip_address:
            raise ValidationError('IP address or URL is required.')
        if not (ip_address.startswith('http://') or ip_address.startswith('https://')):
            raise ValidationError('IP address must start with http:// or https://')
        return ip_address

class AdminCreationForm(forms.Form):
    """Form for superuser to create new admin accounts."""
    username = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Username'}),
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'admin@example.com'}),
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Password'}),
        min_length=8,
    )
    password_confirm = forms.CharField(
        label='Confirm Password',
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Confirm Password'}),
    )

    def clean_username(self):
        from django.contrib.auth import get_user_model
        User = get_user_model()
        username = self.cleaned_data.get('username', '').strip()
        if User.objects.filter(username=username).exists():
            raise ValidationError('A user with this username already exists.')
        return username

    def clean_email(self):
        from django.contrib.auth import get_user_model
        User = get_user_model()
        email = self.cleaned_data.get('email', '').strip().lower()
        if User.objects.filter(email=email).exists():
            raise ValidationError('A user with this email already exists.')
        return email

    def clean(self):
        cleaned_data = super().clean()
        pw = cleaned_data.get('password')
        pw2 = cleaned_data.get('password_confirm')
        if pw and pw2 and pw != pw2:
            self.add_error('password_confirm', 'Passwords do not match.')
        return cleaned_data
