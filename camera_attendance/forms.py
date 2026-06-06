from django import forms
from django.core.exceptions import ValidationError
from .models import Camera


class CameraForm(forms.ModelForm):
    """Form for creating and editing camera records."""

    class Meta:
        model = Camera
        fields = ['name', 'url', 'location', 'is_active']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Main Gate Camera'}),
            'url': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'http://192.168.1.100:8080/video'}),
            'location': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Lab Entrance'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

    def clean_name(self):
        """Validate camera name."""
        name = self.cleaned_data.get('name', '').strip()
        if not name:
            raise ValidationError('Camera name is required.')
        if len(name) < 2:
            raise ValidationError('Camera name must be at least 2 characters.')
        return name

    def clean_url(self):
        """Validate camera URL."""
        url = self.cleaned_data.get('url', '').strip()
        if not url:
            raise ValidationError('Camera URL is required.')
        # Allow numeric webcam index or valid URL
        if not url.isdigit():
            if not (url.startswith('http://') or url.startswith('https://')):
                raise ValidationError('URL must start with http:// or https://')
        return url
