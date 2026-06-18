# Developer Guide

## Project Structure
```
classroom_iot/
├── camera_attendance/       # Camera attendance app
│   ├── migrations/          # Database migrations
│   ├── models.py            # Data models
│   ├── views.py             # Views & API endpoints
│   ├── urls.py              # URL routing
│   └── forms.py             # Forms
├── classroom_monitor/       # Classroom monitoring app
│   ├── behavior_detection.py # ML for behavior
│   ├── face_recognition_helper.py
│   └── ...
├── entrance_cam/            # Entrance & fingerprint app
│   └── ...
├── lab_monitor/             # Lab monitoring app
│   └── ...
├── classroom_iot/           # Project settings
│   ├── settings.py          # Django settings
│   ├── urls.py              # Main URL config
│   └── wsgi.py
├── static/                  # Static files (CSS, JS, images)
├── templates/               # HTML templates
├── media/                   # Uploaded files
├── certs/                   # SSL certificates
├── docker/                  # Docker configs
├── manage.py                # Django management script
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker build file
└── docker-compose.yml       # Docker compose
```

---

## Coding Standards
- Follow PEP 8 for Python code
- Use meaningful variable/function names
- Write docstrings for functions and classes
- Keep functions focused and small
- Use Django ORM instead of raw SQL where possible

---

## Setting Up Development Environment

1. **Clone the repo**
2. **Create virtual env**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Create .env file** (see [Deployment Guide](./deployment-guide.md))
5. **Run migrations**
   ```bash
   python manage.py migrate
   ```
6. **Create superuser**
   ```bash
   python manage.py createsuperuser
   ```
7. **Run dev server**
   ```bash
   python manage.py runserver  # No SSL
   # Or with SSL:
   python manage.py runsslserver
   ```

---

## How to Add a New Feature

### Step 1: Create/Update Models
```python
# In your_app/models.py
from django.db import models

class MyNewModel(models.Model):
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
```

### Step 2: Create and Run Migrations
```bash
python manage.py makemigrations your_app
python manage.py migrate
```

### Step 3: Create Forms (if needed)
```python
# your_app/forms.py
from django import forms
from .models import MyNewModel

class MyNewModelForm(forms.ModelForm):
    class Meta:
        model = MyNewModel
        fields = ['name']
```

### Step 4: Create Views
```python
# your_app/views.py
from django.shortcuts import render
from .models import MyNewModel

def my_view(request):
    items = MyNewModel.objects.all()
    return render(request, 'your_app/template.html', {'items': items})
```

### Step 5: Add URLs
```python
# your_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('my-view/', views.my_view, name='my_view'),
]
```

```python
# classroom_iot/urls.py (add to main urls)
path('your-app/', include('your_app.urls')),
```

### Step 6: Create Templates
Create HTML templates in `templates/your_app/`

---

## Testing Strategy
- Write unit tests for models and utility functions
- Write integration tests for views and API endpoints
- Run tests with:
  ```bash
  python manage.py test
  ```

---

## Contribution Workflow
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-new-feature`
3. Make your changes
4. Run tests: `python manage.py test`
5. Commit your changes: `git commit -m "Add my new feature"`
6. Push to the branch: `git push origin feature/my-new-feature`
7. Open a pull request

---

## Working with ML Models
- Face recognition uses `face-recognition` library
- Emotion detection uses `DeepFace`
- Pose estimation uses `MediaPipe`
- Object detection uses `Ultralytics YOLO`
- Fight detection uses PyTorch 3D CNN

All ML models are lazy-loaded to avoid blocking Django startup.

---

## Debugging Tips
- Use `print()` statements or `logging` module
- Check Django debug toolbar (if installed)
- View logs in `docker-compose logs` (if using Docker)
- Use `pdb` for interactive debugging:
  ```python
  import pdb; pdb.set_trace()
  ```
