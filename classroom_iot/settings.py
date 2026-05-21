from pathlib import Path
import os
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

os.environ['PYTHONIOENCODING'] = 'utf-8'

# ── GPU / TensorFlow configuration ────────────────────────────────────────────
# Allow TF to use GPU memory incrementally instead of grabbing it all at once.
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
# Suppress verbose TF startup logs (0=all, 1=info, 2=warn, 3=error)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
# Disable oneDNN floating-point noise warnings
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Security ───────────────────────────────────────────────────────────────────
# Override SECRET_KEY via environment variable in production.
SECRET_KEY = os.environ.get(
    'DJANGO_SECRET_KEY',
    'django-insecure-classroom-iot-dev-key-change-in-production'
)
DEBUG = os.environ.get('DJANGO_DEBUG', 'True') == 'True'
ALLOWED_HOSTS = os.environ.get(
    'DJANGO_ALLOWED_HOSTS',
    '*, 192.168.1.5, 10.17.5.13, 10.7.31.26'
).split(', ')

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'sslserver',
    'entrance_cam',
    'lab_monitor',
    'classroom_monitor',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'classroom_iot.urls'

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [BASE_DIR / 'templates'],
    'APP_DIRS': True,
    'OPTIONS': {'context_processors': [
        'django.template.context_processors.debug',
        'django.template.context_processors.request',
        'django.contrib.auth.context_processors.auth',
        'django.contrib.messages.context_processors.messages',
    ]},
}]

WSGI_APPLICATION = 'classroom_iot.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
        'OPTIONS': {
            # WAL mode is set via connection_created signal in entrance_cam/apps.py
            'timeout': 20,
        },
        'CONN_MAX_AGE': 60,
    }
}

AUTH_PASSWORD_VALIDATORS = []
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kolkata'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/dashboard/'

# SSL/Development settings
SECURE_SSL_REDIRECT = False

# Cookies must only be marked Secure when actually running over HTTPS.
# Set DJANGO_USE_SSL=true in your environment when using runsslserver.
_USE_SSL = os.environ.get('DJANGO_USE_SSL', 'true').lower() == 'true'
SESSION_COOKIE_SECURE = _USE_SSL
CSRF_COOKIE_SECURE = _USE_SSL
SESSION_COOKIE_HTTPONLY = True
CSRF_COOKIE_HTTPONLY = False  # Must be False — JS needs to read this cookie for fetch/XHR requests
CSRF_COOKIE_SAMESITE = 'Lax'

CSRF_TRUSTED_ORIGINS = [
    'http://127.0.0.1:8000',
    'https://127.0.0.1:8000',
    'http://localhost:8000',
    'https://localhost:8000',
    # 192.168.x network
    'http://192.168.1.5:8000',
    'https://192.168.1.5:8000',
    # 10.17.x network (current IP — check with ipconfig)
    'http://10.17.5.13:8000',
    'https://10.17.5.13:8000',
    # 10.7.x network (previous IP — kept for compatibility)
    'http://10.7.5.13:8000',
    'https://10.7.5.13:8000',
    'http://10.7.31.26:8000',
    'https://10.7.31.26:8000',
    # 172.x network (WSL/Hyper-V virtual adapter)
    'http://172.22.224.1:8000',
    'https://172.22.224.1:8000',
]
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
CSRF_FAILURE_VIEW = 'entrance_cam.views.csrf_failure'
