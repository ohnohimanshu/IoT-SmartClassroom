# -*- coding: utf-8 -*-
from pathlib import Path
import os
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

os.environ['PYTHONIOENCODING'] = 'utf-8'

# Suppress mediapipe / clearcut / TFLite telemetry noise
os.environ.setdefault('GLOG_minloglevel', '3')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

# Load .env file so TWILIO_* and WHATSAPP_* vars are available everywhere
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / '.env')
except ImportError:
    pass  # python-dotenv not installed — vars must be set in the OS environment

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
ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get(
        "DJANGO_ALLOWED_HOSTS",
        "127.0.0.1,localhost,10.7.31.114"
    ).split(",")
    if h.strip()
]

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'sslserver',
    'entrance_cam',
    'camera_attendance',
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
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': os.environ.get('POSTGRES_DB', 'classroom_iot_db'),
        'USER': os.environ.get('POSTGRES_USER', 'classroom_iot_user'),
        'PASSWORD': os.environ.get('POSTGRES_PASSWORD', 'change_this_in_production'),
        'HOST': os.environ.get('POSTGRES_HOST', 'db'),
        'PORT': os.environ.get('POSTGRES_PORT', '5432'),
        'CONN_MAX_AGE': 600,  # Reuse connections for 10 minutes
        'OPTIONS': {
            'connect_timeout': 10,
        }
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
# Set DJANGO_USE_SSL=true in your environment when using runsslserver.
_USE_SSL = os.environ.get('DJANGO_USE_SSL', 'false').lower() == 'true'
SECURE_SSL_REDIRECT = _USE_SSL  # Only redirect HTTP→HTTPS when actually running SSL

# Cookies must only be marked Secure when actually running over HTTPS.
SESSION_COOKIE_SECURE = _USE_SSL
CSRF_COOKIE_SECURE = _USE_SSL
SESSION_COOKIE_HTTPONLY = True
CSRF_COOKIE_HTTPONLY = False  # Must be False — JS needs to read this cookie for fetch/XHR requests
CSRF_COOKIE_SAMESITE = 'Lax'
SESSION_COOKIE_SAMESITE = 'Lax'
CSRF_COOKIE_DOMAIN = None
SESSION_COOKIE_DOMAIN = None

# For runsslserver, use: python manage.py runsslserver 0.0.0.0:8000 --certificate cert.pem --key key.pem
# Generate self-signed cert: openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

CSRF_TRUSTED_ORIGINS = [
    'http://127.0.0.1:8000',
    'https://127.0.0.1:8000',
    'http://localhost:8000',
    'https://localhost:8000',
    # 192.168.x network
    'https://10.7.31.114',
    'http://10.7.31.114:8000',
    'https://10.7.31.114:8000',
    # 172.x network (WSL/Hyper-V virtual adapter)
    'http://172.22.224.1:8000',
    'https://172.22.224.1:8000',
]

SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
CSRF_FAILURE_VIEW = 'entrance_cam.views.csrf_failure'
