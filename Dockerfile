FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POSTGRES_HOST=${POSTGRES_HOST:-db} \
    POSTGRES_PORT=${POSTGRES_PORT:-5432}

# Set work directory
WORKDIR /app

# Install system dependencies optimized for headless OpenCV, dlib compilation, and Mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc \
    g++ \
    git \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libasound2 \
    libgomp1 \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Upgrade core pip layout to handle modern pyproject.toml / wheel builds cleanly
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first to leverage caching
COPY requirements.txt .

# Network-resilient install: added long timeouts and retries for huge ML packages.
# Also limits dlib C++ compilation to 2 cores to prevent Docker RAM exhaustion.
RUN MAKEFLAGS="-j2" pip install \
    --no-cache-dir \
    --default-timeout=1000 \
    --retries 10 \
    -r requirements.txt

# Copy project files
COPY . .

# Collect static files (with safety fallback if DB/env variables aren't injected yet)
RUN python manage.py collectstatic --noinput || true

# Expose port 8000 for Django
EXPOSE 8000

# Install supervisor to manage multiple processes (Django + detection scripts)
RUN pip install --no-cache-dir supervisor

# Create supervisor config directory
RUN mkdir -p /etc/supervisor/conf.d

# Supervisor config for Gunicorn (Django web server)
RUN cat > /etc/supervisor/conf.d/gunicorn.conf << 'EOF'
[program:gunicorn]
command=gunicorn --workers 4 --bind 0.0.0.0:8000 classroom_iot.wsgi:application
directory=/app
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/supervisor/gunicorn.log
priority=999
EOF

# Create a Python helper script to generate supervisor configs from database
COPY scripts/generate_supervisor_configs.py /app/generate_supervisor_configs.py
RUN chmod +x /app/generate_supervisor_configs.py

RUN chmod +x /app/generate_supervisor_configs.py

# Create startup script
RUN cat > /app/start.sh << 'EOF'
#!/bin/bash
set -e

echo "Waiting for PostgreSQL to start..."
while ! nc -z $POSTGRES_HOST $POSTGRES_PORT; do
  sleep 0.1
done
echo "PostgreSQL started"

echo "Applying database migrations..."
python manage.py migrate

echo "Creating superuser if not exists..."
python manage.py shell << PYEOF
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='himanshu').exists():
    User.objects.create_superuser('himanshu', 'himanshuxdei@gmail.com', 'him@nshu131')
    print("Superuser 'himanshu' created successfully!")
else:
    print("Superuser 'himanshu' already exists.")
PYEOF

echo "Generating supervisor configs for camera detection scripts..."
python generate_supervisor_configs.py || echo "⚠ Warning: Could not generate configs (database may not be ready yet)"

echo "Starting supervisor (Django + face detection)..."
exec /usr/local/bin/supervisord -c /etc/supervisor/supervisord.conf -n
EOF

RUN chmod +x /app/start.sh

# Create supervisor main config
RUN cat > /etc/supervisor/supervisord.conf << 'EOF'
[supervisord]
nodaemon=true
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid

[unix_http_server]
file=/var/run/supervisor.sock

[supervisorctl]
serverurl=unix:///var/run/supervisor.sock

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[include]
files = /etc/supervisor/conf.d/*.conf
EOF

# Create log directory for supervisor
RUN mkdir -p /var/log/supervisor

# Use inline startup script as entrypoint
ENTRYPOINT ["/app/start.sh"]