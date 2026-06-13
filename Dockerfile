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

# Create startup script inline
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

echo "Starting Django server..."
exec gunicorn --workers 4 --bind 0.0.0.0:8000 classroom_iot.wsgi:application
EOF

RUN chmod +x /app/start.sh

# Use inline startup script as entrypoint
ENTRYPOINT ["/app/start.sh"]