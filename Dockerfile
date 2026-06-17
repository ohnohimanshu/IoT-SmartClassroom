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
    dos2unix \
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

# Copy external configuration files
COPY docker/ /docker_configs/

# Apply dos2unix to avoid any Windows CRLF issues, then move files to correct locations
RUN dos2unix /docker_configs/start.sh /docker_configs/gunicorn.conf /docker_configs/supervisord.conf && \
    mv /docker_configs/gunicorn.conf /etc/supervisor/conf.d/gunicorn.conf && \
    mv /docker_configs/supervisord.conf /etc/supervisor/supervisord.conf && \
    mv /docker_configs/start.sh /app/start.sh && \
    chmod +x /app/start.sh

# Create a Python helper script to generate supervisor configs from database
COPY scripts/generate_supervisor_configs.py /app/generate_supervisor_configs.py
RUN chmod +x /app/generate_supervisor_configs.py

# Create log directory for supervisor
RUN mkdir -p /var/log/supervisor

# Use bash explicitly to execute startup script to avoid parsing issues
ENTRYPOINT ["/bin/bash", "/app/start.sh"]