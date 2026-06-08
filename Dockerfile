FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

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

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Collect static files (with safety fallback if DB/env variables aren't injected yet)
RUN python manage.py collectstatic --noinput || true

# Expose port 8000 for Django
EXPOSE 8000

# Use custom entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (points to your classroom_iot WSGI app)
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8000", "classroom_iot.wsgi:application"]