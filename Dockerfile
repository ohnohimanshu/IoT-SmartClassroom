FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies for OpenCV, PyTorch, dlib, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxrender1 \
    libxrandr2 \
    libasound2 \
    libpng-dev \
    libjpeg-dev \
    zlib1g-dev \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to cache them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Collect static files (optional, you can also run this via docker-compose command)
RUN python manage.py collectstatic --noinput || true

# Expose port 8000 for Django
EXPOSE 8000

# Run Gunicorn (WSGI server)
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8000", "classroom_iot.wsgi:application"]