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
python generate_supervisor_configs.py || echo "Warning: Could not generate configs (database may not be ready yet)"

echo "Starting supervisor (Django + face detection)..."
exec /usr/local/bin/supervisord -c /etc/supervisor/supervisord.conf -n
