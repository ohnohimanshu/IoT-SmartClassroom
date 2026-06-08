
#!/bin/bash
export POSTGRES_HOST=${POSTGRES_HOST:-db}
export POSTGRES_PORT=${POSTGRES_PORT:-5432}
# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to start..."
while ! nc -z $POSTGRES_HOST $POSTGRES_PORT; do
  sleep 0.1
done
echo "PostgreSQL started"

# Apply database migrations
echo "Applying database migrations..."
python manage.py migrate

# Create superuser if not exists
echo "Creating superuser if not exists..."
python manage.py shell << END
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='himanshu').exists():
    User.objects.create_superuser('himanshu', 'himanshuxdei@gmail.com', 'him@nshu131')
    print("Superuser 'himanshu' created successfully!")
else:
    print("Superuser 'himanshu' already exists.")
END

# Start the Django server (or Gunicorn)
echo "Starting Django server..."
exec "$@"
