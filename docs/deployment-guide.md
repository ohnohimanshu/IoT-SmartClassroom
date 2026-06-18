# Deployment Guide

## Prerequisites
- Docker and Docker Compose (recommended)
- Or Python 3.11+, PostgreSQL 15+

---

## Docker Compose Deployment (Recommended)

### 1. Prepare Environment Variables
Create a `.env` file in the project root:
```env
# Django
DJANGO_SECRET_KEY=your-production-secret-key-here
DJANGO_DEBUG=false
DJANGO_ALLOWED_HOSTS=your-domain.com,www.your-domain.com

# Database
POSTGRES_DB=classroom_iot_db
POSTGRES_USER=classroom_iot_user
POSTGRES_PASSWORD=secure-password-here
POSTGRES_HOST=db
POSTGRES_PORT=5432

# Optional: Twilio for WhatsApp alerts
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
TWILIO_WHATSAPP_FROM=whatsapp:+1234567890
TWILIO_WHATSAPP_TO=whatsapp:+0987654321
```

### 2. SSL Certificates
Place your SSL certificates in the `certs/` directory:
- `certs/server.crt`
- `certs/server.key`

### 3. Build and Start Containers
```bash
docker-compose up -d --build
```

### 4. Run Migrations
```bash
docker-compose exec web python manage.py migrate
```

### 5. Create Superuser
```bash
docker-compose exec web python manage.py createsuperuser
```

### 6. Collect Static Files
```bash
docker-compose exec web python manage.py collectstatic --noinput
```

---

## Manual Deployment

### 1. System Dependencies (Ubuntu/Debian)
```bash
apt-get update
apt-get install -y python3 python3-pip python3-venv postgresql postgresql-contrib nginx
```

### 2. Clone and Setup Project
```bash
git clone <repo-url>
cd classroom_iot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure PostgreSQL
```sql
sudo -u postgres psql
CREATE DATABASE classroom_iot_db;
CREATE USER classroom_iot_user WITH PASSWORD 'secure-password';
GRANT ALL PRIVILEGES ON DATABASE classroom_iot_db TO classroom_iot_user;
\q
```

### 4. Setup Environment Variables
Create `.env` file as described in Docker section.

### 5. Run Migrations and Collect Static
```bash
python manage.py migrate
python manage.py collectstatic --noinput
```

### 6. Setup Supervisor (for detection scripts)
Copy `docker/supervisord.conf` to `/etc/supervisor/conf.d/` and adjust paths.

### 7. Configure Nginx
Use the nginx configuration from the project as a reference.

---

## Infrastructure Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Storage | 20 GB | 50+ GB |
| OS | Linux (Ubuntu 22.04+) | Linux (Ubuntu 22.04+) |

---

## CI/CD Workflow (Example with GitHub Actions)
```yaml
name: Deploy
on: [push]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to server
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.SERVER_HOST }}
          username: ${{ secrets.SERVER_USER }}
          key: ${{ secrets.SSH_PRIVATE_KEY }}
          script: |
            cd /path/to/classroom_iot
            git pull
            docker-compose up -d --build
            docker-compose exec web python manage.py migrate
            docker-compose exec web python manage.py collectstatic --noinput
```

---

## Monitoring & Logging
- Use `docker-compose logs` to view container logs
- Check application logs in `/var/log/supervisor/` (container)
- Monitor database with `pg_stat_activity`

---

## Backup & Recovery

### Database Backup
```bash
docker-compose exec db pg_dump -U classroom_iot_user classroom_iot_db > backup.sql
```

### Restore Database
```bash
cat backup.sql | docker-compose exec -T db psql -U classroom_iot_user classroom_iot_db
```

### Media Files Backup
Regularly backup the `media/` directory.
