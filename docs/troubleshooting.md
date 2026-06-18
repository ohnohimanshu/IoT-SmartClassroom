# Troubleshooting Guide

## Common Errors & Solutions

---

## Build Failures

### Error: "No module named 'xyz'"
**Cause**: Missing dependency
**Solution**:
```bash
pip install -r requirements.txt
```

### Error: "Command 'gcc' failed" (installing dlib/face-recognition)
**Cause**: Missing system dependencies for compiling C extensions
**Solution (Ubuntu/Debian)**:
```bash
apt-get install -y build-essential cmake libopenblas-dev liblapack-dev libjpeg-dev
```

**Solution (Windows)**:
- Install Visual Studio Build Tools
- Or use pre-built wheels: `pip install dlib-20.0.0-cp311-cp311-win_amd64.whl`

### Docker Build Fails
- Check Docker daemon is running
- Check for sufficient disk space
- Try rebuilding with `--no-cache`:
  ```bash
  docker-compose build --no-cache
  ```

---

## Database Issues

### Error: "Could not connect to server: Connection refused"
**Cause**: PostgreSQL not running or wrong host/port
**Solution**:
- Ensure PostgreSQL is running
- Check `POSTGRES_HOST` and `POSTGRES_PORT` in .env
- If using Docker: `docker-compose up -d db`

### Error: "relation 'xyz' does not exist"
**Cause**: Migrations not applied
**Solution**:
```bash
python manage.py migrate
```

### Error: "duplicate key value violates unique constraint"
**Cause**: Trying to create a record with duplicate unique field
**Solution**:
- Check for existing records with that value
- Use `get_or_create()` instead of `create()` where appropriate

---

## Authentication Issues

### Can't Log In
- Check if user exists and is active
- Verify password is correct
- Try resetting password via admin:
  ```bash
  python manage.py shell
  >>> from django.contrib.auth.models import User
  >>> user = User.objects.get(username='your-username')
  >>> user.set_password('new-password')
  >>> user.save()
  ```

### CSRF Verification Failed
- Ensure `CSRF_TRUSTED_ORIGINS` includes your domain in settings.py
- Check that your browser is accepting cookies
- If behind a proxy, ensure `X-Forwarded-Proto` header is set correctly

---

## Camera & Detection Issues

### Camera Not Connecting
- Verify camera URL is correct
- Test URL in VLC or browser
- Check network connectivity to camera
- Ensure camera supports MJPEG/RTSP stream

### Face Recognition Not Working
- Ensure student has a clear photo uploaded
- Check that face encoding was generated (student.is_enrolled should be True)
- Regenerate face encoding:
  ```python
  # In Django shell
  from entrance_cam.models import Student
  student = Student.objects.get(id=1)
  # Re-save to trigger signal (if using signal for encoding)
  student.save()
  ```

### Detection Scripts Not Running
- Check supervisor status:
  ```bash
  supervisorctl status
  ```
- View logs in `/var/log/supervisor/` (container) or project directory
- Restart supervisor:
  ```bash
  supervisorctl restart all
  ```

---

## Deployment Issues

### 502 Bad Gateway (Nginx)
- Check that Django app is running
- Verify Gunicorn is listening on correct port
- Check Nginx error logs:
  ```bash
  tail -f /var/log/nginx/error.log
  ```

### Static Files Not Loading
- Ensure `collectstatic` was run:
  ```bash
  python manage.py collectstatic --noinput
  ```
- Check Nginx config for static files location
- Verify permissions on static directory

### Media Files Not Loading
- Check `MEDIA_URL` and `MEDIA_ROOT` in settings.py
- Verify permissions on media directory
- Ensure Nginx is configured to serve media files

---

## Debugging Procedures

### Enable Debug Mode
In .env file:
```env
DJANGO_DEBUG=true
```
⚠️ **Never use in production!**

### Check Logs
- Django logs: Check console or log file
- Docker logs: `docker-compose logs -f`
- Supervisor logs: `/var/log/supervisor/`
- Nginx logs: `/var/log/nginx/`

### Use Django Debug Toolbar (if installed)
Provides detailed request/response info, SQL queries, etc.

### Shell Plus (if using django-extensions)
```bash
python manage.py shell_plus
```
Automatically imports all models for easy debugging.

---

## Still Stuck?
1. Check the [GitHub Issues](https://github.com/your-repo/issues)
2. Search for similar problems
3. Open a new issue with details:
   - Steps to reproduce
   - Error messages
   - Environment info (OS, Python version, etc.)
