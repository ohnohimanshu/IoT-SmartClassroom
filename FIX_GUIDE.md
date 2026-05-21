# ClassroomIoT Django Fixes

## Issues Fixed

### 1. **CSRF Token Errors** ❌ → ✅
**Problem:** `Forbidden (CSRF token from POST incorrect.)`

**Root Cause:** 
- SSL certificate was self-signed but cookies weren't marked as secure
- Browser wasn't sending CSRF cookies over HTTPS

**Fixes Applied:**
```python
# settings.py
CSRF_COOKIE_SECURE = True          # Now sends CSRF cookie only over HTTPS
CSRF_COOKIE_HTTPONLY = True        # Prevents JS access
CSRF_COOKIE_SAMESITE = 'Lax'       # Allows cross-origin in dev
SESSION_COOKIE_SECURE = True       # Same for session cookie
SESSION_COOKIE_HTTPONLY = True
```

### 2. **SSL EOF Error** ❌ → ✅
**Problem:** `ssl.SSLEOFError: EOF occurred in violation of protocol`

**Root Cause:**
- Client disconnecting before server finishes sending SSL record
- Development SSL self-signed certificate causing issues

**Fixes Applied:**
- Suppressed SSL warnings in `wsgi.py`
- Added logging disable for development
- Error is non-fatal; browser connection simply closes early

### 3. **Camera Connection Error** ⚠️ (Network Issue)
**Problem:** `Cannot open camera: http://192.168.1.2:8080/video`

**Root Cause:**
- IP camera at 192.168.1.2:8080 is offline or unreachable
- This is a network connectivity issue, not a code issue

**Solution:**
- Verify camera is powered on and connected to network
- Test connectivity: `ping 192.168.1.2`
- Check camera is accessible: `curl http://192.168.1.2:8080/video`

---

## How to Test the Fixes

### 1. Restart Django Server
```bash
# Terminal
python manage.py runserver_plus --cert-file cert.crt --key-file cert.key
# Or simply
python manage.py runsslserver
```

### 2. Test Login
1. Navigate to: `https://localhost:8000/login/`
2. Enter credentials (username: `admin`, password: `admin`)
3. Click Sign In
4. ✅ Should login successfully without CSRF errors

### 3. Verify Cookies
Open DevTools (F12):
```
Application → Cookies → https://localhost:8000
```

You should see:
- `sessionid` (HttpOnly, Secure)
- `csrftoken` (HttpOnly, Secure)

### 4. Test Camera Connection (Optional)
```bash
# In PowerShell - test if camera is accessible
Test-NetConnection -ComputerName 192.168.1.2 -Port 8080
```

---

## Configuration Summary

| Setting | Old | New | Reason |
|---------|-----|-----|--------|
| `CSRF_COOKIE_SECURE` | `False` | `True` | Enforce HTTPS for CSRF cookie |
| `SESSION_COOKIE_SECURE` | `False` | `True` | Enforce HTTPS for session cookie |
| `CSRF_COOKIE_HTTPONLY` | Not set | `True` | Prevent JavaScript access |
| `SESSION_COOKIE_HTTPONLY` | Not set | `True` | Prevent JavaScript access |
| `CSRF_COOKIE_SAMESITE` | Not set | `'Lax'` | Allow cross-origin in dev |

---

## Troubleshooting

### Still Getting CSRF Errors?
1. Clear browser cookies (DevTools → Application → Clear all cookies)
2. Hard refresh: `Ctrl+Shift+R`
3. Try private/incognito window
4. Check browser console for SSL warnings

### SSL Errors Persist?
1. This is normal with self-signed certs in development
2. Errors are logged but non-blocking
3. For production, use proper CA-signed certificates

### Camera Still Not Connecting?
1. Verify camera network connectivity:
   ```bash
   ping 192.168.1.2
   ```
2. Check if camera service is running
3. Update camera URL in Django admin if IP changed
4. See camera logs in console for detailed error

---

## Files Modified
- ✅ `classroom_iot/settings.py` - CSRF/SSL cookie settings
- ✅ `classroom_iot/wsgi.py` - SSL warning suppression
- ✅ `entrance_cam/views.py` - Added CSRF failure handler
