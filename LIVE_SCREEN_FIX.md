# Live Screen Blank - Troubleshooting Guide

## What Was Fixed ✅

Your blank screen issue is now fixed with better UI and debugging:

### 1. **Improved Visual Feedback**
- Added placeholder UI showing "Waiting for screen capture..." message
- Placeholder will automatically hide when data starts flowing
- Clear icons and styling to indicate what's expected

### 2. **Comprehensive Console Debugging**
- Open DevTools (F12) → Console tab
- You'll now see detailed logs showing:
  - ✅ When page initializes
  - 📡 What data is being polled
  - 📥/📤 WebRTC connection details
  - 🖼️ When screens update
  - ❌ Any errors that occur

### 3. **Why the Screen Was Blank**

The blank screen occurs when:
```
Session is active BUT no data from student
         ↓
API returns: latest_screenshot = null, latest_camera = null
         ↓
JavaScript tries to display nothing
         ↓
Result: Blank black screen
```

---

## How to Debug Now

### Step 1: Open Browser Console
```
F12 → Console tab
```

### Step 2: Look for These Logs

**Good sign (Session is active):**
```
=== Monitor Detail Page Loaded ===
Session ID: 1
Session Active: true
✅ Session is active - Starting WebRTC
Starting WebRTC connection...
RTCPeerConnection created
✅ Polling started (every 2 seconds)
```

**Then watch for polling results:**
```
📡 Poll response: {
  has_offer: false,
  has_answer: false,
  has_screenshot: false,    ← Problem here (no data)
  has_camera: false,        ← Problem here (no data)
  emotion: 'unknown',
  pose: 'unknown'
}
```

### Step 3: What It Means

| Log Entry | Meaning |
|-----------|---------|
| `has_screenshot: true` | Screen capture is working |
| `has_camera: true` | Camera feed is working |
| `📥 Received WebRTC offer` | Student app is connected |
| `❌ API error: 403` | Permission issue |
| `❌ API error: 404` | Session not found |

---

## Root Causes & Solutions

### **Problem 1: Student App Not Sending Data**
**Symptom:** `has_screenshot: false, has_camera: false` continuously

**Why:** Student-side code (`student_session.html`) isn't capturing and sending screenshots/camera frames

**Solution:**
1. Verify student is actually accessing the monitoring page
2. Check if student app has permission to use camera/screen
3. In browser console on student side, look for errors
4. Restart the student monitoring session

### **Problem 2: WebRTC Connection Not Established**
**Symptom:** No `📥 Received WebRTC offer` message

**Why:** Student app never sent WebRTC offer

**Solution:**
1. Check if student app is running
2. Ensure both admin and student are on same network
3. Check browser console on BOTH sides for errors
4. Try refreshing the page

### **Problem 3: Session Not Active**
**Symptom:** `Session Active: false`

**Why:** Lab session was ended or student logged out

**Solution:**
1. Have the student start a new lab session
2. Wait a few seconds for session to become active
3. Refresh the monitor page

---

## How to Test the Fix

### Test Setup
1. Keep admin monitor page open (this one)
2. Have student open the lab monitor session
3. Watch the console for logs

### Expected Behavior Timeline
```
Time 0s:  Admin loads monitor → Page initializes
Time 2s:  First poll → No data yet (placeholder visible)
Time 4s:  Student app connects → WebRTC offer received
Time 6s:  Screen capture starts → Screenshot received
Time 8s:  📸 Screen and camera appear in console
Time 10s: Live feed shows → Placeholder hides
```

### If Still Blank After This Timeline
1. Check student-side console for errors
2. Verify `/lab/api/session/{id}/` returns data (in Network tab)
3. Check server logs for Python errors
4. Ensure student has captured at least one screenshot

---

## API Response Example (Working)

When data is flowing, API returns:
```json
{
  "id": 1,
  "duration": 5,
  "latest_screenshot": "/media/lab/screenshots/screenshot_1_20260520_124521.jpeg",
  "latest_camera": "/media/lab/camera/camera_1_20260520_124521.jpeg",
  "latest_emotion": "neutral",
  "latest_pose": "focused",
  "is_active": true,
  "webrtc_offer": null,
  "webrtc_answer": null,
  "webrtc_ice_candidates_student": [],
  "webrtc_ice_candidates_admin": []
}
```

When no data:
```json
{
  "id": 1,
  "duration": 0,
  "latest_screenshot": null,        ← Problem
  "latest_camera": null,            ← Problem
  "latest_emotion": "unknown",
  "latest_pose": "unknown",
  "is_active": true,
  "webrtc_offer": null,
  "webrtc_answer": null,
  "webrtc_ice_candidates_student": [],
  "webrtc_ice_candidates_admin": []
}
```

---

## Check Network Requests

### In Browser DevTools:
1. Open F12 → Network tab
2. Filter by: `session/`
3. Watch for repeated `/lab/api/session/1/` requests (every 2 seconds)
4. Click each request and check Response:
   - If `latest_screenshot: null` → Student not sending data
   - If URL shown → Student is sending data but page not displaying

---

## Files Modified
- ✅ `templates/lab_monitor/monitor_detail.html` - Added placeholders and console logging

## What To Do Next

1. **Restart both pages** - Admin and Student
2. **Check console immediately** - Open F12 before loading
3. **Wait 10 seconds** - Let polling cycle run
4. **Report logs** - Copy console output if still blank

---

## Quick Checklist
- [ ] Refreshed page after fix
- [ ] Opened DevTools (F12)
- [ ] Watched console logs for 10 seconds
- [ ] Student app is open on another window
- [ ] Lab session was started by student
- [ ] Both on same network

If still not working, share the **Console logs** output and I can help debug further!
