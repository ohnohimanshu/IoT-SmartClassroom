# User Manual

## Table of Contents
1. [Getting Started](#getting-started)
2. [Admin Dashboard](#admin-dashboard)
3. [Student Management](#student-management)
4. [Attendance Tracking](#attendance-tracking)
5. [Classroom Monitoring](#classroom-monitoring)
6. [Lab Monitoring](#lab-monitoring)
7. [FAQs](#faqs)
8. [Troubleshooting Tips](#troubleshooting-tips)

---

## Getting Started

### Logging In
1. Navigate to the application URL
2. Enter your username and password
3. Click "Login"

### First-Time Setup (Admin)
1. Log in with your superuser account
2. Add students (see [Student Management](#student-management))
3. Configure cameras and ESP32 devices

---

## Admin Dashboard

The dashboard provides an overview of:
- Total students
- Active cameras
- Today's attendance
- Currently present students
- Recent attendance logs
- Weekly attendance trends

---

## Student Management

### Adding a Student
1. Go to "Students" → "Add Student"
2. Fill in the form:
   - Name
   - Roll Number (unique)
   - Email (unique)
   - Course and Branch
   - Year
   - Upload a clear photo (for face recognition)
3. Click "Save"
4. The system will automatically generate a face encoding

### Editing a Student
1. Go to "Students" → Click "Edit" on a student
2. Update the information
3. Click "Save"

### Enrolling Fingerprint
1. Go to student detail page
2. Click "Enroll Fingerprint"
3. Follow the instructions on the ESP32 device

---

## Attendance Tracking

### Camera-Based Attendance
- Automatic via face recognition
- Detects entry/exit times
- Captures emotion at entry/exit
- Logs are visible in "Attendance" page

### Fingerprint Attendance
- Students scan their finger on ESP32 device
- Records entry/exit
- Viewable in attendance logs

### Viewing Attendance
1. Go to "Attendance"
2. Filter by date
3. See both camera and fingerprint logs

---

## Classroom Monitoring

### Starting a Class Session
1. Go to "Classroom Monitor" → "Live Monitor"
2. Select a camera
3. Enter subject and teacher name
4. Click "Start Session"

### Live Monitoring
- Real-time engagement tracking
- Detects student poses (focused, looking away, etc.)
- Alerts for phone use, eating, or fights
- Viewable in "Live Monitor" page

### Ending a Session
1. Go to "Classroom Monitor" → "Active Sessions"
2. Click "End Session"

### Viewing Session History
1. Go to "Classroom Monitor" → "Session History"

---

## Lab Monitoring

### Student - Starting a Lab Session
1. Log in as a student
2. Go to "Lab Session"
3. Click "Start Session"
4. Allow camera and screen sharing permissions
5. The session is now active

### Admin - Monitoring Lab Sessions
1. Go to "Lab Monitor" → "Dashboard"
2. View all active sessions
3. Click on a session to view details, screenshots, and camera feed
4. Use WebRTC to view student's screen and camera in real-time

### Ending a Lab Session
- Student: Click "End Session"
- Admin: Can end sessions from monitor dashboard

---

## FAQs

**Q: Why isn't face recognition working for a student?**
A: Make sure the student's photo is clear, well-lit, and shows their face clearly. Try re-uploading the photo.

**Q: How do I add a new camera?**
A: Go to "Camera Attendance" → "Cameras" → "Add Camera". Enter the camera's stream URL or use webcam index 0 for local camera.

**Q: Can I use the system without cameras?**
A: Yes, you can use only fingerprint-based attendance.

**Q: How are WhatsApp alerts sent?**
A: Alerts are sent for critical incidents like fights or phone use (needs Twilio configuration).

---

## Troubleshooting Tips

### Camera Not Connecting
- Check the camera URL
- Ensure the camera is on the same network
- Test the URL in a browser first

### Face Recognition Not Working
- Regenerate the student's face encoding (admin)
- Check if the student's photo is clear
- Ensure good lighting in the room

### ESP32 Device Offline
- Check the device's network connection
- Verify the API key
- Restart the device

### Lab Session WebRTC Issues
- Ensure browser supports WebRTC (Chrome, Firefox, Edge)
- Check camera/screen permissions
- Try refreshing the page

For more detailed troubleshooting, see [Troubleshooting Guide](./troubleshooting.md)
