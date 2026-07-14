# stream.py — Video IP Camera Simulator

A tiny standalone Flask utility for testing the camera attendance pipeline without a physical IP camera.

---

## What it does

- Upload any video file via a browser form
- Loops the video and serves it as a live MJPEG stream at `/video`
- Simulates an IP camera that detection scripts can point at

---

## How it works

1. `POST /upload` — saves the uploaded file to `uploads/`, opens it with `cv2.VideoCapture`
2. `GET /video` — streams frames as `multipart/x-mixed-replace` MJPEG at ~30fps
3. When the video ends, `cap.set(CAP_PROP_POS_FRAMES, 0)` restarts it seamlessly

Frames are read under a `threading.Lock` so the upload and stream threads don't conflict.

---

## Usage

```bash
python stream.py
# Open http://localhost:8080 in a browser
# Upload a video file
# Point your detection script at http://<your-ip>:8080/video
```

---

## Dependencies

`flask`, `opencv-python` (or `-headless`)
