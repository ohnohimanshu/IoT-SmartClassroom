import cv2
from flask import Flask, Response, render_template_string, request
import threading
import time
import os

app = Flask(__name__)

video_path = None
cap = None
lock = threading.Lock()

HTML = """
<!doctype html>
<html>
<head>
    <title>Video IP Camera</title>
</head>
<body>
    <h2>Upload Video</h2>
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <input type="file" name="video" accept="video/*">
        <button type="submit">Upload</button>
    </form>

    <h2>Stream</h2>
    <img src="/video" width="800">
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/upload", methods=["POST"])
def upload():
    global video_path, cap

    if "video" not in request.files:
        return "No file uploaded", 400

    file = request.files["video"]

    os.makedirs("uploads", exist_ok=True)

    video_path = os.path.join("uploads", file.filename)
    file.save(video_path)

    with lock:
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(video_path)

    return "Video uploaded successfully. Open /video"


def generate_frames():
    global cap

    while True:
        if cap is None:
            time.sleep(0.1)
            continue

        with lock:
            ret, frame = cap.read()

            if not ret:
                # Restart video when it reaches the end
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )

        time.sleep(0.03)  # ~30 FPS


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)