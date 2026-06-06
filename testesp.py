import cv2

url = "http://10.17.6.157/stream"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Stream open nahi hua")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame read failed")
        break

    cv2.imshow("ESP32-CAM Stream", frame)

    # Q dabane par exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()