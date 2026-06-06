# ESP32-CAM Setup Guide for Attendance System

## Quick Start (Recommended)
1. **Use HTTP by default** (the code is pre-configured for this!)
2. Update WiFi credentials (lines 75-76 in `esp32cam_optimized__1_.ino`)
3. Upload code
4. Add camera to Django admin with URL like `http://192.168.1.100/stream`
5. Run server

## Prerequisites
1. Arduino IDE with ESP32 board support
2. USB-TTL adapter (CH340, CP2102, etc.)
3. ESP32-CAM module
4. 5V power supply (recommended for stability)

## Choose Server Type
The ESP32-CAM code supports two modes:
- **HTTP (default, easy)**: No certificates needed, great for testing
- **HTTPS (secure)**: Requires self-signed certificates

### To switch between modes:
Edit `esp32cam_optimized__1_.ino` lines 3-4:
```cpp
#define USE_HTTP  // No certificates needed, easy for testing
// #define USE_HTTPS  // Secure, requires certificates below
```
Comment/uncomment to select your preferred mode.

## (Optional) Generate Self-Signed Certificates (for HTTPS only)
Run these commands in your terminal:

```bash
# Generate private key
openssl genrsa -out key.pem 2048

# Generate certificate (valid for 10 years)
openssl req -new -x509 -key key.pem -out cert.pem -days 3650 -subj "/CN=esp32cam"
```

## Step 1: Configure ESP32-CAM Code
1. Open `esp32cam_optimized__1_.ino` in Arduino IDE
2. (HTTPS only) If using HTTPS:
   - Paste contents of `cert.pem` into `SERVER_CERT_PEM` (lines 8-25)
   - Paste contents of `key.pem` into `SERVER_KEY_PEM` (lines 27-56)
3. Update WiFi credentials (lines 75-76):
   - `ssid` = your WiFi network name
   - `password` = your WiFi password

## Step 2: Upload the Code
1. Connect your USB-TTL adapter to ESP32-CAM:
   - GND → GND
   - TX → RX
   - RX → TX
   - 3.3V/5V → VCC
   - GPIO 0 → GND (for upload mode)
2. Select correct board and port in Arduino IDE
3. Press upload button
4. When upload completes, disconnect GPIO 0 from GND and restart the board

## Step 3: Configure Server-Side
1. Install required Python packages:
   ```bash
   pip install mediapipe
   ```
2. Run the server:
   ```bash
   # Either
   python manage.py runall
   # Or with SSL (if using HTTPS on ESP32)
   python manage.py runall --ssl
   ```

## Step 4: Add Camera in Django Admin
1. Go to `http://localhost:8000/admin/`
2. Add a new Camera
3. Set the URL to your ESP32-CAM's IP:
   - For HTTP: `http://192.168.1.100/stream`
   - For HTTPS: `https://192.168.1.100/stream`
4. Check "Active" and save

## Troubleshooting
- **ESP32 certificate errors**: Use HTTP mode instead of HTTPS for testing
- **Camera not connecting**: Check IP address and WiFi connection
- **No face detection**: Install mediapipe, ensure proper lighting

