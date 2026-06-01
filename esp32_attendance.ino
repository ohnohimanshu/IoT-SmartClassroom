#include <Adafruit_Fingerprint.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION — edit these before flashing
// ═══════════════════════════════════════════════════════════════════════════════

const char* WIFI_SSID     = "Leo_4G";
const char* WIFI_PASSWORD = "Jack@9187";

// Server running runsslserver → https://
// No trailing slash here — slashes are added per-path below
const char* SERVER_URL    = "https://192.168.1.9:8000";

// Must match ESP32_API_KEY in your Django .env
const char* API_KEY       = "him@nshu131";

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

#define BUZZER_PIN      25
#define OLED_SDA        21
#define OLED_SCL        22
#define FP_RX_PIN       16
#define FP_TX_PIN       17
#define SCREEN_WIDTH   128
#define SCREEN_HEIGHT   64
#define OLED_RESET      -1

const unsigned long IDLE_TIMEOUT     = 30000;
const unsigned long ENROLL_TIMEOUT   = 30000;
const unsigned long HTTP_TIMEOUT_MS  = 8000;
const unsigned long POLL_INTERVAL_MS = 3000;
const unsigned long WIFI_CHECK_MS    = 10000;
const int           HTTP_MAX_RETRIES = 3;

// ═══════════════════════════════════════════════════════════════════════════════
// GLOBALS
// ═══════════════════════════════════════════════════════════════════════════════

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
HardwareSerial   fpSerial(2);
Adafruit_Fingerprint finger(&fpSerial);

volatile bool enrollMode       = false;
int           enrollID         = -1;
unsigned long lastActivityTime = 0;
unsigned long lastWifiCheck    = 0;
volatile int  pendingEnrollID  = -1;

// ═══════════════════════════════════════════════════════════════════════════════
// FORWARD DECLARATIONS
// ═══════════════════════════════════════════════════════════════════════════════

void showIdle();
void showMessage(const char* line1, const char* line2, bool success);
void beep(int times, int durationMs);
void connectWiFi();
void cancelEnroll();
void notifyFlaskEnrolled(int fpID, bool success);
void startEnrollMode(int id);
void handleEnrollment();
void handleAttendance();
void pollForCommands();
void maintainWiFi();
void pollTask(void* param);
void markAttendance(int fpID, int confidence);
void uploadFingerprintImage(int fpID);
bool fp_uploadImage(uint8_t* buf);
size_t base64Encode(const uint8_t* src, size_t srcLen, char* dst);
void showBootAnimation();

// ═══════════════════════════════════════════════════════════════════════════════
// HTTPS HELPER
// setInsecure() skips cert verification — correct for runsslserver self-signed cert
// ═══════════════════════════════════════════════════════════════════════════════

WiFiClientSecure makeSecureClient() {
  WiFiClientSecure client;
  client.setInsecure();
  return client;
}

// ═══════════════════════════════════════════════════════════════════════════════
// HTTP HELPERS
// IMPORTANT: Never call client.connect() manually before http.begin().
//            Let HTTPClient manage the SSL connection entirely.
// ═══════════════════════════════════════════════════════════════════════════════

int httpPost(const String& path, const String& body, String& responseOut) {
  if (WiFi.status() != WL_CONNECTED) return -1;

  String url = String(SERVER_URL) + path;

  for (int attempt = 1; attempt <= HTTP_MAX_RETRIES; attempt++) {
    WiFiClientSecure client = makeSecureClient();
    HTTPClient http;

    if (!http.begin(client, url)) {
      Serial.printf("[HTTP] begin() failed: %s\n", url.c_str());
      http.end();
      delay(1000);
      continue;
    }

    http.setTimeout(HTTP_TIMEOUT_MS);
    http.addHeader("Content-Type", "application/json");
    http.addHeader("X-API-Key", API_KEY);

    int code = http.POST(body);

    if (code > 0) {
      responseOut = http.getString();
      http.end();
      Serial.printf("[HTTP] POST %s → %d (attempt %d)\n", path.c_str(), code, attempt);
      return code;
    }

    Serial.printf("[HTTP] POST %s failed (attempt %d/%d): %s\n",
                  path.c_str(), attempt, HTTP_MAX_RETRIES,
                  HTTPClient::errorToString(code).c_str());
    http.end();
    if (attempt < HTTP_MAX_RETRIES) delay(1500 * attempt);
  }
  return -1;
}

int httpGet(const String& path, String& responseOut) {
  if (WiFi.status() != WL_CONNECTED) return -1;

  String url = String(SERVER_URL) + path;
  WiFiClientSecure client = makeSecureClient();
  HTTPClient http;

  if (!http.begin(client, url)) {
    Serial.printf("[HTTP] begin() failed: %s\n", url.c_str());
    http.end();
    return -1;
  }

  http.setTimeout(HTTP_TIMEOUT_MS);
  http.addHeader("X-API-Key", API_KEY);

  int code = http.GET();
  if (code > 0) responseOut = http.getString();
  http.end();

  Serial.printf("[HTTP] GET %s → %d\n", path.c_str(), code);
  return code;
}

// ═══════════════════════════════════════════════════════════════════════════════
// BOOT ANIMATION
// ═══════════════════════════════════════════════════════════════════════════════

void showBootAnimation() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(2);
  display.setCursor(10, 20);
  display.println("Welcome");
  display.setTextSize(1);
  display.setCursor(30, 44);
  display.println("to");
  display.display();
  delay(900);

  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(14, 14);
  display.println("Biometric");
  display.setCursor(8, 30);
  display.println("Attendance");
  display.display();
  delay(900);

  display.clearDisplay();
  display.drawRect(0, 0, 128, 64, SSD1306_WHITE);
  display.setTextSize(2);
  display.setCursor(16, 22);
  display.println("System");
  display.display();
  delay(900);

  for (int p = 0; p <= 128; p += 8) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(4, 8);
    display.println("Welcome to");
    display.setCursor(4, 22);
    display.println("Biometric Attendance");
    display.setCursor(28, 36);
    display.println("System");
    display.drawRect(4, 52, 120, 8, SSD1306_WHITE);
    display.fillRect(4, 52, p, 8, SSD1306_WHITE);
    display.display();
    delay(40);
  }
  delay(400);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SETUP
// ═══════════════════════════════════════════════════════════════════════════════

void setup() {
  Serial.begin(115200);

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  Wire.begin(OLED_SDA, OLED_SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("[ERROR] SSD1306 not found");
    while (true) delay(10);
  }

  showMessage("Booting...", "", false);
  showBootAnimation();

  fpSerial.begin(57600, SERIAL_8N1, FP_RX_PIN, FP_TX_PIN);
  finger.begin(57600);
  delay(100);

  if (!finger.verifyPassword()) {
    showMessage("Sensor Error!", "Check wiring", false);
    Serial.println("[ERROR] Fingerprint sensor password failed");
    while (true) delay(10);
  }
  Serial.println("[OK] Fingerprint sensor ready");

  connectWiFi();

  // Test server reachability via HTTPS
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("[SETUP] Testing server...");
    String testResp;
    int testCode = httpGet("/api/esp32/command/", testResp);
    if (testCode == 200 || testCode == 401) {
      Serial.printf("[SETUP] ✓ Server reachable (code %d)\n", testCode);
    } else {
      Serial.printf("[SETUP] ⚠ Server returned %d — check SERVER_URL\n", testCode);
    }
  }

  xTaskCreatePinnedToCore(pollTask, "pollTask", 8192, NULL, 1, NULL, 0);

  showIdle();
}

// ═══════════════════════════════════════════════════════════════════════════════
// POLL TASK — Core 0 (background)
// ═══════════════════════════════════════════════════════════════════════════════

void pollTask(void* param) {
  for (;;) {
    if (WiFi.status() == WL_CONNECTED && !enrollMode) {
      pollForCommands();
    }
    vTaskDelay(pdMS_TO_TICKS(POLL_INTERVAL_MS));
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN LOOP — Core 1
// FIX: clear pendingEnrollID BEFORE calling startEnrollMode to prevent
//      the poll task from re-setting it during enrollment.
// ═══════════════════════════════════════════════════════════════════════════════

void loop() {
  maintainWiFi();

  if (!enrollMode && pendingEnrollID > 0) {
    int idToEnroll  = pendingEnrollID;
    pendingEnrollID = -1;          // clear BEFORE starting — prevents re-trigger
    startEnrollMode(idToEnroll);
  }

  if (enrollMode) {
    handleEnrollment();
  } else {
    handleAttendance();
  }

  if (!enrollMode && millis() - lastActivityTime > IDLE_TIMEOUT) {
    showIdle();
    lastActivityTime = millis();
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WIFI
// ═══════════════════════════════════════════════════════════════════════════════

void connectWiFi() {
  Serial.printf("[WiFi] Connecting to %s\n", WIFI_SSID);
  showMessage("Connecting WiFi", WIFI_SSID, false);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] Connected — IP: %s\n", WiFi.localIP().toString().c_str());
    Serial.printf("[WiFi] Gateway: %s\n", WiFi.gatewayIP().toString().c_str());
    showMessage("WiFi Connected", WiFi.localIP().toString().c_str(), false);
    beep(1, 100);
    delay(1000);
  } else {
    Serial.println("\n[WiFi] Failed — offline mode");
    showMessage("WiFi Failed", "Offline mode", false);
    delay(2000);
  }
}

void maintainWiFi() {
  if (millis() - lastWifiCheck < WIFI_CHECK_MS) return;
  lastWifiCheck = millis();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Lost — reconnecting...");
    showMessage("WiFi Lost", "Reconnecting...", false);
    WiFi.disconnect();
    delay(500);
    connectWiFi();
    if (!enrollMode) showIdle();
  } else {
    Serial.printf("[WiFi] OK. Signal: %d dBm, Heap: %u bytes\n",
                  WiFi.RSSI(), ESP.getFreeHeap());
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMMAND POLLING
// ═══════════════════════════════════════════════════════════════════════════════

void pollForCommands() {
  String response;
  int code = httpGet("/api/esp32/command/", response);

  if (code == 200) {
    StaticJsonDocument<256> doc;
    DeserializationError err = deserializeJson(doc, response);
    if (err) {
      Serial.printf("[POLL] JSON parse error: %s\n", err.c_str());
      return;
    }

    if (doc["command"].isNull() || doc["command"] == "") return;

    String cmd = doc["command"].as<String>();
    Serial.printf("[POLL] Command: %s\n", cmd.c_str());

    if (cmd == "ENROLL") {
      int id = doc["fingerprint_id"] | -1;
      if (id > 0) {
        pendingEnrollID = id;
        Serial.printf("[POLL] ✓ pendingEnrollID = %d\n", id);
      } else {
        Serial.println("[POLL] ✗ ENROLL missing valid fingerprint_id");
      }
    }

  } else if (code == 401) {
    Serial.println("[POLL] ✗ 401 Unauthorized — check API_KEY");
  } else if (code == -1) {
    Serial.println("[POLL] ✗ Connection failed");
  } else {
    Serial.printf("[POLL] Unexpected code: %d\n", code);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ATTENDANCE
// ═══════════════════════════════════════════════════════════════════════════════

void handleAttendance() {
  uint8_t result = finger.getImage();
  if (result != FINGERPRINT_OK) { delay(20); return; }

  result = finger.image2Tz();
  if (result != FINGERPRINT_OK) return;

  result = finger.fingerSearch();
  if (result == FINGERPRINT_OK) {
    int fpID       = finger.fingerID;
    int confidence = finger.confidence;
    Serial.printf("[FP] Match — ID #%d, confidence %d\n", fpID, confidence);
    markAttendance(fpID, confidence);
    lastActivityTime = millis();
  } else if (result == FINGERPRINT_NOTFOUND) {
    showMessage("Not Registered", "Contact admin", false);
    beep(3, 100);
    delay(2000);
    showIdle();
  }
}

void markAttendance(int fpID, int confidence) {
  showMessage("Verifying...", "Please wait", false);

  if (WiFi.status() != WL_CONNECTED) {
    showMessage("No WiFi!", "Cannot mark", false);
    beep(2, 200);
    delay(2000);
    showIdle();
    return;
  }

  StaticJsonDocument<128> doc;
  doc["fingerprint_id"] = fpID;
  doc["confidence"]     = confidence;
  String body;
  serializeJson(doc, body);

  String response;
  int statusCode = httpPost("/api/mark-attendance/", body, response);

  if (statusCode == 200) {
    StaticJsonDocument<256> res;
    DeserializationError err = deserializeJson(res, response);
    if (err) {
      showMessage("Parse Error", "", false);
      beep(2, 200);
    } else {
      String name   = res["name"]   | "Unknown";
      String action = res["action"] | "Marked";
      String line1  = (action == "IN") ? "Welcome IN" : "Goodbye OUT";
      showMessage(line1.c_str(), name.c_str(), true);
      beep((action == "IN") ? 1 : 2, 150);
    }
  } else if (statusCode == 404) {
    showMessage("Not Registered", "Contact admin", false);
    beep(3, 100);
  } else if (statusCode == -1) {
    showMessage("Unreachable", "Check network", false);
    beep(2, 300);
  } else {
    showMessage("Server Error", String(statusCode).c_str(), false);
    beep(2, 300);
  }

  delay(2500);
  showIdle();
}

// ═══════════════════════════════════════════════════════════════════════════════
// R307 RAW IMAGE UPLOAD — FIXED
//
// The Adafruit library does NOT expose writePacket/readPacket/address publicly,
// and it does not define FINGERPRINT_UPLOADIMAGE.  We therefore send the UpImage
// command (0x0A) directly over fpSerial and parse the reply ourselves.
// ═══════════════════════════════════════════════════════════════════════════════

#define FINGERPRINT_UPLOADIMAGE 0x0A   // UpImage — NOT defined by Adafruit lib
#define FP_IMG_BYTES  36864            // 256 × 144 nibbles packed = 256×288/2
#define FP_IMG_WIDTH  256
#define FP_IMG_HEIGHT 288

// Sensor default address (from R307 datasheet)
static const uint32_t FP_DEFAULT_ADDRESS = 0xFFFFFFFF;

// Helper: compute checksum for a packet (sum of packet type + length + payload)
static uint16_t fp_checksum(const uint8_t* data, size_t len) {
  uint16_t sum = 0;
  for (size_t i = 0; i < len; i++) sum += data[i];
  return sum;
}

// Helper: send a command packet directly over fpSerial
static void fp_sendCommand(uint8_t cmd) {
  uint8_t packet[16];
  size_t idx = 0;

  // Header
  packet[idx++] = 0xEF;
  packet[idx++] = 0x01;

  // Address (4 bytes, big-endian)
  packet[idx++] = (FP_DEFAULT_ADDRESS >> 24) & 0xFF;
  packet[idx++] = (FP_DEFAULT_ADDRESS >> 16) & 0xFF;
  packet[idx++] = (FP_DEFAULT_ADDRESS >>  8) & 0xFF;
  packet[idx++] = (FP_DEFAULT_ADDRESS      ) & 0xFF;

  // Packet type: command
  packet[idx++] = 0x01;

  // Length (2 bytes, big-endian) — payload is just 1 byte (cmd) + 2 bytes checksum
  uint16_t len = 1 + 2;
  packet[idx++] = (len >> 8) & 0xFF;
  packet[idx++] =  len       & 0xFF;

  // Payload
  packet[idx++] = cmd;

  // Checksum (2 bytes, big-endian) — sum of packet type + length + payload
  uint16_t cs = fp_checksum(&packet[6], 4); // type + lenH + lenL + cmd = 4 bytes
  packet[idx++] = (cs >> 8) & 0xFF;
  packet[idx++] =  cs       & 0xFF;

  fpSerial.write(packet, idx);
  fpSerial.flush();
}

bool fp_uploadImage(uint8_t* buf) {
  // ── Clear any stale bytes ─────────────────────────────────────────────────
  while (fpSerial.available()) fpSerial.read();

  // ── Send UpImage (0x0A) command directly ──────────────────────────────────
  fp_sendCommand(FINGERPRINT_UPLOADIMAGE);

  // ── Read confirmation ACK packet ──────────────────────────────────────────
  // Packet format: EF 01 [addr×4] [type] [lenH lenL] [confCode] [csH csL]
  uint8_t ack[12];
  size_t ackIdx = 0;
  uint32_t ackTimeout = millis();

  while (ackIdx < 12) {
    if (millis() - ackTimeout > 3000) {
      Serial.println("[FP] ACK timeout");
      return false;
    }
    if (fpSerial.available()) {
      ack[ackIdx++] = fpSerial.read();
      ackTimeout = millis(); // reset per-byte timeout
    }
  }

  // Verify header
  if (ack[0] != 0xEF || ack[1] != 0x01) {
    Serial.printf("[FP] Bad ACK header: 0x%02X 0x%02X\n", ack[0], ack[1]);
    return false;
  }

  // Confirmation code is at offset 9 (after 6-byte header + 2-byte len + 1-byte type)
  uint8_t confCode = ack[9];
  if (confCode != 0x00) {
    Serial.printf("[FP] UpImage ACK failed: confCode=0x%02X\n", confCode);
    return false;
  }

  Serial.println("[FP] UpImage ACK OK — streaming data packets...");

  // ── Stream the image data packets ─────────────────────────────────────────
  size_t   received = 0;
  uint32_t packetTimeout = millis();
  const uint32_t PKT_TIMEOUT_MS = 5000;

  while (received < FP_IMG_BYTES) {
    // Wait for 6-byte packet header: 0xEF 0x01 + 4-byte address
    while (fpSerial.available() < 6) {
      if (millis() - packetTimeout > PKT_TIMEOUT_MS) {
        Serial.printf("[FP] Timeout waiting for packet header at byte %u\n", received);
        return false;
      }
      delay(1);
    }

    uint8_t hdr[6];
    fpSerial.readBytes(hdr, 6);

    if (hdr[0] != 0xEF || hdr[1] != 0x01) {
      Serial.printf("[FP] Bad packet header 0x%02X 0x%02X at byte %u — draining & aborting\n",
                    hdr[0], hdr[1], received);
      delay(100);
      while (fpSerial.available()) fpSerial.read();
      return false;
    }

    // Next 3 bytes: packet type + 2-byte length
    while (fpSerial.available() < 3) {
      if (millis() - packetTimeout > PKT_TIMEOUT_MS) {
        Serial.println("[FP] Timeout reading packet type/len");
        return false;
      }
      delay(1);
    }
    uint8_t  thisPktType = fpSerial.read();
    uint8_t  lenH        = fpSerial.read();
    uint8_t  lenL        = fpSerial.read();
    uint16_t dataLen     = (uint16_t)(lenH << 8 | lenL);

    if (dataLen < 2 || dataLen > 512) {
      Serial.printf("[FP] Implausible dataLen=%u (pktType=0x%02X) — aborting\n",
                    dataLen, thisPktType);
      delay(100);
      while (fpSerial.available()) fpSerial.read();
      return false;
    }

    uint16_t payloadLen = dataLen - 2;   // strip trailing 2-byte checksum

    // Wait for full payload
    uint32_t payloadWait = millis();
    while ((int)fpSerial.available() < (int)payloadLen) {
      if (millis() - payloadWait > PKT_TIMEOUT_MS) {
        Serial.printf("[FP] Timeout reading payload (%u bytes) at byte %u\n",
                      payloadLen, received);
        return false;
      }
      delay(1);
    }

    // Copy into buffer (guard against overflow)
    size_t toRead = min((size_t)payloadLen, FP_IMG_BYTES - received);
    fpSerial.readBytes(buf + received, toRead);

    // Drain excess if any
    for (size_t extra = toRead; extra < (size_t)payloadLen; extra++) {
      uint32_t drainWait = millis();
      while (!fpSerial.available()) {
        if (millis() - drainWait > 500) break;
        delay(1);
      }
      if (fpSerial.available()) fpSerial.read();
    }

    received += toRead;

    // Consume 2-byte checksum
    uint32_t csWait = millis();
    while (fpSerial.available() < 2) {
      if (millis() - csWait > 500) break;
      delay(1);
    }
    if (fpSerial.available() >= 2) {
      fpSerial.read();
      fpSerial.read();
    }

    packetTimeout = millis();   // reset per-packet timeout

    if (thisPktType == 0x08) break;  // 0x08 = end-of-data packet
  }

  Serial.printf("[FP] Image stream complete: %u bytes received\n", received);
  return (received > 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BASE64 ENCODER
// ═══════════════════════════════════════════════════════════════════════════════

static const char b64chars[] =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

size_t base64Encode(const uint8_t* src, size_t srcLen, char* dst) {
  size_t out = 0;
  for (size_t i = 0; i < srcLen; i += 3) {
    uint32_t b = (uint32_t)src[i] << 16;
    if (i + 1 < srcLen) b |= (uint32_t)src[i + 1] << 8;
    if (i + 2 < srcLen) b |= src[i + 2];
    dst[out++] = b64chars[(b >> 18) & 0x3F];
    dst[out++] = b64chars[(b >> 12) & 0x3F];
    dst[out++] = (i + 1 < srcLen) ? b64chars[(b >> 6) & 0x3F] : '=';
    dst[out++] = (i + 2 < srcLen) ? b64chars[b & 0x3F]        : '=';
  }
  dst[out] = '\0';
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
// UPLOAD FINGERPRINT IMAGE TO SERVER
// Call this BEFORE image2Tz() while the raw image is still in the sensor buffer.
// ═══════════════════════════════════════════════════════════════════════════════

void uploadFingerprintImage(int fpID) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[FP] Skipping image upload — no WiFi");
    return;
  }

  showMessage("Saving image...", "", false);
  Serial.println("[FP] Uploading fingerprint image...");

  // Prefer PSRAM for the 36 KB image buffer, fall back to heap
  uint8_t* imgBuf = (uint8_t*)ps_malloc(FP_IMG_BYTES);
  if (!imgBuf) imgBuf = (uint8_t*)malloc(FP_IMG_BYTES);
  if (!imgBuf) {
    Serial.println("[FP] OOM: cannot allocate image buffer");
    return;
  }

  if (!fp_uploadImage(imgBuf)) {
    Serial.println("[FP] Image capture failed — skipping upload");
    free(imgBuf);
    return;
  }

  // Allocate base64 output buffer
  size_t b64Len = ((FP_IMG_BYTES + 2) / 3) * 4 + 1;
  char*  b64Buf = (char*)malloc(b64Len);
  if (!b64Buf) {
    Serial.println("[FP] OOM: cannot allocate base64 buffer");
    free(imgBuf);
    return;
  }

  base64Encode(imgBuf, FP_IMG_BYTES, b64Buf);
  free(imgBuf);

  String url = String(SERVER_URL) + "/api/esp32/upload-image/";

  // Build JSON body carefully — String(char*) doesn't always copy data
  // Build prefix first
  String body = "{\"fingerprint_id\":" + String(fpID) + ",\"image\":\"";
  
  // Append base64 data directly from the buffer
  body += b64Buf;
  
  // Append suffix
  body += "\"}";
  
  free(b64Buf);

  Serial.printf("[FP] JSON body size: %d bytes\n", body.length());

  WiFiClientSecure client = makeSecureClient();
  HTTPClient http;

  if (http.begin(client, url)) {
    http.setTimeout(20000);
    http.addHeader("Content-Type", "application/json");
    http.addHeader("X-API-Key", API_KEY);
    
    int code = http.POST(body);
    Serial.printf("[FP] Image upload → HTTP %d\n", code);
    
    if (code != 200) {
      String response = http.getString();
      Serial.printf("[FP] Error response: %s\n", response.c_str());
    }
    
    http.end();
  } else {
    Serial.println("[FP] Image upload: http.begin() failed");
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ENROLLMENT
//
// FIX 1: uploadFingerprintImage() is called BEFORE image2Tz(1) while the
//         raw image still lives in the sensor's buffer.
// FIX 2: enrollMode / enrollID cleared BEFORE any delay() or HTTP call so
//         loop() cannot re-enter handleEnrollment() during that window.
// FIX 3: pendingEnrollID cleared in loop() before startEnrollMode() so the
//         poll task cannot queue a duplicate enrollment.
// ═══════════════════════════════════════════════════════════════════════════════

void startEnrollMode(int id) {
  enrollMode       = true;
  enrollID         = id;
  lastActivityTime = millis();
  Serial.printf("[ENROLL] Starting for slot %d\n", id);
  showMessage("Enroll Mode", ("ID: " + String(id)).c_str(), false);
  beep(2, 80);
  delay(1000);
}

void handleEnrollment() {
  // ── Step 1: first scan ────────────────────────────────────────────────────
  showMessage("Place Finger", ("ID: " + String(enrollID)).c_str(), false);

  unsigned long stepStart = millis();
  while (finger.getImage() != FINGERPRINT_OK) {
    delay(50);
    if (millis() - stepStart > ENROLL_TIMEOUT) { cancelEnroll(); return; }
  }

  // Convert to template slot 1 (no image upload needed)
  if (finger.image2Tz(1) != FINGERPRINT_OK) {
    showMessage("Bad scan", "Try again", false);
    delay(1500);
    return;   // stay in enrollMode; loop() will call us again
  }

  // ── Step 2: lift finger ───────────────────────────────────────────────────
  showMessage("Remove Finger", "", false);
  delay(1500);
  while (finger.getImage() != FINGERPRINT_NOFINGER) delay(100);
  finger.getImage();
  delay(200);

  // ── Step 3: second scan ───────────────────────────────────────────────────
  showMessage("Place Again", "Same finger", false);
  stepStart = millis();
  while (finger.getImage() != FINGERPRINT_OK) {
    delay(50);
    if (millis() - stepStart > ENROLL_TIMEOUT) { cancelEnroll(); return; }
  }

  if (finger.image2Tz(2) != FINGERPRINT_OK) {
    showMessage("Bad scan", "Retry", false);
    delay(1500);
    return;
  }

  // ── Step 4: create model ──────────────────────────────────────────────────
  if (finger.createModel() != FINGERPRINT_OK) {
    showMessage("Mismatch!", "Try again", false);
    beep(3, 100);
    delay(2000);
    return;
  }

  // ── Step 5: store & notify ────────────────────────────────────────────────
  // CRITICAL: clear enrollMode / enrollID BEFORE any delay() or HTTP call.
  // If we delay first, loop() re-enters handleEnrollment() during that window.
  int  savedID = enrollID;
  bool storeOk = (finger.storeModel(savedID) == FINGERPRINT_OK);

  enrollMode = false;   // ← must happen before delay() or notifyFlaskEnrolled()
  enrollID   = -1;

  if (storeOk) {
    Serial.printf("[ENROLL] ✓ Template stored in slot %d\n", savedID);
    showMessage("Enrolled!", ("ID: " + String(savedID)).c_str(), true);
    beep(3, 80);
    notifyFlaskEnrolled(savedID, true);
  } else {
    Serial.printf("[ENROLL] ✗ storeModel failed for slot %d\n", savedID);
    showMessage("Store Failed", "Try again", false);
    beep(2, 300);
    notifyFlaskEnrolled(savedID, false);
  }

  delay(3000);
  showIdle();
}

void notifyFlaskEnrolled(int fpID, bool success) {
  StaticJsonDocument<128> doc;
  doc["fingerprint_id"] = fpID;
  doc["success"]        = success;
  String body;
  serializeJson(doc, body);

  String response;
  httpPost("/api/esp32/enroll-result/", body, response);
  Serial.printf("[ENROLL] Server notified: fp=%d success=%s → %s\n",
                fpID, success ? "true" : "false", response.c_str());
}

void cancelEnroll() {
  int savedID = enrollID;
  enrollMode  = false;   // clear immediately
  enrollID    = -1;
  showMessage("Enroll Timeout", "Cancelled", false);
  beep(2, 200);
  delay(2000);
  notifyFlaskEnrolled(savedID, false);
  showIdle();
}

// ═══════════════════════════════════════════════════════════════════════════════
// DISPLAY HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

void showIdle() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(10, 5);
  display.println("Attendance System");
  display.drawLine(0, 15, 127, 15, SSD1306_WHITE);
  display.setTextSize(2);
  display.setCursor(8, 28);
  display.println("Scan");
  display.setCursor(8, 48);
  display.println("Finger");
  display.display();
  lastActivityTime = millis();
}

void showMessage(const char* line1, const char* line2, bool success) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  if (success) {
    display.drawRect(0, 0, 128, 64, SSD1306_WHITE);
    display.drawLine(4, 4, 124, 4, SSD1306_WHITE);
  }
  display.setTextSize(1);
  display.setCursor(4, success ? 10 : 4);
  display.println(line1);
  if (strlen(line2) > 0) {
    display.setTextSize(2);
    display.setCursor(4, success ? 30 : 28);
    display.println(line2);
  }
  display.display();
}

// ═══════════════════════════════════════════════════════════════════════════════
// BUZZER
// ═══════════════════════════════════════════════════════════════════════════════

void beep(int times, int durationMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(durationMs);
    digitalWrite(BUZZER_PIN, LOW);
    if (i < times - 1) delay(80);
  }
}