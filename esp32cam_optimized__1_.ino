
#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include "esp_timer.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"

// ─── WiFi ────────────────────────────────────────────────────────────────────
const char* ssid     = "DEIWF";
const char* password = "";

// ─── Camera model: AI Thinker ────────────────────────────────────────────────
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ─── Stream settings ─────────────────────────────────────────────────────────
#define CAM_FRAME_SIZE   FRAMESIZE_QVGA   // 320×240 — optimal for face_recognition
/*
  JPEG quality: 1-63 on ESP32 (HIGHER number = MORE compression = SMALLER file).
  18 → ~12-18 KB/frame.  Was 12 (too large → WiFi floods).
  Do NOT go below 15 unless you have a strong router and PSRAM.
*/
#define CAM_JPEG_QUALITY  18
#define CAM_FB_COUNT       2              // double-buffer (requires PSRAM)
#define CAM_XCLK_FREQ   20000000         // 20 MHz

/*
  Frame throttle: minimum milliseconds between sending frames.
  80 ms ≈ 12 fps.  Keeps TCP write queue short → no lwIP buffer exhaustion.
  Lower = smoother but more likely to drop WiFi on weak signal.
  Raise to 100 (10 fps) if you still see disconnects.
*/
#define FRAME_MIN_MS  80

// ─── MJPEG boundary ──────────────────────────────────────────────────────────
static const char* STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=gc0p4Jq0M2Yt08jU534c0p";
static const char* STREAM_BOUNDARY =
    "\r\n--gc0p4Jq0M2Yt08jU534c0p\r\n";
static const char* STREAM_PART =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

httpd_handle_t stream_httpd = NULL;

// ─── WiFi reconnect state ────────────────────────────────────────────────────
static uint8_t  wifiFailCount   = 0;
static uint32_t lastWifiCheck   = 0;

// ─────────────────────────────────────────────────────────────────────────────
//  STREAM HANDLER
// ─────────────────────────────────────────────────────────────────────────────
static esp_err_t stream_handler(httpd_req_t* req)
{
    camera_fb_t* fb  = NULL;
    esp_err_t    res = ESP_OK;
    char         part_buf[128];

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store, no-cache");
    httpd_resp_set_hdr(req, "Connection", "close");   // no keep-alive on stream

    res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;

    int sock = httpd_req_to_sockfd(req);

    // TCP_NODELAY: flush every write immediately (no Nagle buffering)
    int nodelay = 1;
    lwip_setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

    // Send timeout: if the client stalls > 4 s, abort the connection
    // so we don't hold the camera fb indefinitely.
    struct timeval tv = { .tv_sec = 4, .tv_usec = 0 };
    lwip_setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    // Enlarge TCP send buffer: reduces fragmentation & retransmits
    int sndbuf = 32768;
    lwip_setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

    uint32_t lastFrame = 0;

    while (true)
    {
        // ── Per-frame throttle ──────────────────────────────────────────────
        uint32_t now = (uint32_t)(esp_timer_get_time() / 1000ULL);
        int32_t  wait = (int32_t)FRAME_MIN_MS - (int32_t)(now - lastFrame);
        if (wait > 0)
            vTaskDelay(pdMS_TO_TICKS(wait));

        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("[WARN] Capture failed — skipping");
            vTaskDelay(pdMS_TO_TICKS(5));
            continue;
        }

        lastFrame = (uint32_t)(esp_timer_get_time() / 1000ULL);

        size_t hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART, fb->len);

        res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
        if (res == ESP_OK)
            res = httpd_resp_send_chunk(req, part_buf, hlen);
        if (res == ESP_OK)
            res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);

        esp_camera_fb_return(fb);
        fb = NULL;

        if (res != ESP_OK) {
            Serial.println("[INFO] Stream client disconnected");
            break;
        }

        // Yield so WiFi / TCP tasks get CPU time
        taskYIELD();
    }

    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
//  INFO ENDPOINT  — GET /info
// ─────────────────────────────────────────────────────────────────────────────
static esp_err_t info_handler(httpd_req_t* req)
{
    char buf[300];
    snprintf(buf, sizeof(buf),
        "{\"psram\":%s,\"frame\":\"QVGA 320x240\","
        "\"quality\":%d,\"fps_cap\":%d,\"ip\":\"%s\","
        "\"free_heap\":%u}",
        psramFound() ? "true" : "false",
        CAM_JPEG_QUALITY,
        (int)(1000 / FRAME_MIN_MS),
        WiFi.localIP().toString().c_str(),
        (unsigned)esp_get_free_heap_size()
    );
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, buf, strlen(buf));
}

// ─────────────────────────────────────────────────────────────────────────────
//  HEALTH ENDPOINT  — GET /healthz
//  Returns 200 OK immediately; lets Python confirm the ESP is alive.
// ─────────────────────────────────────────────────────────────────────────────
static esp_err_t health_handler(httpd_req_t* req)
{
    const char* ok = "{\"status\":\"ok\"}";
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, ok, strlen(ok));
}

// ─────────────────────────────────────────────────────────────────────────────
//  START HTTP SERVER
// ─────────────────────────────────────────────────────────────────────────────
void startCameraServer()
{
    httpd_config_t config     = HTTPD_DEFAULT_CONFIG();
    config.stack_size         = 8192;
    config.server_port        = 80;
    config.ctrl_port          = 32768;
    config.max_open_sockets   = 4;     // 1 stream + 1 info + 1 health + 1 spare
    /*
      CRITICAL FIX: task_priority was 5 = same as WiFi/lwIP driver tasks.
      When the stream loop ran flat-out it starved the WiFi stack → disconnects.
      Priority 2 = below WiFi (5) and below idle (0+1) but above background.
    */
    config.task_priority      = 2;
    config.core_id            = 1;     // pin HTTP to core 1; WiFi uses core 0
    config.recv_wait_timeout  = 5;
    config.send_wait_timeout  = 5;
    config.lru_purge_enable   = true;  // auto-close stale sockets

    httpd_uri_t stream_uri = { "/stream",  HTTP_GET, stream_handler, NULL };
    httpd_uri_t info_uri   = { "/info",    HTTP_GET, info_handler,   NULL };
    httpd_uri_t health_uri = { "/healthz", HTTP_GET, health_handler, NULL };

    if (httpd_start(&stream_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(stream_httpd, &stream_uri);
        httpd_register_uri_handler(stream_httpd, &info_uri);
        httpd_register_uri_handler(stream_httpd, &health_uri);
        Serial.println("[OK] HTTP server started on port 80");
    } else {
        Serial.println("[ERR] HTTP server failed to start — restarting");
        delay(2000);
        ESP.restart();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  WiFi EVENT HANDLER
// ─────────────────────────────────────────────────────────────────────────────
void WiFiEvent(WiFiEvent_t event)
{
    switch (event)
    {
        case ARDUINO_EVENT_WIFI_STA_DISCONNECTED:
            Serial.printf("[WIFI] Disconnected (fail#%d)\n", wifiFailCount + 1);
            // Don't call WiFi.reconnect() here — loop() handles it with back-off
            break;

        case ARDUINO_EVENT_WIFI_STA_GOT_IP:
            wifiFailCount = 0;
            Serial.print("[WIFI] Connected  IP: ");
            Serial.println(WiFi.localIP());
            break;

        case ARDUINO_EVENT_WIFI_STA_CONNECTED:
            Serial.println("[WIFI] Associated to AP");
            break;

        default:
            break;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  SETUP
// ─────────────────────────────────────────────────────────────────────────────
void setup()
{
    Serial.begin(115200);   // was 9600 — too slow to see debug output in real time
    Serial.println("\nESP32-CAM Attendance Camera Starting…");

    // ── Camera init ──────────────────────────────────────────────────────────
    camera_config_t cam;
    memset(&cam, 0, sizeof(cam));

    cam.ledc_channel = LEDC_CHANNEL_0;
    cam.ledc_timer   = LEDC_TIMER_0;
    cam.pin_d0       = Y2_GPIO_NUM;
    cam.pin_d1       = Y3_GPIO_NUM;
    cam.pin_d2       = Y4_GPIO_NUM;
    cam.pin_d3       = Y5_GPIO_NUM;
    cam.pin_d4       = Y6_GPIO_NUM;
    cam.pin_d5       = Y7_GPIO_NUM;
    cam.pin_d6       = Y8_GPIO_NUM;
    cam.pin_d7       = Y9_GPIO_NUM;
    cam.pin_xclk     = XCLK_GPIO_NUM;
    cam.pin_pclk     = PCLK_GPIO_NUM;
    cam.pin_vsync    = VSYNC_GPIO_NUM;
    cam.pin_href     = HREF_GPIO_NUM;
    cam.pin_sccb_sda = SIOD_GPIO_NUM;
    cam.pin_sccb_scl = SIOC_GPIO_NUM;
    cam.pin_pwdn     = PWDN_GPIO_NUM;
    cam.pin_reset    = RESET_GPIO_NUM;
    cam.xclk_freq_hz = CAM_XCLK_FREQ;
    cam.pixel_format = PIXFORMAT_JPEG;

    if (psramFound()) {
        cam.frame_size   = CAM_FRAME_SIZE;
        cam.jpeg_quality = CAM_JPEG_QUALITY;
        cam.fb_count     = CAM_FB_COUNT;
        cam.fb_location  = CAMERA_FB_IN_PSRAM;
        cam.grab_mode    = CAMERA_GRAB_LATEST;
        Serial.println("[OK] PSRAM found — using double-buffer + CAMERA_GRAB_LATEST");
    } else {
        cam.frame_size   = FRAMESIZE_QVGA;
        cam.jpeg_quality = 20;           // slightly lower quality without PSRAM
        cam.fb_count     = 1;
        cam.fb_location  = CAMERA_FB_IN_DRAM;
        cam.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
        Serial.println("[WARN] No PSRAM — single buffer, lower quality");
    }

    esp_err_t err = esp_camera_init(&cam);
    if (err != ESP_OK) {
        Serial.printf("[ERR] Camera init 0x%x — restarting\n", err);
        delay(2000);
        ESP.restart();
    }
    Serial.println("[OK] Camera initialised");

    // ── Sensor fine-tuning ───────────────────────────────────────────────────
    sensor_t* s = esp_camera_sensor_get();
    if (s) {
        s->set_framesize(s,     CAM_FRAME_SIZE);
        s->set_quality(s,       CAM_JPEG_QUALITY);
        s->set_exposure_ctrl(s, 1);   // auto-exposure ON
        s->set_aec2(s,          0);   // AEC2 DSP OFF (adds latency)
        s->set_ae_level(s,      0);   // neutral EV
        s->set_whitebal(s,      1);   // auto white balance ON
        s->set_awb_gain(s,      1);
        s->set_gain_ctrl(s,     1);   // auto gain ON
        s->set_agc_gain(s,      0);
        s->set_bpc(s,           0);   // bad-pixel correction OFF
        s->set_wpc(s,           1);
        s->set_raw_gma(s,       1);   // gamma correction ON
        s->set_lenc(s,          0);   // lens correction OFF (saves CPU)
        s->set_hmirror(s,       0);
        s->set_vflip(s,         0);
        /*
          brightness +1 helps emotion detection in indoor lighting
          (DeepFace struggles with under-exposed faces).
        */
        s->set_brightness(s,    1);
        s->set_contrast(s,      1);
        s->set_saturation(s,    0);
        Serial.println("[OK] Sensor tuned");
    }

    // ── WiFi ─────────────────────────────────────────────────────────────────
    WiFi.onEvent(WiFiEvent);
    WiFi.mode(WIFI_STA);

    // Disable modem sleep — prevents radio from powering down mid-stream
    WiFi.setSleep(false);
    esp_wifi_set_ps(WIFI_PS_NONE);     // completely disable power-save

    WiFi.setAutoReconnect(false);      // we handle reconnect manually in loop()
    WiFi.persistent(false);            // don't write to flash on every reconnect

    // Maximum TX power (19.5 dBm) — strongest signal, most stable
    WiFi.setTxPower(WIFI_POWER_19_5dBm);

    WiFi.begin(ssid, password);

    Serial.print("Connecting to WiFi");
    uint8_t tries = 0;
    while (WiFi.status() != WL_CONNECTED && tries < 40) {
        delay(500);
        Serial.print(".");
        tries++;
    }

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("\n[ERR] WiFi initial connect failed — restarting");
        delay(1000);
        ESP.restart();
    }

    Serial.println("\n[OK] WiFi connected");
    Serial.print  ("  IP     : ");  Serial.println(WiFi.localIP());
    Serial.print  ("  Signal : ");  Serial.print(WiFi.RSSI()); Serial.println(" dBm");

    // ── HTTP server ───────────────────────────────────────────────────────────
    startCameraServer();

    Serial.println();
    Serial.println("================================================");
    Serial.println("  CAMERA STREAM READY");
    Serial.println("================================================");
    Serial.printf ("  Stream  : http://%s/stream\n",  WiFi.localIP().toString().c_str());
    Serial.printf ("  Info    : http://%s/info\n",    WiFi.localIP().toString().c_str());
    Serial.printf ("  Health  : http://%s/healthz\n", WiFi.localIP().toString().c_str());
    Serial.printf ("  Quality : %d  (QVGA, ~12 fps cap)\n", CAM_JPEG_QUALITY);
    Serial.println("================================================");
}

// ─────────────────────────────────────────────────────────────────────────────
//  LOOP  — WiFi watchdog with exponential back-off
// ─────────────────────────────────────────────────────────────────────────────
void loop()
{
    uint32_t now = millis();

    // Check WiFi every 3 s (was 5 s — quicker recovery)
    if (now - lastWifiCheck >= 3000)
    {
        lastWifiCheck = now;

        if (WiFi.status() != WL_CONNECTED)
        {
            wifiFailCount++;
            Serial.printf("[WIFI] Lost connection (attempt %d)\n", wifiFailCount);

            // Stop HTTP server so its open sockets don't block reconnect
            if (stream_httpd) {
                httpd_stop(stream_httpd);
                stream_httpd = NULL;
                Serial.println("[WIFI] HTTP server stopped for reconnect");
            }

            WiFi.disconnect(true);
            delay(500);

            WiFi.begin(ssid, password);

            // Wait up to 15 s for reconnect
            uint32_t start = millis();
            while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
                delay(300);
                Serial.print(".");
            }
            Serial.println();

            if (WiFi.status() == WL_CONNECTED)
            {
                Serial.println("[WIFI] Reconnected!");
                Serial.print  ("[WIFI] IP: "); Serial.println(WiFi.localIP());
                wifiFailCount = 0;
                // Restart HTTP server
                startCameraServer();
            }
            else
            {
                Serial.printf("[WIFI] Reconnect failed (%d times)\n", wifiFailCount);
                if (wifiFailCount >= 5) {
                    Serial.println("[WIFI] Too many failures — full restart");
                    delay(1000);
                    ESP.restart();
                }
                // Exponential back-off: 5 s, 10 s, 20 s, 40 s, then restart
                uint32_t backoff = min((uint32_t)(5000 * (1 << min(wifiFailCount - 1, 3))), (uint32_t)40000);
                Serial.printf("[WIFI] Waiting %u ms before retry\n", backoff);
                delay(backoff);
            }
        }
        else
        {
            // Connected — print signal strength occasionally
            static uint8_t rssiTick = 0;
            if (++rssiTick >= 20) {    // every ~60 s
                rssiTick = 0;
                int rssi = WiFi.RSSI();
                Serial.printf("[WIFI] OK  RSSI=%d dBm  heap=%u\n",
                              rssi, esp_get_free_heap_size());
                if (rssi < -80) {
                    Serial.println("[WIFI] WARN: weak signal (<-80 dBm) — may disconnect");
                }
            }
        }
    }

    // Feed watchdog / allow other tasks
    delay(50);
}
