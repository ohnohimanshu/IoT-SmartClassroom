
#include "esp_camera.h"
#include <WiFi.h>
#include "esp_https_server.h"   // replaces esp_http_server — provides TLS/HTTPS
#include "esp_timer.h"
#include "esp_wifi.h"
#include "esp_system.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"

// ─── Self-signed TLS certificate & private key (PEM format) ──────────────────
//
//  Generate once on your PC with OpenSSL:
//
//    openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem \
//        -days 3650 -nodes -subj "/CN=esp32cam"
//
//  Then paste the contents of cert.pem into SERVER_CERT_PEM
//  and the contents of key.pem into SERVER_KEY_PEM below.
//
//  Because the cert is self-signed, browsers will show a security warning.
//  Click "Advanced → Proceed" (Chrome) or "Accept the Risk" (Firefox) once.
//  For Python/requests clients use verify=False; for curl use -k.
//
//  !! NEVER expose these keys publicly — this is a local-network device. !!
// ─────────────────────────────────────────────────────────────────────────────
static const char SERVER_CERT_PEM[] =
    "-----BEGIN CERTIFICATE-----\n"
    "PASTE YOUR cert.pem CONTENTS HERE, ONE LINE PER \\n\n"
    "-----END CERTIFICATE-----\n";

static const char SERVER_KEY_PEM[] =
    "-----BEGIN PRIVATE KEY-----\n"
    "PASTE YOUR key.pem CONTENTS HERE, ONE LINE PER \\n\n"
    "-----END PRIVATE KEY-----\n";

// ─── WiFi credentials ────────────────────────────────────────────────────────
const char* ssid     = "DEIWF";
const char* password = "";

// ─── Camera pin map: AI Thinker ──────────────────────────────────────────────
#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM      5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22

// ─── LED pins (AI Thinker) ────────────────────────────────────────────────────
#define LED_STATUS_PIN  33   // small red LED, active-LOW
#define LED_FLASH_PIN    4   // bright white flash, active-HIGH — keep OFF

// ─── Stream tuning ───────────────────────────────────────────────────────────
#define CAM_FRAME_SIZE    FRAMESIZE_QVGA   // 320×240
#define CAM_XCLK_FREQ     20000000

// Quality & frame-rate differ by PSRAM presence — set in setup()
// Without PSRAM: quality=25 (~9 KB/frame), interval=150 ms (~6 fps)
// With    PSRAM: quality=18 (~14 KB/frame), interval=80  ms (~12 fps)
static int    camJpegQuality = 25;
static uint32_t frameMinMs   = 150;

// ─── MJPEG boundary strings ───────────────────────────────────────────────────
static const char* STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=gc0p4Jq0M2Yt08jU534c0p";
static const char* STREAM_BOUNDARY =
    "\r\n--gc0p4Jq0M2Yt08jU534c0p\r\n";
static const char* STREAM_PART =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// ─── State ───────────────────────────────────────────────────────────────────
httpd_handle_t  stream_httpd  = NULL;
static uint8_t  wifiFailCount = 0;
static uint32_t lastWifiCheck = 0;
static uint32_t lastKeepAlive = 0;
static TaskHandle_t blinkTaskHandle = NULL;


// ─────────────────────────────────────────────────────────────────────────────
//  WIFI SETTINGS — applied after every connect / reconnect
//  FIX #1: esp_wifi_set_ps must be called AFTER the connection is up,
//  not before WiFi.begin(). The stack reinitialises PS mode on connect.
// ─────────────────────────────────────────────────────────────────────────────
static void wifi_apply_settings()
{
    // Disable ALL power-save modes
    esp_wifi_set_ps(WIFI_PS_NONE);

    // Protocol: 802.11b/g/n — same as AP default; fixes mismatched rates
    esp_wifi_set_protocol(WIFI_IF_STA,
        WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);

    // Max TX power: reduces range slightly vs 19.5 dBm but cuts peak
    // current draw by ~60 mA — prevents voltage sag on weak USB cables
    WiFi.setTxPower(WIFI_POWER_17dBm);

    Serial.println("[WIFI] Settings applied: PS=NONE, TxPwr=17dBm, 11b/g/n");
}


// ─────────────────────────────────────────────────────────────────────────────
//  LED BLINK TASK
// ─────────────────────────────────────────────────────────────────────────────
void blinkTask(void* param)
{
    int blinks = (int)(intptr_t)param;
    for (int i = 0; i < blinks; i++) {
        digitalWrite(LED_STATUS_PIN, LOW);       // ON  (active LOW)
        vTaskDelay(pdMS_TO_TICKS(200));
        digitalWrite(LED_STATUS_PIN, HIGH);      // OFF
        if (i < blinks - 1)
            vTaskDelay(pdMS_TO_TICKS(200));
    }
    blinkTaskHandle = NULL;
    vTaskDelete(NULL);
}

void triggerBlink(int times = 2)
{
    if (blinkTaskHandle != NULL) return;
    xTaskCreate(blinkTask, "blink", 1024,
                (void*)(intptr_t)times, 1, &blinkTaskHandle);
}


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
    httpd_resp_set_hdr(req, "Connection", "close");
    res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;

    int sock = httpd_req_to_sockfd(req);

    // TCP_NODELAY: flush each chunk immediately
    int nodelay = 1;
    lwip_setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

    // Send timeout 4 s: abort stalled clients
    struct timeval tv = { .tv_sec = 4, .tv_usec = 0 };
    lwip_setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    // Send buffer 16 KB (halved from 32 KB) — smaller buffer means the
    // send call returns sooner, giving WiFi tasks more CPU time
    int sndbuf = 16384;
    lwip_setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

    uint32_t lastFrame = 0;

    while (true)
    {
        // FIX #4: vTaskDelay(1) instead of taskYIELD() — guarantees WiFi/
        // lwIP tasks get a full FreeRTOS tick, not just a voluntary yield.
        vTaskDelay(1);

        uint32_t now  = (uint32_t)(esp_timer_get_time() / 1000ULL);
        int32_t  wait = (int32_t)frameMinMs - (int32_t)(now - lastFrame);
        if (wait > 1)
            vTaskDelay(pdMS_TO_TICKS(wait));

        fb = esp_camera_fb_get();
        if (!fb) {
            vTaskDelay(pdMS_TO_TICKS(10));
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
            Serial.println("[STREAM] Client disconnected");
            break;
        }

        // Bail early on severe heap pressure (no PSRAM)
        if (esp_get_free_heap_size() < 8000) {
            Serial.printf("[WARN] Low heap (%u B) — closing stream\n",
                          esp_get_free_heap_size());
            break;
        }
    }
    return res;
}


// ─────────────────────────────────────────────────────────────────────────────
//  BLINK ENDPOINT  — POST /blink
// ─────────────────────────────────────────────────────────────────────────────
static esp_err_t blink_handler(httpd_req_t* req)
{
    char query[32] = {0};
    int  times = 2;
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char param[8] = {0};
        if (httpd_query_key_value(query, "times", param, sizeof(param)) == ESP_OK) {
            int n = atoi(param);
            if (n >= 1 && n <= 5) times = n;
        }
    }
    triggerBlink(times);
    Serial.printf("[BLINK] %d blink(s) triggered\n", times);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, "{\"status\":\"blinking\"}", 20);
}


// ─────────────────────────────────────────────────────────────────────────────
//  INFO / HEALTH ENDPOINTS
// ─────────────────────────────────────────────────────────────────────────────
static esp_err_t info_handler(httpd_req_t* req)
{
    char buf[300];
    snprintf(buf, sizeof(buf),
        "{\"psram\":%s,\"frame\":\"QVGA\",\"quality\":%d,"
        "\"fps_cap\":%d,\"ip\":\"%s\",\"free_heap\":%u,\"rssi\":%d}",
        psramFound() ? "true" : "false",
        camJpegQuality,
        (int)(1000 / frameMinMs),
        WiFi.localIP().toString().c_str(),
        (unsigned)esp_get_free_heap_size(),
        WiFi.RSSI()
    );
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, buf, strlen(buf));
}

static esp_err_t health_handler(httpd_req_t* req)
{
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, "{\"status\":\"ok\"}", 15);
}


// ─────────────────────────────────────────────────────────────────────────────
//  HTTPS SERVER  (replaces plain HTTP server)
//  Uses esp_https_server which wraps esp_http_server with mbedTLS.
//  All handler code (stream_handler, blink_handler, etc.) is unchanged —
//  only the startup function and the handle type differ.
// ─────────────────────────────────────────────────────────────────────────────
void startCameraServer()
{
    // ── Base HTTP config (same tuning as before) ──────────────────────────────
    httpd_config_t config    = HTTPD_DEFAULT_CONFIG();
    config.stack_size        = 10240;  // bumped: TLS handshake needs extra stack
    config.server_port       = 443;    // standard HTTPS port
    config.ctrl_port         = 32768;
    config.max_open_sockets  = 4;      // 1 fewer: TLS sessions cost ~30 KB each
    config.task_priority     = 4;
    config.core_id           = 1;
    config.recv_wait_timeout = 5;
    config.send_wait_timeout = 5;
    config.lru_purge_enable  = true;

    // ── TLS config — point at the PEM strings defined at the top ─────────────
    httpd_ssl_config_t ssl_config        = HTTPD_SSL_CONFIG_DEFAULT();
    ssl_config.httpd                     = config;
    ssl_config.servercert                = (const uint8_t*)SERVER_CERT_PEM;
    ssl_config.servercert_len            = sizeof(SERVER_CERT_PEM);
    ssl_config.prvtkey_pem               = (const uint8_t*)SERVER_KEY_PEM;
    ssl_config.prvtkey_len               = sizeof(SERVER_KEY_PEM);
    // Port is taken from ssl_config.httpd.server_port (443) set above.
    // ssl_config.port_secure defaults to 443 anyway, kept explicit for clarity:
    ssl_config.port_secure               = 443;

    httpd_uri_t uris[] = {
        { "/stream",  HTTP_GET,  stream_handler, NULL },
        { "/info",    HTTP_GET,  info_handler,   NULL },
        { "/healthz", HTTP_GET,  health_handler, NULL },
        { "/blink",   HTTP_POST, blink_handler,  NULL },
    };

    if (httpd_ssl_start(&stream_httpd, &ssl_config) == ESP_OK) {
        for (auto& u : uris)
            httpd_register_uri_handler(stream_httpd, &u);
        Serial.println("[OK] HTTPS server started on port 443");
    } else {
        Serial.println("[ERR] HTTPS server failed — check cert/key PEM strings — restarting");
        delay(2000);
        ESP.restart();
    }
}


// ─────────────────────────────────────────────────────────────────────────────
//  WiFi EVENT HANDLER
//  FIX #1: wifi_apply_settings() called on GOT_IP, not in setup() before
//  WiFi.begin(). This guarantees PS=NONE survives the stack initialisation.
// ─────────────────────────────────────────────────────────────────────────────
void WiFiEvent(WiFiEvent_t event)
{
    switch (event)
    {
        case ARDUINO_EVENT_WIFI_STA_GOT_IP:
            wifiFailCount = 0;
            Serial.printf("[WIFI] Got IP: %s  RSSI: %d dBm\n",
                          WiFi.localIP().toString().c_str(), WiFi.RSSI());
            wifi_apply_settings();   // ← FIX: apply AFTER stack is up
            break;

        case ARDUINO_EVENT_WIFI_STA_CONNECTED:
            Serial.println("[WIFI] Associated to AP");
            break;

        case ARDUINO_EVENT_WIFI_STA_DISCONNECTED:
            Serial.printf("[WIFI] Disconnected (fail #%d)\n", wifiFailCount + 1);
            break;

        default:
            break;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
//  WIFI KEEP-ALIVE
//  FIX #5: called from loop() every 10 s — re-applies WIFI_PS_NONE and
//  prints diagnostics. Some access points re-negotiate PS mode over time.
// ─────────────────────────────────────────────────────────────────────────────
static void wifi_keep_alive()
{
    if (WiFi.status() != WL_CONNECTED) return;

    // Re-assert power-save disabled (some APs re-enable it silently)
    esp_wifi_set_ps(WIFI_PS_NONE);

    int rssi = WiFi.RSSI();
    Serial.printf("[WIFI] Alive — RSSI=%d dBm  heap=%u B  PS=NONE re-asserted\n",
                  rssi, esp_get_free_heap_size());

    if (rssi < -80)
        Serial.println("[WIFI] WARNING: signal weak (<-80 dBm) — consider moving closer to AP");
}


// ─────────────────────────────────────────────────────────────────────────────
//  SETUP
// ─────────────────────────────────────────────────────────────────────────────
void setup()
{
    // Disable brownout reset — ESP32-CAM draws ~400-500 mA during WiFi TX;
    // weak USB cables sag below 3.0 V and trigger a spurious reset.
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

    Serial.begin(115200);
    Serial.println("\n\nESP32-CAM Attendance Camera v5 (HTTPS) Starting…");
    Serial.println("[PWR] Brownout detector disabled — use 5V/2A supply");

    // LEDs
    pinMode(LED_STATUS_PIN, OUTPUT);
    digitalWrite(LED_STATUS_PIN, HIGH);  // OFF
    pinMode(LED_FLASH_PIN, OUTPUT);
    digitalWrite(LED_FLASH_PIN, LOW);    // OFF — never use flash while streaming

    triggerBlink(1);
    delay(600);

    // ── Camera ───────────────────────────────────────────────────────────────
    camera_config_t cam;
    memset(&cam, 0, sizeof(cam));
    cam.ledc_channel = LEDC_CHANNEL_0;
    cam.ledc_timer   = LEDC_TIMER_0;
    cam.pin_d0  = Y2_GPIO_NUM;   cam.pin_d1  = Y3_GPIO_NUM;
    cam.pin_d2  = Y4_GPIO_NUM;   cam.pin_d3  = Y5_GPIO_NUM;
    cam.pin_d4  = Y6_GPIO_NUM;   cam.pin_d5  = Y7_GPIO_NUM;
    cam.pin_d6  = Y8_GPIO_NUM;   cam.pin_d7  = Y9_GPIO_NUM;
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
    cam.frame_size   = CAM_FRAME_SIZE;

    if (psramFound()) {
        // FIX #7 / PSRAM path: high quality, fast, double-buffered
        camJpegQuality  = 18;
        frameMinMs      = 80;
        cam.jpeg_quality = camJpegQuality;
        cam.fb_count     = 2;
        cam.fb_location  = CAMERA_FB_IN_PSRAM;
        cam.grab_mode    = CAMERA_GRAB_LATEST;
        Serial.println("[CAM] PSRAM: quality=18, 12fps, double-buffer");
    } else {
        // FIX #2 / No-PSRAM: lower quality (smaller frames) + slower rate
        // quality=25 → ~9 KB/frame vs ~15 KB at 18 — halves TCP pressure
        // 150 ms interval → ~6 fps — still fine for face detection
        camJpegQuality  = 25;
        frameMinMs      = 150;
        cam.jpeg_quality = camJpegQuality;
        cam.fb_count     = 1;
        cam.fb_location  = CAMERA_FB_IN_DRAM;
        cam.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
        Serial.println("[CAM] No PSRAM: quality=25, 6fps, single-buffer");
        Serial.println("[CAM] TIP: board with PSRAM (e.g. AI-Thinker w/ 8MB) gives better stream");
    }

    esp_err_t err = esp_camera_init(&cam);
    if (err != ESP_OK) {
        Serial.printf("[ERR] Camera init failed: 0x%x — restarting\n", err);
        delay(2000); ESP.restart();
    }
    Serial.println("[OK] Camera initialised");

    sensor_t* s = esp_camera_sensor_get();
    if (s) {
        s->set_framesize(s,     CAM_FRAME_SIZE);
        s->set_quality(s,       camJpegQuality);
        s->set_exposure_ctrl(s, 1);   // auto-exposure
        s->set_aec2(s,          0);
        s->set_ae_level(s,      0);
        s->set_whitebal(s,      1);   // auto white balance
        s->set_awb_gain(s,      1);
        s->set_gain_ctrl(s,     1);   // auto gain
        s->set_agc_gain(s,      0);
        s->set_bpc(s,           0);
        s->set_wpc(s,           1);
        s->set_raw_gma(s,       1);
        s->set_lenc(s,          0);
        s->set_hmirror(s,       0);
        s->set_vflip(s,         0);
        s->set_brightness(s,    1);   // +1 brightness helps indoor detection
        s->set_contrast(s,      1);
        s->set_saturation(s,    0);
        Serial.println("[OK] Sensor tuned");
    }

    // ── WiFi ─────────────────────────────────────────────────────────────────
    WiFi.onEvent(WiFiEvent);
    WiFi.mode(WIFI_STA);

    // FIX #1: Do NOT call wifi_apply_settings() here — the stack
    // will overwrite it. It is now called from WiFiEvent(GOT_IP).
    WiFi.setSleep(false);           // belt-and-suspenders pre-connect hint
    WiFi.setAutoReconnect(false);   // we manage reconnect manually
    WiFi.persistent(false);         // don't write flash every connect

    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");

    uint8_t tries = 0;
    while (WiFi.status() != WL_CONNECTED && tries < 40) {
        delay(500); Serial.print("."); tries++;
    }
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("\n[ERR] WiFi failed — restarting");
        delay(1000); ESP.restart();
    }
    Serial.println("\n[OK] WiFi connected");

    startCameraServer();
    triggerBlink(2);   // 2 blinks = ready

    Serial.println("================================================");
    Serial.println("  CAMERA STREAM READY  (v5)");
    Serial.println("================================================");
    Serial.printf ("  Stream  : https://%s/stream\n",  WiFi.localIP().toString().c_str());
    Serial.printf ("  Health  : https://%s/healthz\n", WiFi.localIP().toString().c_str());
    Serial.printf ("  Blink   : POST https://%s/blink\n", WiFi.localIP().toString().c_str());
    Serial.printf ("  Mode    : %s — quality=%d fps=~%d\n",
        psramFound() ? "PSRAM" : "No-PSRAM",
        camJpegQuality, (int)(1000/frameMinMs));
    Serial.println("================================================");
}


// ─────────────────────────────────────────────────────────────────────────────
//  LOOP
// ─────────────────────────────────────────────────────────────────────────────
void loop()
{
    uint32_t now = millis();

    // ── Keep-alive: re-assert PS=NONE every 10 s ─────────────────────────────
    // FIX #5: prevents AP from silently re-enabling power-save
    if (now - lastKeepAlive >= 10000) {
        lastKeepAlive = now;
        wifi_keep_alive();
    }

    // ── WiFi watchdog every 3 s ───────────────────────────────────────────────
    if (now - lastWifiCheck >= 3000) {
        lastWifiCheck = now;

        if (WiFi.status() != WL_CONNECTED) {
            wifiFailCount++;
            Serial.printf("[WIFI] Lost (attempt %d)\n", wifiFailCount);

            if (stream_httpd) {
                httpd_stop(stream_httpd);
                stream_httpd = NULL;
            }

            WiFi.disconnect(true);
            delay(300);
            WiFi.begin(ssid, password);   // GOT_IP event will call wifi_apply_settings()

            uint32_t t0 = millis();
            while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
                delay(300); Serial.print(".");
            }
            Serial.println();

            if (WiFi.status() == WL_CONNECTED) {
                Serial.println("[WIFI] Reconnected");
                wifiFailCount = 0;
                startCameraServer();
            } else {
                Serial.printf("[WIFI] Failed (%d)\n", wifiFailCount);
                if (wifiFailCount >= 5) {
                    Serial.println("[WIFI] Too many failures — restart");
                    delay(500); ESP.restart();
                }
                uint32_t backoff = min(
    (uint32_t)(5000u * (1u << min((uint32_t)(wifiFailCount - 1), (uint32_t)3))),
    (uint32_t)40000
);
            }
        }
    }

    delay(50);
}
