#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include "esp_timer.h"

// LwIP headers — required for setsockopt on ESP32 Arduino core 3.x
#include "lwip/sockets.h"
#include "lwip/netdb.h"

// ===========================
// WIFI CONFIG
// ===========================

const char* ssid     = "Himanshu";
const char* password = "himanshu";

// ===========================
// CAMERA MODEL — AI THINKER
// ===========================

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

// ===========================
// STREAM SETTINGS
// ===========================

#define CAM_FRAME_SIZE   FRAMESIZE_QVGA   // 320x240 — best for face recognition
#define CAM_JPEG_QUALITY 12               // 10-15; lower = bigger file = slower
#define CAM_FB_COUNT     2                // double-buffer
#define CAM_XCLK_FREQ    20000000         // 20 MHz

// ===========================
// HTTP SERVER
// ===========================

httpd_handle_t stream_httpd = NULL;

static const char* STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=gc0p4Jq0M2Yt08jU534c0p";
static const char* STREAM_BOUNDARY =
    "\r\n--gc0p4Jq0M2Yt08jU534c0p\r\n";
static const char* STREAM_PART =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// ===========================
// STREAM HANDLER
// ===========================

static esp_err_t stream_handler(httpd_req_t* req)
{
    camera_fb_t* fb  = NULL;
    esp_err_t    res = ESP_OK;
    char         part_buf[128];

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store, no-cache");
    httpd_resp_set_hdr(req, "Pragma", "no-cache");

    res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;

    // Disable Nagle — send every chunk immediately
    int sock = httpd_req_to_sockfd(req);
    int nodelay = 1;
    lwip_setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

    // Send timeout — stalled client won't block camera loop forever
    struct timeval tv;
    tv.tv_sec  = 3;
    tv.tv_usec = 0;
    lwip_setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    while (true)
    {
        fb = esp_camera_fb_get();

        if (!fb)
        {
            Serial.println("[WARN] Camera capture failed — skipping frame");
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        size_t hlen = snprintf(part_buf, sizeof(part_buf),
                               STREAM_PART, fb->len);

        res = httpd_resp_send_chunk(req, STREAM_BOUNDARY,
                                    strlen(STREAM_BOUNDARY));

        if (res == ESP_OK)
            res = httpd_resp_send_chunk(req, part_buf, hlen);

        if (res == ESP_OK)
            res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);

        esp_camera_fb_return(fb);
        fb = NULL;

        if (res != ESP_OK)
        {
            Serial.println("[INFO] Client disconnected");
            break;
        }

        taskYIELD();
    }

    return res;
}

// ===========================
// INFO ENDPOINT
// ===========================

static esp_err_t info_handler(httpd_req_t* req)
{
    char buf[256];
    snprintf(buf, sizeof(buf),
             "{\"psram\":%s,\"frame\":\"QVGA 320x240\",\"quality\":%d,\"ip\":\"%s\"}",
             psramFound() ? "true" : "false",
             CAM_JPEG_QUALITY,
             WiFi.localIP().toString().c_str());

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, buf, strlen(buf));
}

// ===========================
// START SERVER
// ===========================

void startCameraServer()
{
    httpd_config_t config  = HTTPD_DEFAULT_CONFIG();
    config.stack_size      = 8192;
    config.server_port     = 80;
    config.ctrl_port       = 32768;
    config.max_open_sockets = 5;
    config.task_priority   = 5;
    config.core_id         = 0;

    httpd_uri_t stream_uri = {
        .uri      = "/stream",
        .method   = HTTP_GET,
        .handler  = stream_handler,
        .user_ctx = NULL
    };

    httpd_uri_t info_uri = {
        .uri      = "/info",
        .method   = HTTP_GET,
        .handler  = info_handler,
        .user_ctx = NULL
    };

    if (httpd_start(&stream_httpd, &config) == ESP_OK)
    {
        httpd_register_uri_handler(stream_httpd, &stream_uri);
        httpd_register_uri_handler(stream_httpd, &info_uri);
        Serial.println("[OK] HTTP server started");
    }
    else
    {
        Serial.println("[ERR] HTTP server failed to start");
    }
}

// ===========================
// SETUP
// ===========================

void setup()
{
    Serial.begin(115200);
    Serial.println("\nESP32-CAM Optimized Stream Starting...");

    // ---------- Camera Config ----------
    camera_config_t config;

    config.ledc_channel    = LEDC_CHANNEL_0;
    config.ledc_timer      = LEDC_TIMER_0;
    config.pin_d0          = Y2_GPIO_NUM;
    config.pin_d1          = Y3_GPIO_NUM;
    config.pin_d2          = Y4_GPIO_NUM;
    config.pin_d3          = Y5_GPIO_NUM;
    config.pin_d4          = Y6_GPIO_NUM;
    config.pin_d5          = Y7_GPIO_NUM;
    config.pin_d6          = Y8_GPIO_NUM;
    config.pin_d7          = Y9_GPIO_NUM;
    config.pin_xclk        = XCLK_GPIO_NUM;
    config.pin_pclk        = PCLK_GPIO_NUM;
    config.pin_vsync       = VSYNC_GPIO_NUM;
    config.pin_href        = HREF_GPIO_NUM;
    config.pin_sccb_sda    = SIOD_GPIO_NUM;
    config.pin_sccb_scl    = SIOC_GPIO_NUM;
    config.pin_pwdn        = PWDN_GPIO_NUM;
    config.pin_reset       = RESET_GPIO_NUM;
    config.xclk_freq_hz    = CAM_XCLK_FREQ;
    config.pixel_format    = PIXFORMAT_JPEG;

    if (psramFound())
    {
        config.frame_size   = CAM_FRAME_SIZE;
        config.jpeg_quality = CAM_JPEG_QUALITY;
        config.fb_count     = CAM_FB_COUNT;
        config.fb_location  = CAMERA_FB_IN_PSRAM;
        config.grab_mode    = CAMERA_GRAB_LATEST;
        Serial.println("[OK] PSRAM found — double-buffer enabled");
    }
    else
    {
        config.frame_size   = FRAMESIZE_QVGA;
        config.jpeg_quality = 15;
        config.fb_count     = 1;
        config.fb_location  = CAMERA_FB_IN_DRAM;
        config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
        Serial.println("[WARN] No PSRAM — single buffer mode");
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK)
    {
        Serial.printf("[ERR] Camera init failed: 0x%x\n", err);
        return;
    }

    // ---------- Sensor Fine-Tuning ----------
    sensor_t* s = esp_camera_sensor_get();
    if (s)
    {
        s->set_framesize(s,      CAM_FRAME_SIZE);
        s->set_quality(s,        CAM_JPEG_QUALITY);
        s->set_exposure_ctrl(s,  1);   // auto exposure ON
        s->set_aec2(s,           0);   // disable AEC2 DSP (adds latency)
        s->set_ae_level(s,       0);   // neutral exposure
        s->set_whitebal(s,       1);   // auto white balance ON
        s->set_awb_gain(s,       1);
        s->set_gain_ctrl(s,      1);   // auto gain ON
        s->set_agc_gain(s,       0);
        s->set_bpc(s,            0);   // bad pixel correction OFF
        s->set_wpc(s,            1);
        s->set_raw_gma(s,        1);   // gamma ON
        s->set_lenc(s,           0);   // lens correction OFF
        s->set_hmirror(s,        0);
        s->set_vflip(s,          0);
        Serial.println("[OK] Sensor tuned");
    }

    // ---------- WiFi ----------
    WiFi.setSleep(false);                      // CRITICAL: no modem sleep
    WiFi.setTxPower(WIFI_POWER_19_5dBm);
    WiFi.begin(ssid, password);

    Serial.print("Connecting to WiFi");
    uint8_t tries = 0;
    while (WiFi.status() != WL_CONNECTED && tries < 40)
    {
        delay(500);
        Serial.print(".");
        tries++;
    }

    if (WiFi.status() != WL_CONNECTED)
    {
        Serial.println("\n[ERR] WiFi failed — restarting");
        ESP.restart();
    }

    Serial.println("\n[OK] WiFi connected");

    startCameraServer();

    Serial.println();
    Serial.println("================================================");
    Serial.println("  CAMERA STREAM READY");
    Serial.println("================================================");
    Serial.print  ("  Stream : http://");
    Serial.print  (WiFi.localIP());
    Serial.println("/stream");
    Serial.print  ("  Info   : http://");
    Serial.print  (WiFi.localIP());
    Serial.println("/info");
    Serial.println("================================================");
}

// ===========================
// LOOP
// ===========================

void loop()
{
    vTaskDelay(pdMS_TO_TICKS(1000));
}

