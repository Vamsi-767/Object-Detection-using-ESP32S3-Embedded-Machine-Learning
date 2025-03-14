#include "esp_camera.h"
#include <WiFi.h>
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "fb_gfx.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "NeuralNetwork.h"
#include "esp_http_server.h"
#define SETUP_AP 1 
#define CAMERA_MODEL_XIAO_ESP32S3
#include "camera_pins.h"
#define INPUT_W 320
#define INPUT_H 320
#define LED_BUILT_IN 21
#define CONF_THRESHOLD 0.5
#define BOTTLE_CLASS_INDEX 0 
#define DEBUG_TFLITE 0
#define GREEN 0x07E0
NeuralNetwork *g_nn;
const char* ssid = "ESP_3143153";
const char* password = "123456789";

#define PART_BOUNDARY "123456789000000000000987654321"
#define _STREAM_CONTENT_TYPE "multipart/x-mixed-replace;boundary=" PART_BOUNDARY
#define _STREAM_BOUNDARY "\r\n--" PART_BOUNDARY "\r\n"
#define _STREAM_PART "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n"

httpd_handle_t stream_httpd = NULL;

struct Box {
    float x_center, y_center, width, height, confidence;
    int class_id;};
inline void convertRGB565toRGB888(uint16_t color, uint8_t &r, uint8_t &g, uint8_t &b) {
    r = ((color >> 11) & 0x1F) << 3;    
    g = ((color >> 5) & 0x3F) << 2;    
    b = (color & 0x1F) << 3;  }          

inline void calculateCropArea(int img_width, int img_height, int crop_width, int crop_height, int &startx, int &starty) {
    startx = (img_width - crop_width) / 2;
    starty = (img_height - crop_height) / 2;}
// Process Image and Populate Tensor
void ProcessImage(camera_fb_t *fb, TfLiteTensor *input_tensor, float scale, int zero_point) {
    assert(fb->format == PIXFORMAT_RGB565);
    int post = 0;
    int startx, starty;
    calculateCropArea(fb->width, fb->height, INPUT_W, INPUT_H, startx, starty);
    uint8_t r, g, b;
    for (int y = 0; y < INPUT_H; ++y) {
        for (int x = 0; x < INPUT_W; ++x) {
            int index = (starty + y) * fb->width + (startx + x);
            uint16_t pixel = ((uint16_t *)fb->buf)[index];
            // Convert RGB565 to RGB888
            convertRGB565toRGB888(pixel, r, g, b);
            if (input_tensor->type == kTfLiteUInt8) {
                uint8_t *data = input_tensor->data.uint8;
                data[post * 3 + 0] = (uint8_t)((r / 255.0f) / scale + zero_point);
                data[post * 3 + 1] = (uint8_t)((g / 255.0f) / scale + zero_point);
                data[post * 3 + 2] = (uint8_t)((b / 255.0f) / scale + zero_point);
            } else {
                float *data = input_tensor->data.f;
                data[post * 3 + 0] = r / 255.0f;
                data[post * 3 + 1] = g / 255.0f;
                data[post * 3 + 2] = b / 255.0f;
            }

            post++;
        }
    }
}
// function to calculate intersection coordinates
void calculateIntersection(const Box &a, const Box &b, float &x1, float &y1, float &x2, float &y2) {
    x1 = std::max(a.x_center - a.width / 2, b.x_center - b.width / 2);
    y1 = std::max(a.y_center - a.height / 2, b.y_center - b.height / 2);
    x2 = std::min(a.x_center + a.width / 2, b.x_center + b.width / 2);
    y2 = std::min(a.y_center + a.height / 2, b.y_center + b.height / 2);
}

// function to calculate area of a box
float calculateArea(const Box &box) {
    return box.width * box.height;
}

// IoU Calculation
float IoU(const Box &a, const Box &b) {
    float x1, y1, x2, y2;
    calculateIntersection(a, b, x1, y1, x2, y2);
    float intersection_width  = std::max(0.0f, x2 - x1);
    float intersection_height = std::max(0.0f, y2 - y1);
    float intersection_area   = intersection_width * intersection_height;
    float area_a = calculateArea(a);
    float area_b = calculateArea(b);
    return intersection_area / (area_a + area_b - intersection_area);
}
// function to sort boxes by confidence
void sortBoxesByConfidence(std::vector<Box> &boxes) {
    std::sort(boxes.begin(), boxes.end(), [](const Box &a, const Box &b) {
        return a.confidence > b.confidence;
    });
}
// Non-Maximum Suppression
std::vector<Box> NonMaximumSuppression(std::vector<Box> &boxes, float iou_threshold) {
    std::vector<Box> result;
    sortBoxesByConfidence(boxes);
    std::vector<bool> suppress(boxes.size(), false);
    for (size_t i = 0; i < boxes.size(); i++) {
        if (suppress[i]) continue;
        result.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); j++) {
            if (IoU(boxes[i], boxes[j]) > iou_threshold) {
                suppress[j] = true;
            }
        }
    }
    return result;
}

// function to clip coordinates within valid bounds
int clipCoordinate(int value, int min, int max) {
    return std::max(min, std::min(value, max));
}

// function to calculate clipped bounding box coordinates
void calculateClippedBoxCoordinates(const Box &box, int fb_width, int fb_height, 
                                    int &x_min, int &y_min, int &x_max, int &y_max) {
    x_min = clipCoordinate((box.x_center - box.width / 2) * fb_width, 0, fb_width - 1);
    y_min = clipCoordinate((box.y_center - box.height / 2) * fb_height, 0, fb_height - 1);
    x_max = clipCoordinate((box.x_center + box.width / 2) * fb_width, 0, fb_width - 1);
    y_max = clipCoordinate((box.y_center + box.height / 2) * fb_height, 0, fb_height - 1);
}

// function to draw horizontal edges of the bounding box
void drawHorizontalEdges(uint16_t *ptr, int fb_width, int x_min, int x_max, int y, uint16_t color) {
    for (int x = x_min; x <= x_max; x++) {
        ptr[y * fb_width + x] = color;
    }
}

void drawVerticalEdges(uint16_t *ptr, int fb_width, int y_min, int y_max, int x, uint16_t color) {
    for (int y = y_min; y <= y_max; y++) {
        ptr[y * fb_width + x] = color;
    }
}

void draw_box(camera_fb_t *fb, const Box &box, uint16_t color) {
    uint16_t *ptr = (uint16_t*)fb->buf;
    int fb_width  = fb->width;
    int fb_height = fb->height;

    int x_min, y_min, x_max, y_max;
    calculateClippedBoxCoordinates(box, fb_width, fb_height, x_min, y_min, x_max, y_max);

  
    drawHorizontalEdges(ptr, fb_width, x_min, x_max, y_min, color); 
    drawHorizontalEdges(ptr, fb_width, x_min, x_max, y_max, color); 

 
    drawVerticalEdges(ptr, fb_width, y_min, y_max, x_min, color);
    drawVerticalEdges(ptr, fb_width, y_min, y_max, x_max, color);
}


// function to process object detection
void processObjectDetection(camera_fb_t *fb) {
    TfLiteTensor *input = g_nn->getInput();
    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;

    ProcessImage(fb, input, input_scale, input_zero_point);

    g_nn->predict();

    TfLiteTensor *output = g_nn->getOutput();
    float *data = nullptr;

    // Ceil the output zero point
    int ceiled_output_zero_point = static_cast<int>(ceil(output->params.zero_point));
    if (output->type == kTfLiteUInt8) {
        data = new float[output->bytes / sizeof(uint8_t)];
        for (int i = 0; i < output->bytes; i++) {
            data[i] = (output->data.uint8[i] - ceiled_output_zero_point) * output->params.scale;
        }
    } else {
        data = output->data.f;
    }

    int num_boxes = output->dims->data[1];
    int box_size = output->dims->data[2];
    std::vector<Box> detected_boxes;

    for (int i = 0; i < num_boxes; i++) {
        float confidence = data[i * box_size + 4];
        int class_id = static_cast<int>(data[i * box_size + 5]);

        if (confidence > CONF_THRESHOLD && class_id == BOTTLE_CLASS_INDEX) {
            Box box;
            box.x_center = data[i * box_size + 0] * INPUT_W;
            box.y_center = data[i * box_size + 1] * INPUT_H;
            box.width = data[i * box_size + 2] * INPUT_W;
            box.height = data[i * box_size + 3] * INPUT_H;
            box.confidence = confidence;
            box.class_id = class_id;

            detected_boxes.push_back(box);
        }
    }

    // Perform Non-Maximum Suppression
    float iou_threshold = 0.5;
    std::vector<Box> final_boxes = NonMaximumSuppression(detected_boxes, iou_threshold);

    bool detection_made = false;
    for (const auto &box : final_boxes) {
        if (box.width != 0 && box.height != 0 && box.confidence > 0.5) {
            draw_box(fb, box, GREEN);
            Serial.println("Bottle detected!");
            Serial.printf("Detected box: [x=%.2f, y=%.2f, w=%.2f, h=%.2f, conf=%.2f,class_id=%d]\n",
                          box.x_center, box.y_center, box.width, box.height, box.confidence,box.class_id);
            detection_made = true;
        }
    }

    if (!detection_made) {
        Serial.println("No bottle detected.");
    }

    if (output->type == kTfLiteUInt8) {
        delete[] data;
    }
}

// function to handle JPEG conversion
bool convertToJPEG(camera_fb_t *fb, uint8_t **_jpg_buf, size_t *_jpg_buf_len) {
    if (fb->format != PIXFORMAT_JPEG) {
        bool jpeg_converted = frame2jpg(fb, 40, _jpg_buf, _jpg_buf_len);
        esp_camera_fb_return(fb);
        if (!jpeg_converted) {
            Serial.println("JPEG compression failed");
            return false;
        }
        return true;
    } else {
        *_jpg_buf = fb->buf;
        *_jpg_buf_len = fb->len;
        return true;
    }
}

// function to send the JPEG frame as a response
esp_err_t sendJPEGResponse(httpd_req_t *req, uint8_t *_jpg_buf, size_t _jpg_buf_len) {
    char part_buf[512];
    size_t hlen = snprintf((char *)part_buf, sizeof(part_buf), _STREAM_PART, _jpg_buf_len);

    esp_err_t res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    if (res == ESP_OK) {
        res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
        res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    }
    return res;
}

// Stream handler for the camera
static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;
    size_t _jpg_buf_len = 0;
    uint8_t *_jpg_buf = NULL;

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) {
        return res;
    }
    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("Camera capture failed");
            res = ESP_FAIL;
            break;
        }
        // Process object detection and JPEG conversion
        if (fb->format != PIXFORMAT_JPEG) {
            processObjectDetection(fb);
        }
        if (!convertToJPEG(fb, &_jpg_buf, &_jpg_buf_len)) {
            res = ESP_FAIL;
            break;
        }
        // Send the JPEG response
        res = sendJPEGResponse(req, _jpg_buf, _jpg_buf_len);
        if (res != ESP_OK) {
            break;
        }
        // Free dynamically allocated memory
        if (_jpg_buf != NULL) {
            free(_jpg_buf);
            _jpg_buf = NULL;
        }
        if (fb != NULL) {
            esp_camera_fb_return(fb);
            fb = NULL;
        }
    }
    return res;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

  httpd_uri_t index_uri = {
    .uri       = "/",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &index_uri);
  }
}
void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

    Serial.begin(115200);

    while (!Serial) {
        static int retries = 0;
        delay(100);
        if (retries++ > 3) {
            break;
        }
    }
    

    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 10000000;
    config.frame_size = FRAMESIZE_VGA;
    config.pixel_format = PIXFORMAT_RGB565;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.jpeg_quality = 8;
    config.fb_count = 4;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x", err);
        return;
    }

    g_nn = new NeuralNetwork();

    #if SETUP_AP==1
    WiFi.softAP(ssid, password);
    Serial.print("Camera Stream Ready! Go to: http://");
    Serial.print(WiFi.softAPIP());
    #else
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
      delay(500);
      Serial.print(".");
    }
    Serial.println("");
    Serial.println("WiFi connected");
    Serial.print("Camera Stream Ready! Go to: http://");
    Serial.print(WiFi.localIP());
    #endif
    startCameraServer();
}



// Main loop
void loop() {
delay(1000);
}