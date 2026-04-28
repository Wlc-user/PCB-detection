// ============================================================================
// C++ REST客户端 - libcurl实现
// ============================================================================
//
// 编译:
//   g++ -std=c++17 rest_client.cpp -lcurl -lpthread -o rest_client
//
// 依赖: libcurl-dev
//
// ============================================================================

#include <curl/curl.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <opencv2/opencv.hpp>

// ============================================================================
// 配置
// ============================================================================

struct Config {
    std::string server_url = "http://192.168.1.100:8080/api/detect";
    std::string camera_url = "rtsp://192.168.1.50:554/stream";
    int camera_id = 0;
    bool use_camera = true;
    int fps_limit = 30;
};

// ============================================================================
// Base64编码
// ============================================================================

static const char* BASE64_CHARS = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(const unsigned char* bytes, size_t len) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];
    
    while (len--) {
        char_array_3[i++] = *(bytes++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            
            for(i = 0; i < 4; i++)
                ret += BASE64_CHARS[char_array_4[i]];
            i = 0;
        }
    }
    
    if (i) {
        for(j = i; j < 3; j++)
            char_array_3[j] = '\0';
        
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        
        for (j = 0; j < i + 1; j++)
            ret += BASE64_CHARS[char_array_4[j]];
        
        while((i++ < 3))
            ret += '=';
    }
    
    return ret;
}

// ============================================================================
// HTTP客户端
// ============================================================================

class HttpClient {
public:
    HttpClient() {
        curl_ = curl_easy_init();
    }
    
    ~HttpClient() {
        if (curl_) curl_easy_cleanup(curl_);
    }
    
    std::string post_json(const std::string& url, const std::string& json_data) {
        std::string response;
        
        if (!curl_) return response;
        
        curl_easy_setopt(curl_, CURL_URL, url.c_str());
        curl_easy_setopt(curl_, CURL_POSTFIELDS, json_data.c_str());
        curl_easy_setopt(curl_, CURL_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl_, CURL_WRITEDATA, &response);
        
        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl_, CURL_HTTPHEADER, headers);
        
        CURLcode res = curl_easy_perform(curl_);
        
        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform failed: " << curl_easy_strerror(res) << std::endl;
        }
        
        curl_slist_free_all(headers);
        return response;
    }
    
private:
    static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
        ((std::string*)userp)->append((char*)contents, size * nmemb);
        return size * nmemb;
    }
    
    CURL* curl_;
};

// ============================================================================
// 控制决策
// ============================================================================

class ControlDecision {
public:
    struct ControlCmd {
        std::string command;
        float speed;
        float angle;
        std::string message;
        float danger_level;
    };
    
    ControlCmd parse(const std::string& json_response) {
        ControlCmd cmd;
        cmd.command = "move_forward";
        cmd.speed = 0.5;
        cmd.angle = 0.0;
        cmd.message = "OK";
        cmd.danger_level = 0.0;
        
        // 简单JSON解析
        // 实际应用中建议使用nlohmann/json库
        
        return cmd;
    }
    
    void sendCommand(const ControlCmd& cmd) {
        // TODO: 发送到小车控制器
        // serial_port.write(cmd);
        std::cout << "[>>] CMD: " << cmd.command 
                  << " | Speed: " << cmd.speed 
                  << " | Angle: " << cmd.angle
                  << " | " << cmd.message << std::endl;
    }
};

// ============================================================================
// 主程序
// ============================================================================

int main() {
    Config config;
    HttpClient http;
    ControlDecision ctrl;
    
    std::cout << "========================================" << std::endl;
    std::cout << "  REST Perception Client for AGV" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Server: " << config.server_url << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 打开视频源
    cv::VideoCapture cap;
    if (config.use_camera) {
        cap.open(config.camera_id);
    } else {
        cap.open(config.camera_url);
    }
    
    if (!cap.isOpened()) {
        std::cerr << "[!] Cannot open video source" << std::endl;
        return 1;
    }
    
    std::cout << "[OK] Video source opened" << std::endl;
    
    cv::Mat frame;
    auto last_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "[!] Empty frame" << std::endl;
            break;
        }
        
        // FPS控制
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_time).count();
        if (elapsed < 1.0 / config.fps_limit) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        last_time = now;
        
        // 编码为JPEG
        std::vector<unsigned char> buf;
        cv::imencode(".jpg", frame, buf);
        
        // Base64编码
        std::string img_base64 = base64_encode(buf.data(), buf.size());
        
        // 构建JSON
        std::string json = "{\"image_base64\":\"" + img_base64 + "\",\"camera_id\":\"agv_cam_0\"}";
        
        // 发送请求
        std::string response = http.post_json(config.server_url, json);
        
        if (!response.empty()) {
            // 解析响应
            auto cmd = ctrl.parse(response);
            ctrl.sendCommand(cmd);
            
            // 绘制结果
            // 实际应用中在本地显示
        }
        
        frame_count++;
        if (frame_count % 30 == 0) {
            std::cout << "[*] Processed " << frame_count << " frames" << std::endl;
        }
        
        // ESC退出
        if (cv::waitKey(1) == 27) break;
    }
    
    cap.release();
    return 0;
}
