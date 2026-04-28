// ============================================================================
// C++ 嵌入式小车 - Python感知服务客户端
// ============================================================================
// 
// 编译:
//   g++ -std=c++17 perception_client.cpp logistics.grpc.pb.cc logistics.pb.cc \
//        -I${GRPC_INCLUDE} -L${GRPC_LIB} -lgrpc++ -lprotobuf -pthread -o perception_client
//
// ============================================================================

#include <grpcpp/grpcpp.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>

#include "logistics.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using grpc::ClientReaderWriter;

using logistics::PerceptionService;
using logistics::ImageData;
using logistics::DetectionResult;
using logistics::DetectedObject;
using logistics::BoundingBox;
using logistics::DetectionEvent;
using logistics::SubscribeRequest;

// ============================================================================
// 配置
// ============================================================================

struct Config {
    std::string server_address = "192.168.1.100:50051";
    int camera_id = 0;
    int frame_skip = 2;
    float confidence_threshold = 0.5;
};

// ============================================================================
// 感知客户端
// ============================================================================

class PerceptionClient {
public:
    PerceptionClient(std::shared_ptr<Channel> channel)
        : stub_(PerceptionService::NewStub(channel)) {}
    
    DetectionResult DetectFrame(const ImageData& image) {
        ClientContext context;
        DetectionResult response;
        Status status = stub_->DetectFrame(&context, image, &response);
        if (!status.ok()) {
            std::cerr << "DetectFrame failed: " << status.error_message() << std::endl;
        }
        return response;
    }
    
    std::unique_ptr<ClientReaderWriter<ImageData, DetectionResult>> SubscribeStream() {
        ClientContext context;
        return stub_->DetectStream(&context);
    }
    
    std::unique_ptr<ClientReader<SubscribeRequest, DetectionEvent>> SubscribeEvents(
        const SubscribeRequest& request) {
        ClientContext context;
        return stub_->SubscribeEvents(&context, request);
    }
    
private:
    std::unique_ptr<PerceptionService::Stub> stub_;
};

// ============================================================================
// 控制决策
// ============================================================================

class ControlDecision {
public:
    struct ControlCmd {
        std::string command;  // stop, move_forward, turn_left, turn_right, slow_down
        float speed;          // 0.0 - 1.0
        float angle;          // -90 to 90 degrees
    };
    
    ControlCmd makeDecision(const DetectionResult& result) {
        ControlCmd cmd;
        cmd.command = "move_forward";
        cmd.speed = 0.5;
        cmd.angle = 0.0;
        
        for (const auto& obj : result.objects()) {
            if (obj.category() == "obstacle" || obj.category() == "person") {
                if (obj.confidence() > 0.6) {
                    float center_y = obj.center_y();
                    
                    if (center_y > 0.7) {
                        cmd.command = "stop";
                        cmd.speed = 0.0;
                        std::cout << "[!] STOP: " << obj.class_name() << std::endl;
                    } else if (center_y > 0.4) {
                        cmd.command = "slow_down";
                        cmd.speed = 0.2;
                        std::cout << "[!] SLOW: " << obj.class_name() << std::endl;
                    }
                }
            }
        }
        
        return cmd;
    }
    
    void sendCommand(const ControlCmd& cmd) {
        std::cout << "[>>] Command: " << cmd.command 
                  << " speed=" << cmd.speed 
                  << " angle=" << cmd.angle << std::endl;
    }
    
    void handleEvent(const DetectionEvent& event) {
        std::cout << "[EVENT] " << event.event_type() << ": " << event.message() << std::endl;
    }
    
private:
    std::mutex cmd_mutex_;
};

// ============================================================================
// 主程序
// ============================================================================

int main(int argc, char** argv) {
    Config config;
    
    std::cout << "========================================" << std::endl;
    std::cout << "  C++ Perception Client for AGV" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Server: " << config.server_address << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto channel = grpc::CreateChannel(
        config.server_address,
        grpc::InsecureChannelCredentials()
    );
    
    channel->WaitForConnected(gpr_time_add(
        gpr_now(GPR_CLOCK_REALTIME),
        gpr_time_from_seconds(5, GPR_TIMESPAN)
    ));
    
    PerceptionClient client(channel);
    ControlDecision ctrl;
    
    std::cout << "[OK] Connected to perception server" << std::endl;
    
    // 事件订阅模式
    SubscribeRequest request;
    request.add_categories("person");
    request.add_categories("obstacle");
    request.add_categories("vehicle");
    request.set_confidence_threshold(0.5);
    
    auto events = client.SubscribeEvents(request);
    
    DetectionEvent event;
    while (events->Read(&event)) {
        ctrl.handleEvent(event);
        
        // 根据事件生成控制指令
        if (event.danger_level() > 0.7) {
            ControlDecision::ControlCmd cmd;
            cmd.command = "stop";
            cmd.speed = 0.0;
            ctrl.sendCommand(cmd);
        }
    }
    
    return 0;
}
