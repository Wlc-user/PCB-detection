"""
多平台部署适配器
支持: ONNX Runtime / TensorRT / C++ / C# / 移动端
"""
import os
from ultralytics import YOLO


def export_all_formats():
    """导出所有常用格式"""
    model_path = "models/yolov8/train/weights/best.pt"
    model = YOLO(model_path)
    
    # 导出格式说明
    formats = {
        # 格式: (format, args, 说明)
        "onnx": ({"format": "onnx", "imgsz": 640, "simplify": True}, "跨平台通用"),
        "onnx_opset12": ({"format": "onnx", "imgsz": 640, "opset": 12}, "ONNX Opset12"),
        "torchscript": ({"format": "torchscript", "imgsz": 640}, "LibTorch C++"),
        "engine": ({"format": "engine", "imgsz": 640, "half": True}, "TensorRT FP16"),
        "engine_int8": ({"format": "engine", "imgsz": 640, "int8": True}, "TensorRT INT8"),
        "tflite": ({"format": "tflite", "imgsz": 640, "int8": True}, "移动端 TFLite"),
        "coreml": ({"format": "coreml", "imgsz": 640}, "苹果 CoreML"),
    }
    
    print("=" * 60)
    print("多平台模型导出")
    print("=" * 60)
    
    for name, (kwargs, desc) in formats.items():
        print(f"\n[{name}] {desc}")
        try:
            export_path = model.export(**kwargs)
            print(f"  成功: {export_path}")
        except Exception as e:
            print(f"  失败: {e}")


def test_onnx_inference():
    """测试 ONNX 推理"""
    import cv2
    import numpy as np
    import onnxruntime as ort
    
    onnx_path = "models/yolov8/train/weights/best.onnx"
    
    print("\n" + "=" * 60)
    print("ONNX Runtime 推理测试")
    print("=" * 60)
    
    # 创建 session
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # 读取测试图片
    img = cv2.imread("yolo_pcb_dataset/images/test/01_missing_hole_02.jpg")
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # 推理
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    import time
    times = []
    for _ in range(20):
        start = time.time()
        outputs = session.run([output_name], {input_name: img})
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000
    fps = 1000 / avg_time
    
    print(f"  输入 shape: {img.shape}")
    print(f"  输出 shape: {outputs[0].shape}")
    print(f"  平均推理时间: {avg_time:.1f} ms")
    print(f"  FPS: {fps:.1f}")


def generate_cpp_code():
    """生成 C++ 推理代码模板"""
    cpp_code = '''
// YOLOv8 PCB 缺陷检测 - C++ ONNX Runtime
// 编译: g++ -o detector detector.cpp -lonnxruntime
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect bbox;
};

class YOLODetector {
private:
    Ort::Env env;
    Ort::Session session;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    int img_size = 640;

public:
    YOLODetector(const char* model_path) 
        : env(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLO")),
          session(env, ORT_MODEL_PATH, Ort::SessionOptions{}) {
        
        // 获取输入输出名称
        auto input = session.GetInputNames();
        auto output = session.GetOutputNames();
        input_names = input;
        output_names = output;
    }

    std::vector<Detection> detect(const cv::Mat& img) {
        // 预处理
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(img_size, img_size));
        cv::Mat blob;
        resized.convertTo(blob, CV_32FC3, 1.0/255.0);
        
        // 转换为输入 tensor
        std::vector<float> input_data(3 * img_size * img_size);
        std::vector<cv::Mat> channels(3);
        cv::split(blob, channels);
        for(int i=0; i<3; i++)
            memcpy(input_data.data() + i*img_size*img_size, 
                   channels[i].data, img_size*img_size*sizeof(float));
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
            input_data.data(), input_data.size(),
            {1, 3, img_size, img_size});
        
        // 推理
        auto outputs = session.Run(Ort::RunOptions{nullptr},
            input_names.data(), &input_tensor, 1,
            output_names.data(), 1);
        
        // 后处理 (需要根据输出解析 bbox)
        // ...
        
        return {}; // 返回检测结果
    }
};

int main() {
    YOLODetector detector("best.onnx");
    cv::Mat img = cv::imread("test.jpg");
    auto results = detector.detect(img);
    return 0;
}
'''
    
    print("\n" + "=" * 60)
    print("C++ 代码模板")
    print("=" * 60)
    print(cpp_code)
    print("\n保存到: cpp_detector.cpp")
    
    with open("cpp_detector.cpp", "w", encoding="utf-8") as f:
        f.write(cpp_code)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="导出所有格式")
    parser.add_argument("--test-onnx", action="store_true", help="测试ONNX推理")
    parser.add_argument("--cpp", action="store_true", help="生成C++代码")
    
    args = parser.parse_args()
    
    if args.export:
        export_all_formats()
    if args.test_onnx:
        test_onnx_inference()
    if args.cpp:
        generate_cpp_code()
    
    if not any(vars(args).values()):
        # 默认测试ONNX
        test_onnx_inference()
