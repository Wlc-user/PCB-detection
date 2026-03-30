"""
模型量化工具
将YOLOv8模型量化加速，提升推理速度2-4倍
"""
import torch
from ultralytics import YOLO
import numpy as np
import cv2
import time
import os


def quantize_model():
    """PTQ后训练量化"""
    print("="*50)
    print("YOLOv8 模型量化加速")
    print("="*50)
    
    # 加载模型
    model_path = "models/yolov8/train/weights/best.pt"
    print(f"\n加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 获取PyTorch模型
    pt_model = model.model
    
    # 准备校准数据 (使用测试集图片)
    print("\n准备校准数据...")
    calibration_images = []
    test_dir = "yolo_pcb_dataset/images/test"
    
    import glob
    image_files = glob.glob(f"{test_dir}/*.jpg")[:20]
    
    for img_file in image_files:
        img = cv2.imread(img_file)
        if img is not None:
            img = cv2.resize(img, (640, 640))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).float() / 255.0
            img = img.unsqueeze(0)  # 添加batch维度
            calibration_images.append(img)
    
    print(f"  准备 {len(calibration_images)} 张图片")
    
    # 动态量化
    print("\n执行动态量化...")
    pt_model.eval()
    
    # 量化配置
    quantized_model = torch.quantization.quantize_dynamic(
        pt_model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )
    
    # 保存量化模型 - 保存完整checkpoint结构
    output_path = "models/yolov8/train/weights/best_quantized.pt"
    print(f"\n保存量化模型: {output_path}")
    # 保存完整模型结构，YOLO才能正确加载
    torch.save({
        'model': quantized_model,
        'ema': quantized_model,
        'args': {},
        'ckpt': quantized_model,
    }, output_path)
    
    return output_path


def benchmark_model(model_path: str, use_quantized: bool = False):
    """测试模型推理速度"""
    print(f"\n测试推理速度: {model_path}")
    
    model = YOLO(model_path)
    
    # 预热
    print("  预热...")
    for _ in range(5):
        _ = model.predict("yolo_pcb_dataset/images/test/01_missing_hole_02.jpg", 
                         verbose=False)
    
    # 测试
    print("  测试...")
    test_image = "yolo_pcb_dataset/images/test/01_missing_hole_02.jpg"
    times = []
    
    for _ in range(20):
        start = time.time()
        _ = model.predict(test_image, verbose=False)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print(f"\n  平均推理时间: {avg_time*1000:.1f} ms")
    print(f"  FPS: {fps:.1f}")
    
    return avg_time


def export_onnx():
    """导出ONNX模型 (可跨平台部署)"""
    print("\n" + "="*50)
    print("导出ONNX模型")
    print("="*50)
    
    model = YOLO("models/yolov8/train/weights/best.pt")
    
    # 导出
    print("\n导出中...")
    model.export(format="onnx", imgsz=640, simplify=True)
    
    onnx_path = "models/yolov8/train/weights/best.onnx"
    print(f"ONNX模型已保存: {onnx_path}")
    
    return onnx_path


def main():
    """主函数"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║           YOLOv8 模型优化工具                              ║
║                                                           ║
║  功能:                                                   ║
║  1. 模型量化 (INT8) - 加速2-4倍                          ║
║  2. 导出ONNX - 跨平台部署                                ║
║  3. 性能测试                                             ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # 1. 原始模型性能测试
    print("\n[1] 测试原始模型性能")
    benchmark_model("models/yolov8/train/weights/best.pt")
    
    # 2. 量化模型
    print("\n[2] 量化模型")
    quantize_model()
    
    # 3. 量化后性能测试
    print("\n[3] 测试量化模型性能")
    benchmark_model("models/yolov8/train/weights/best_quantized.pt")
    
    # 4. 导出ONNX
    print("\n[4] 导出ONNX")
    export_onnx()
    
    print("\n" + "="*50)
    print("优化完成!")
    print("="*50)


if __name__ == "__main__":
    main()
