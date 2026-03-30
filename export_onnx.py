"""
ONNX模型导出工具
支持跨平台部署 (Windows/Linux/Jetson/手机)
"""
from ultralytics import YOLO
import sys


def export_onnx():
    """导出ONNX模型"""
    model_path = "models/yolov8/train/weights/best.pt"
    
    print("="*50)
    print("YOLOv8 ONNX 导出")
    print("="*50)
    
    print(f"\n加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 导出为ONNX
    print("\n导出为ONNX格式...")
    export_path = model.export(format="onnx", imgsz=640, simplify=True)
    
    print(f"\n✅ ONNX模型已保存: {export_path}")
    print(f"\n文件大小: {get_size(export_path)} MB")
    
    return export_path


def get_size(filepath):
    """获取文件大小MB"""
    import os
    size = os.path.getsize(filepath) / (1024 * 1024)
    return round(size, 2)


def test_onnx():
    """测试ONNX模型"""
    import cv2
    import numpy as np
    import onnxruntime as ort
    
    onnx_path = "models/yolov8/train/weights/best.onnx"
    
    print("\n测试ONNX模型...")
    
    # 创建推理会话
    session = ort.InferenceSession(onnx_path)
    
    # 读取图片
    img = cv2.imread("yolo_pcb_dataset/images/test/01_missing_hole_02.jpg")
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # 推理
    import time
    start = time.time()
    output = session.run(None, {"images": img})
    elapsed = time.time() - start
    
    print(f"推理时间: {elapsed*1000:.1f} ms")
    print(f"输出shape: {output[0].shape}")
    
    return output


if __name__ == "__main__":
    try:
        export_onnx()
        
        # 测试ONNX
        test_onnx()
        
    except Exception as e:
        print(f"错误: {e}")
        print("\n如果onnxruntime未安装，运行:")
        print("pip install onnxruntime")
