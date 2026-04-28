"""
YOLOv8s 微调优化脚本 - PCB缺陷检测
使用更大的 YOLOv8s 模型
"""
from ultralytics import YOLO
import os
import sys

PROJECT_DIR = r"e:\pyspace\opencv"
# 使用增强数据集
DATA_YAML = os.path.join(PROJECT_DIR, "yolo_pcb_dataset", "data.yaml")
def update_processed_image(self):
    """实时更新预处理图像"""
    img = self.original_image.copy()
    
    # 滤波 - 实时应用
    if self.filter_type == 'gaussian':
        img = cv2.GaussianBlur(img, (self.filter_kernel, self.filter_kernel), 0)
    # ...
    
    # 边缘检测 - 实时应用
    if self.edge_type == 'canny':
        edges = cv2.Canny(gray, 50, 150)
        img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # ...
    
    self.processed_image = img
    self.update_display()
def main():
    print("=" * 60)
    print("YOLOv8s 微调训练 - PCB 缺陷检测")
    print("=" * 60)
    print(f"数据集: {DATA_YAML}")
    print(f"模型: yolov8s.pt (small, 11M参数)")
    print(f"图像尺寸: 1280")
    print("=" * 60)
    
    # 加载 YOLOv8s 模型
    print("\n加载 YOLOv8s 模型...")
    try:
        model = YOLO("yolov8s.pt")
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("\n请手动下载 yolov8s.pt:")
        print("  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt")
        print("或使用镜像:")
        print("  https://modelscope.cn/models/AI-ModelScope/yolov8s")
        return
    
    print("\n开始训练 (80 epochs, 图像尺寸 1280)...")
    results = model.train(
        data=DATA_YAML,
        epochs=80,
        imgsz=1280,
        batch=4,              # YOLOv8s 更大，降低batch
        patience=25,
        save=True,
        project=os.path.join(PROJECT_DIR, "models", "yolov8s"),
        name="train_v8s_finetune",
        exist_ok=True,
        
        # 优化参数
        optimizer="AdamW",
        lr0=0.0005,          # 较低学习率
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # 数据增强 (PCB专用)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.8,           # 保留mosaic
        mixup=0.1,
        copy_paste=0.1,
        
        pretrained=True,
        verbose=True,
        workers=0,
        cache=True,
    )
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    
    # 验证
    best_path = os.path.join(PROJECT_DIR, "models", "yolov8s", "train_v8s_finetune", "weights", "best.pt")
    if os.path.exists(best_path):
        print("\n验证最佳模型...")
        val_model = YOLO(best_path)
        val_results = val_model.val(data=DATA_YAML, split="test", imgsz=1280)
        
        print("\n最终测试结果:")
        print(f"  Precision: {val_results.box.mp:.1%}")
        print(f"  Recall: {val_results.box.mr:.1%}")
        print(f"  mAP50: {val_results.box.map50:.1%}")
        print(f"  mAP50-95: {val_results.box.map:.1%}")


if __name__ == "__main__":
    main()
