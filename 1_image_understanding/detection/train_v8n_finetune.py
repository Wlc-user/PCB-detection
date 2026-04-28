"""
YOLOv8n 微调优化脚本 - 基于最佳模型继续训练
使用更大的图像尺寸和更强的数据增强
"""
from ultralytics import YOLO
import os

PROJECT_DIR = r"e:\pyspace\opencv"
DATA_YAML = os.path.join(PROJECT_DIR, "yolo_pcb_dataset", "data.yaml")
BEST_MODEL = os.path.join(PROJECT_DIR, "models", "yolov8", "train", "weights", "best.pt")

def main():
    print("=" * 60)
    print("YOLOv8n 微调优化")
    print("=" * 60)
    print(f"数据集: {DATA_YAML}")
    print(f"基础模型: {BEST_MODEL}")
    print("图像尺寸: 1280 (更大尺寸检测小目标)")
    print("=" * 60)
    
    # 加载最佳模型继续训练
    model = YOLO(BEST_MODEL)
    
    print("\n开始微调 (50 epochs, 图像尺寸 1280)...")
    results = model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=1280,           # 更大尺寸
        batch=4,              # 减小batch适配大图像
        patience=20,
        save=True,
        project=os.path.join(PROJECT_DIR, "models", "yolov8"),
        name="train_v8n_finetune",
        exist_ok=True,
        
        # 微调参数
        optimizer="AdamW",
        lr0=0.0001,           # 低学习率微调
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # 增强数据
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,          # 减小旋转
        translate=0.1,
        scale=0.3,            # 减小缩放
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.8,           # 减小mosaic
        mixup=0.05,           # 减小mixup
        copy_paste=0.05,
        
        pretrained=True,
        verbose=True,
        workers=0,
        cache=True,           # 缓存数据加速
    )
    
    print("\n" + "=" * 60)
    print("微调完成!")
    print("=" * 60)
    
    # 验证
    best_path = os.path.join(PROJECT_DIR, "models", "yolov8", "train_v8n_finetune", "weights", "best.pt")
    val_model = YOLO(best_path)
    val_results = val_model.val(data=DATA_YAML, split="test", imgsz=1280)
    
    print("\n最终测试结果:")
    print(f"  Precision: {val_results.box.mp:.1%}")
    print(f"  Recall: {val_results.box.mr:.1%}")
    print(f"  mAP50: {val_results.box.map50:.1%}")
    print(f"  mAP50-95: {val_results.box.map:.1%}")

if __name__ == "__main__":
    main()
