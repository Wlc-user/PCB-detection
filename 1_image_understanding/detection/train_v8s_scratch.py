"""
YOLOv8n 从头训练脚本 - PCB 缺陷检测
使用 YOLOv8n (nano) 模型从零训练
"""
from ultralytics import YOLO
import os

PROJECT_DIR = r"e:\pyspace\opencv"
DATA_YAML = os.path.join(PROJECT_DIR, "yolo_pcb_dataset", "data.yaml")
MODEL_FILE = os.path.join(PROJECT_DIR, "yolov8n.pt")

def main():
    print("=" * 60)
    print("YOLOv8n 从头训练 - PCB 缺陷检测")
    print("=" * 60)
    print(f"数据集: {DATA_YAML}")
    
    # 加载模型
    print(f"使用模型: {MODEL_FILE}")
    model = YOLO(MODEL_FILE)
    
    # 开始训练
    print("\n开始训练 (100 epochs)...")
    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=8,
        patience=30,
        save=True,
        project=os.path.join(PROJECT_DIR, "models", "yolov8"),
        name="train_v8n_scratch",
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        pretrained=False,
        verbose=True,
        workers=0,
    )
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
