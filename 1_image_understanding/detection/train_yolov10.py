"""
YOLOv10 Finetuning - PCB Defect Detection
Using Augmented Dataset + YOLOv10 Model
"""
from ultralytics import YOLO
import os

# Load trained model for prediction
model = YOLO(r'E:/pyspace/opencv/models/yolov10/train_yolov10/weights/best.pt') 

# Predict on unlabeled images and auto-generate txt labels
results = model.predict(
    source='E:/pyspace/opencv/models/yolov8/train',
    save_txt=True,
    save_conf=True,
    conf=0.25,
    project='path/to/save/output'
)

PROJECT_DIR = r"e:\pyspace\opencv"
AUG_DATA_DIR = os.path.join(PROJECT_DIR, "yolo_pcb_dataset_aug")
ORIG_DATA_DIR = os.path.join(PROJECT_DIR, "yolo_pcb_dataset")
DATA_YAML = os.path.join(ORIG_DATA_DIR, "data.yaml")

# Merge augmented training data if available
if os.path.exists(AUG_DATA_DIR):
    aug_train_img = os.path.join(AUG_DATA_DIR, "images", "train")
    aug_train_label = os.path.join(AUG_DATA_DIR, "labels", "train")

    orig_train_img = os.path.join(ORIG_DATA_DIR, "images", "train")
    orig_train_label = os.path.join(ORIG_DATA_DIR, "labels", "train")

    if os.path.exists(aug_train_img) and os.path.exists(orig_train_img):
        print("\nMerging augmented data into training set...")
        import shutil
        aug_count = 0

        for f in os.listdir(aug_train_img):
            src = os.path.join(aug_train_img, f)
            dst = os.path.join(orig_train_img, f"aug_{f}")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                aug_count += 1

        for f in os.listdir(aug_train_label):
            src = os.path.join(aug_train_label, f)
            dst = os.path.join(orig_train_label, f"aug_{f}")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        print(f"Added {aug_count} augmented images to training set")
        print(f"Using dataset: {DATA_YAML}")
    else:
        print(f"Augmented train set incomplete, using original data: {DATA_YAML}")
else:
    print(f"Augmented dataset not found, using original data: {DATA_YAML}")

def main():
    print("=" * 60)
    print("YOLOv10 Finetuning - PCB Defect Detection")
    print("=" * 60)
    print(f"Dataset: {DATA_YAML}")
    print(f"Model: yolov10s.pt (small)")
    print(f"Image size: 1280")
    print("=" * 60)

    print("\nLoading YOLOv10s model...")
    try:
        model = YOLO("yolov10s.pt")
        print("✅ YOLOv10s model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("\nYOLOv10 requires installation:")
        print("  https://docs.ultralytics.com/models/yolov10")
        print("\nFallback: Use YOLOv8s")
        print("  python train_yolov8s.py")
        return

    print("\nStarting training (80 epochs, image size 1280)...")
    results = model.train(
        data=DATA_YAML,
        epochs=80,
        imgsz=1280,
        batch=4,
        patience=25,
        save=True,
        project=os.path.join(PROJECT_DIR, "models", "yolov10"),
        name="train_yolov10",
        exist_ok=True,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

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
        mosaic=0.8,
        mixup=0.1,
        copy_paste=0.1,

        pretrained=True,
        verbose=True,
        workers=4,
        cache=True,
    )

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    best_path = os.path.join(PROJECT_DIR, "models", "yolov10", "train_yolov10", "weights", "best.pt")
    if os.path.exists(best_path):
        print("\nValidating best model...")
        val_model = YOLO(best_path)
        val_results = val_model.val(data=DATA_YAML, split="test", imgsz=1280)

        print("\nFinal test results:")
        print(f"  Precision: {val_results.box.mp:.1%}")
        print(f"  Recall: {val_results.box.mr:.1%}")
        print(f"  mAP50: {val_results.box.map50:.1%}")
        print(f"  mAP50-95: {val_results.box.map:.1%}")

if __name__ == "__main__":
    main()