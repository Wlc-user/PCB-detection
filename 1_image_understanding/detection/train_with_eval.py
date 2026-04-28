"""
PCB缺陷检测 - 训练 + 自动评估
训练完成后自动进行全面评估
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import torch
import numpy as np

# 类别名称
CLASS_NAMES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']


def auto_evaluate(model_path, name="模型"):
    """自动评估模型"""
    print("\n" + "="*70)
    print(f"📊 自动评估: {name}")
    print("="*70)
    
    model = YOLO(model_path)
    
    # 1. 搜索最优置信度
    print("\n[置信度扫描]")
    print(f"{'Conf':<8} {'P':<10} {'R':<10} {'F1':<10} {'mAP50':<10}")
    print("-" * 48)
    
    best_f1, best_conf = 0, 0.2
    
    for conf in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        metrics = model.val(data='models/yolov8/data.yaml', conf=conf, verbose=False, workers=0)
        p, r = metrics.box.mp, metrics.box.mr
        f1 = 2 * p * r / (p + r + 1e-8)
        map50 = metrics.box.map50
        
        print(f"{conf:<8.2f} {p*100:>8.1f}% {r*100:>8.1f}% {f1:>8.3f} {map50:>8.3f}")
        
        if f1 > best_f1:
            best_f1, best_conf = f1, conf
            best_p, best_r, best_map = p, r, map50
    
    print("-" * 48)
    print(f"\n🏆 最优配置: conf={best_conf:.2f}")
    print(f"   Precision={best_p*100:.1f}%, Recall={best_r*100:.1f}%, mAP50={best_map:.3f}, F1={best_f1:.3f}")
    
    # 2. 各类别AP50
    print("\n" + "="*70)
    print("📋 各类别 AP50:")
    print("="*70)
    
    metrics = model.val(data='models/yolov8/data.yaml', conf=best_conf, verbose=False, workers=0)
    
    if hasattr(metrics.box, 'ap_class_index'):
        ap_dict = {}
        for i, idx in enumerate(metrics.box.ap_class_index):
            if idx < len(CLASS_NAMES):
                ap50 = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
                ap_dict[CLASS_NAMES[idx]] = ap50
        
        for name in CLASS_NAMES:
            ap = ap_dict.get(name, 0)
            bar = "█" * int(ap * 20)
            status = "✓" if ap >= 0.4 else "✗"
            print(f"  {name:<18}: {ap:.3f} |{bar:<20}| {status}")
    
    # 3. 目标对比
    print("\n" + "="*70)
    print("🎯 目标 vs 当前:")
    print("="*70)
    print(f"{'指标':<15} {'目标':<12} {'当前':<12} {'差距':<10} {'状态'}")
    print("-" * 60)
    
    targets = [('Precision', 0.70), ('Recall', 0.75), ('mAP50', 0.80)]
    current = [('Precision', best_p), ('Recall', best_r), ('mAP50', best_map)]
    
    for (t_name, t_val), (c_name, c_val) in zip(targets, current):
        gap = t_val - c_val
        status = "✅ 达标" if c_val >= t_val else f"❌ 差 {gap*100:.1f}%"
        print(f"{t_name:<15} ≥{t_val*100:>5.0f}%{'':>5} {c_val*100:>6.1f}%{'':>5} {gap*100:>+6.1f}%   {status}")
    
    # 4. 部署建议
    print("\n" + "="*70)
    print("📝 部署建议:")
    print("="*70)
    print(f"  model = YOLO('{model_path}')")
    print(f"  results = model.predict(conf={best_conf}, iou=0.45)")
    
    # 评估整体是否满足要求
    all_pass = (best_p >= 0.70 and best_r >= 0.75 and best_map >= 0.80)
    if all_pass:
        print("\n  ✅ 模型已达标，可以部署！")
    else:
        print("\n  ⚠️  模型未达标，建议继续优化或调整阈值")
    
    return best_conf, best_f1


def train_and_evaluate():
    """训练并自动评估"""
    print("="*70)
    print("🚀 PCB缺陷检测 - 训练 + 自动评估")
    print("="*70)
    
    # 配置
    DATA_CONFIG = {
        'path': str(Path('yolo_pcb_dataset').absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': CLASS_NAMES,
        'nc': 6
    }
    
    Path('models/yolov8').mkdir(parents=True, exist_ok=True)
    with open('models/yolov8/data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(DATA_CONFIG, f, allow_unicode=True)
    
    # 检查可用模型
    model_path = 'yolov8n.pt'
    if Path('yolov8s.pt').exists():
        model_path = 'yolov8s.pt'
    elif Path('models/yolov8/train/weights/best.pt').exists():
        model_path = 'models/yolov8/train/weights/best.pt'
    
    print(f"\n使用基础模型: {model_path}")
    
    # 开始训练
    model = YOLO(model_path)
    
    results = model.train(
        data='models/yolov8/data.yaml',
        
        # 训练配置
        epochs=300,
        imgsz=1280,
        batch=4,
        patience=80,
        
        name='train_auto',
        project='models/yolov8',
        exist_ok=True,
        
        # 优化器
        optimizer='AdamW',
        lr0=0.0002,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_bias_lr=0.1,
        
        # 损失函数
        box=7.5,
        cls=3.0,
        dfl=1.5,
        
        # 设备
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=0,
        amp=True,
        
        # 数据增强
        augment=True,
        mosaic=1.0,
        mixup=0.3,
        copy_paste=0.3,
        
        degrees=10.0,
        translate=0.15,
        scale=0.5,
        shear=1.0,
        perspective=0.0005,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        flipud=0.0,
        fliplr=0.5,
        
        val=True,
        plots=True,
        conf=0.1,
        iou=0.5,
        verbose=True,
    )
    
    # 训练完成后自动评估
    best_model_path = 'models/yolov8/train_auto/weights/best.pt'
    auto_evaluate(best_model_path, "训练完成模型")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='训练并评估')
    parser.add_argument('--eval', type=str, default='', help='评估指定模型')
    args = parser.parse_args()
    
    if args.train:
        train_and_evaluate()
    elif args.eval:
        auto_evaluate(args.eval, args.eval)
    else:
        print("用法:")
        print("  python train_with_eval.py --train    # 训练并自动评估")
        print("  python train_with_eval.py --eval models/yolov8/train/weights/best.pt")
