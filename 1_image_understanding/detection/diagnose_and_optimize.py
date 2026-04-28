"""
PCB缺陷检测 - 数据集诊断与优化
分析标注质量和类别分布
"""

from ultralytics import YOLO
from pathlib import Path
from collections import Counter
import yaml
import numpy as np
import matplotlib.pyplot as plt

# 类别名称
CLASS_NAMES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

def analyze_dataset():
    """分析数据集统计信息"""
    print("="*60)
    print("数据集统计与分析")
    print("="*60)
    
    # 读取标签
    dataset_path = Path('yolo_pcb_dataset')
    train_labels = dataset_path / 'labels' / 'train'
    
    # 统计
    class_counts = Counter()
    box_sizes = {'small': 0, 'medium': 0, 'large': 0}
    total_boxes = 0
    image_sizes = []
    
    if train_labels.exists():
        for label_file in train_labels.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        w, h = float(parts[3]), float(parts[4])
                        
                        # 类别统计
                        if cls < len(CLASS_NAMES):
                            class_counts[cls] += 1
                        
                        # 尺寸统计 (COCO标准: small<32x32, medium 32-96, large>96)
                        area = w * h
                        if area < 0.01:  # <1%图像面积
                            box_sizes['small'] += 1
                        elif area < 0.1:  # 1-10%
                            box_sizes['medium'] += 1
                        else:
                            box_sizes['large'] += 1
                        total_boxes += 1
    
    # 输出统计
    print(f"\n[类别分布]")
    for cls_id, count in sorted(class_counts.items()):
        if cls_id < len(CLASS_NAMES):
            pct = count / sum(class_counts.values()) * 100 if class_counts else 0
            print(f"  {CLASS_NAMES[cls_id]}: {count} ({pct:.1f}%)")
    
    print(f"\n[目标尺寸分布]")
    print(f"  小目标 (<1%): {box_sizes['small']} ({box_sizes['small']/max(total_boxes,1)*100:.1f}%)")
    print(f"  中目标 (1-10%): {box_sizes['medium']} ({box_sizes['medium']/max(total_boxes,1)*100:.1f}%)")
    print(f"  大目标 (>10%): {box_sizes['large']} ({box_sizes['large']/max(total_boxes,1)*100:.1f}%)")
    
    # 类别不平衡分析
    print(f"\n[类别不平衡分析]")
    max_count = max(class_counts.values()) if class_counts else 1
    min_count = min(class_counts.values()) if class_counts else 1
    imbalance_ratio = max_count / max(min_count, 1)
    print(f"  最大/最小比例: {imbalance_ratio:.1f}:1")
    if imbalance_ratio > 5:
        print("  ⚠️ 严重不平衡! 需要使用class weights或过采样")
    
    return class_counts, box_sizes


def analyze_model_confusion():
    """分析模型在各类别上的表现"""
    print("\n" + "="*60)
    print("模型各类别表现分析")
    print("="*60)
    
    model = YOLO('models/yolov8/train/weights/best.pt')
    
    # 低置信度评估，获取详细结果
    metrics = model.val(data='models/yolov8/data.yaml', conf=0.1, verbose=False)
    
    print(f"\n[默认阈值结果]")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall: {metrics.box.mr:.3f}")
    print(f"  mAP50: {metrics.box.map50:.3f}")
    
    # 尝试获取各类别指标
    if hasattr(metrics.box, 'p') and hasattr(metrics.box, 'r'):
        print(f"\n[各类别 P/R]")
        for i, name in enumerate(CLASS_NAMES):
            if i < len(metrics.box.p) and i < len(metrics.box.r):
                print(f"  {name}: P={metrics.box.p[i]:.3f}, R={metrics.box.r[i]:.3f}")
    
    if hasattr(metrics.box, 'ap50'):
        print(f"\n[各类别 AP50]")
        for i, name in enumerate(CLASS_NAMES):
            if i < len(metrics.box.ap50):
                print(f"  {name}: {metrics.box.ap50[i]:.3f}")


def recommend_optimization():
    """给出优化建议"""
    print("\n" + "="*60)
    print("优化建议")
    print("="*60)
    
    suggestions = """
    根据分析结果，推荐以下优化策略：
    
    1. 【类别不平衡】如果类别不平衡严重：
       - 添加 class weights: cls=2.5, 添加 --class-weight
       - 对少数类使用过采样
    
    2. 【小目标检测】99%是小目标：
       - imgsz=1280 (已经设置)
       - 可以尝试 imgsz=1920 更高分辨率
       - 使用 close_power=2 在neck层
    
    3. 【过拟合/欠拟合】根据cls_loss判断：
       - cls_loss很高 → 欠拟合，需要更多训练或调整
       - 降低学习率 lr0=0.0001
       - 增加warmup时间
    
    4. 【训练策略】重新从头训练：
       - 不从best.pt继续（可能已经过拟合）
       - 使用yolov8s.pt重新训练
       - 使用更强的数据增强
    
    5. 【快速验证】检查标注质量：
       - 随机抽查标注是否正确
       - 检查是否有错标、漏标
    """
    print(suggestions)


def create_balanced_training():
    """创建平衡训练的脚本"""
    print("\n" + "="*60)
    print("创建优化训练脚本...")
    print("="*60)
    
    script = '''"""
PCB缺陷检测 - 平衡优化训练 v5
解决类别不平衡和小目标问题
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import torch

# 配置 - 使用平衡的数据配置
DATA_CONFIG = {
    'path': str(Path('yolo_pcb_dataset').absolute()),
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'names': ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper'],
    'nc': 6
}

Path('models/yolov8').mkdir(parents=True, exist_ok=True)
with open('models/yolov8/data.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(DATA_CONFIG, f, allow_unicode=True)


def train_balanced():
    """平衡优化训练 - 重新从头训练"""
    print("\\n" + "="*60)
    print("开始平衡优化训练 v5")
    print("="*60)
    
    # 重新从头训练，使用yolov8s.pt
    model = YOLO('yolov8s.pt')
    
    results = model.train(
        data='models/yolov8/data.yaml',
        
        # ========== 训练配置 ==========
        epochs=300,
        imgsz=1280,
        batch=4,
        patience=80,
        
        name='train_balanced',
        project='models/yolov8',
        exist_ok=True,
        
        # ========== 优化器 (更稳定) ==========
        optimizer='AdamW',
        lr0=0.0002,        # 更小的初始学习率
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,   # 更长warmup
        warmup_bias_lr=0.1,
        warmup_momentum=0.8,
        
        # ========== 损失函数 ==========
        box=7.5,
        cls=3.0,           # 大幅增加分类损失
        dfl=1.5,
        
        # ========== 设备 ==========
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=0,
        amp=True,
        
        # ========== 数据增强 (平衡策略) ==========
        augment=True,
        mosaic=1.0,
        mixup=0.3,
        copy_paste=0.3,
        
        # 几何增强
        degrees=10.0,
        translate=0.15,
        scale=0.5,
        shear=1.0,
        perspective=0.0005,
        
        # 颜色增强
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        
        # 翻转
        flipud=0.0,
        fliplr=0.5,
        
        # ========== 验证 ==========
        val=True,
        plots=True,
        
        # ========== 推理配置 ==========
        conf=0.1,
        iou=0.5,
        
        verbose=True,
    )
    
    print("\\n训练完成!")


if __name__ == '__main__':
    train_balanced()
'''
    
    script_path = Path('train_v5_balanced.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script)
    
    print(f"已创建: {script_path}")
    print("\\n运行方式:")
    print("  python train_v5_balanced.py")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze-data', action='store_true', help='分析数据集')
    parser.add_argument('--analyze-model', action='store_true', help='分析模型')
    parser.add_argument('--recommend', action='store_true', help='优化建议')
    parser.add_argument('--create', action='store_true', help='创建优化脚本')
    args = parser.parse_args()
    
    if args.analyze_data:
        analyze_dataset()
    
    if args.analyze_model:
        analyze_model_confusion()
    
    if args.recommend:
        recommend_optimization()
    
    if args.create:
        create_balanced_training()
    
    # 默认运行全部分析
    if not any(vars(args).values()):
        analyze_dataset()
        print()
        analyze_model_confusion()
        recommend_optimization()
        create_balanced_training()
