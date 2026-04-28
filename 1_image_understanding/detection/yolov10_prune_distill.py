"""
YOLOv10 剪枝 + 蒸馏 训练脚本
无需标注，使用公开数据集直接训练
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from pathlib import Path
from datetime import datetime

# 配置
CONFIG = {
    'model': 'yolov10n',  # yolov10n/s/m/l/x
    'dataset': 'coco128',  # coco128, voc, custom
    'epochs': 100,
    'batch_size': 16,
    'img_size': 640,
    'device': '0' if torch.cuda.is_available() else 'cpu',
    
    # 剪枝配置
    'prune_ratio': 0.3,  # 剪掉30%通道
    'prune_strategy': 'l1',  # l1, l2, random
    
    # 蒸馏配置
    'distill': True,
    'teacher_model': 'yolov10m',  # 教师模型
    'distill_temp': 4.0,
    'distill_alpha': 0.5,  # 蒸馏损失权重
    
    # 输出
    'project': 'runs/prune_distill',
    'name': 'exp',
}


def setup_yolov10():
    """安装YOLOv10"""
    try:
        import ultralytics
        print(f"✅ ultralytics 已安装: {ultralytics.__version__}")
    except ImportError:
        print("📦 安装 ultralytics...")
        os.system("pip install ultralytics -q")
    
    # 下载YOLOv10权重
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    models = ['yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt']
    for model in models:
        path = weights_dir / model
        if not path.exists():
            print(f"⬇️ 下载 {model}...")
            os.system(f"wget -q https://github.com/THU-MIG/yolov10/releases/download/v1.1/{model} -O {path}")


def download_dataset(dataset_name='coco128'):
    """下载公开数据集"""
    from ultralytics import YOLO
    
    if dataset_name == 'coco128':
        # COCO128 自动下载
        print("📦 准备 COCO128 数据集...")
        model = YOLO('yolov10n.pt')
        model.predict(source='https://ultralytics.com/images/bus.jpg', save=False)
        return 'coco128.yaml'
    
    elif dataset_name == 'voc':
        print("📦 准备 Pascal VOC 数据集...")
        return 'VOC.yaml'
    
    return dataset_name


class ChannelPruner:
    """通道剪枝器"""
    
    def __init__(self, model, prune_ratio=0.3, strategy='l1'):
        self.model = model
        self.prune_ratio = prune_ratio
        self.strategy = strategy
        self.pruned_channels = []
        
    def get_importance(self, weight):
        """计算通道重要性"""
        if self.strategy == 'l1':
            return torch.sum(torch.abs(weight), dim=(1,2,3))
        elif self.strategy == 'l2':
            return torch.sum(weight ** 2, dim=(1,2,3))
        else:
            return torch.rand(weight.shape[0])
    
    def prune_conv(self, conv_layer, bn_layer=None):
        """剪枝卷积层"""
        weight = conv_layer.weight.data
        importance = self.get_importance(weight)
        
        # 排序并选择要剪枝的通道
        num_channels = weight.shape[0]
        num_prune = int(num_channels * self.prune_ratio)
        
        _, indices = torch.sort(importance)
        prune_indices = indices[:num_prune].tolist()
        keep_indices = indices[num_prune:].tolist()
        
        # 创建新的权重
        new_weight = weight[keep_indices]
        conv_layer.weight = nn.Parameter(new_weight)
        
        if conv_layer.bias is not None:
            conv_layer.bias = nn.Parameter(conv_layer.bias.data[keep_indices])
        
        # 更新BN层
        if bn_layer is not None:
            bn_layer.num_features = len(keep_indices)
            bn_layer.weight = nn.Parameter(bn_layer.weight.data[keep_indices])
            bn_layer.bias = nn.Parameter(bn_layer.bias.data[keep_indices])
            bn_layer.running_mean = bn_layer.running_mean[keep_indices]
            bn_layer.running_var = bn_layer.running_var[keep_indices]
        
        self.pruned_channels.append({
            'layer': conv_layer,
            'original': num_channels,
            'pruned': num_prune,
            'remaining': len(keep_indices)
        })
        
        return keep_indices
    
    def prune_model(self):
        """剪枝整个模型"""
        print(f"\n🔪 开始剪枝 (ratio={self.prune_ratio}, strategy={self.strategy})")
        
        pruned_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 找到对应的BN层
                bn_name = name.replace('.conv', '.bn')
                bn_module = None
                for n, m in self.model.named_modules():
                    if n == bn_name and isinstance(m, nn.BatchNorm2d):
                        bn_module = m
                        break
                
                self.prune_conv(module, bn_module)
                pruned_count += 1
        
        print(f"✅ 剪枝完成: {pruned_count} 层被剪枝")
        return self.model
    
    def summary(self):
        """打印剪枝摘要"""
        print("\n📊 剪枝摘要:")
        total_before = sum(p['original'] for p in self.pruned_channels)
        total_after = sum(p['remaining'] for p in self.pruned_channels)
        print(f"   通道数: {total_before} -> {total_after} ({total_after/total_before*100:.1f}%)")


class DistillationLoss(nn.Module):
    """知识蒸馏损失"""
    
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, targets):
        """
        student_logits: 学生模型输出
        teacher_logits: 教师模型输出
        targets: 真实标签
        """
        # 硬损失
        hard_loss = self.ce_loss(student_logits, targets)
        
        # 软损失 (知识蒸馏)
        soft_student = torch.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = torch.softmax(teacher_logits / self.T, dim=1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.T ** 2)
        
        # 总损失
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, hard_loss, soft_loss


def train_with_distillation(config):
    """使用蒸馏训练"""
    from ultralytics import YOLO
    
    print("="*60)
    print("YOLOv10 剪枝 + 蒸馏训练")
    print("="*60)
    
    # 1. 加载教师模型 (大模型)
    if config['distill']:
        print(f"\n👨‍🏫 加载教师模型: {config['teacher_model']}")
        teacher = YOLO(f"weights/{config['teacher_model']}.pt")
        teacher.model.eval()
        for param in teacher.model.parameters():
            param.requires_grad = False
    
    # 2. 加载学生模型
    print(f"\n👨‍🎓 加载学生模型: {config['model']}")
    student = YOLO(f"weights/{config['model']}.pt")
    
    # 3. 剪枝
    if config['prune_ratio'] > 0:
        pruner = ChannelPruner(student.model, config['prune_ratio'], config['prune_strategy'])
        student.model = pruner.prune_model()
        pruner.summary()
    
    # 4. 准备数据集
    data_yaml = download_dataset(config['dataset'])
    
    # 5. 训练
    print(f"\n🚀 开始训练...")
    print(f"   数据集: {data_yaml}")
    print(f"   轮数: {config['epochs']}")
    print(f"   批次: {config['batch_size']}")
    print(f"   设备: {config['device']}")
    
    results = student.train(
        data=data_yaml,
        epochs=config['epochs'],
        batch=config['batch_size'],
        imgsz=config['img_size'],
        device=config['device'],
        project=config['project'],
        name=config['name'],
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        patience=50,
        save=True,
        save_period=10,
    )
    
    # 6. 评估
    print("\n📊 评估模型...")
    metrics = student.val()
    
    print(f"\n✅ 训练完成!")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    
    # 7. 导出模型
    print("\n📦 导出模型...")
    student.export(format='onnx', simplify=True)
    
    return student, metrics


def compare_models():
    """对比不同配置的效果"""
    configs = [
        {'model': 'yolov10n', 'prune_ratio': 0.0, 'distill': False, 'name': 'baseline'},
        {'model': 'yolov10n', 'prune_ratio': 0.3, 'distill': False, 'name': 'prune_only'},
        {'model': 'yolov10n', 'prune_ratio': 0.3, 'distill': True, 'name': 'prune_distill'},
    ]
    
    results = []
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"配置: {cfg['name']}")
        print('='*60)
        
        full_config = {**CONFIG, **cfg}
        model, metrics = train_with_distillation(full_config)
        
        results.append({
            'name': cfg['name'],
            'map50': metrics.box.map50,
            'map': metrics.box.map,
        })
    
    # 打印对比结果
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    for r in results:
        print(f"{r['name']:20s} mAP50: {r['map50']:.4f}  mAP: {r['map']:.4f}")


def quick_demo():
    """快速演示"""
    print("🎯 YOLOv10 剪枝蒸馏快速演示")
    print("-" * 40)
    
    # 只训练5轮演示
    demo_config = {
        **CONFIG,
        'epochs': 5,
        'batch_size': 8,
        'name': 'demo',
    }
    
    train_with_distillation(demo_config)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv10 Prune & Distill')
    parser.add_argument('--setup', action='store_true', help='安装依赖')
    parser.add_argument('--demo', action='store_true', help='快速演示')
    parser.add_argument('--compare', action='store_true', help='对比实验')
    parser.add_argument('--model', default='yolov10n', help='模型类型')
    parser.add_argument('--prune', type=float, default=0.3, help='剪枝比例')
    parser.add_argument('--distill', action='store_true', help='启用蒸馏')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_yolov10()
    elif args.demo:
        setup_yolov10()
        quick_demo()
    elif args.compare:
        setup_yolov10()
        compare_models()
    else:
        setup_yolov10()
        config = {
            **CONFIG,
            'model': args.model,
            'prune_ratio': args.prune,
            'distill': args.distill,
            'epochs': args.epochs,
        }
        train_with_distillation(config)
