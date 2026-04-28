"""
YOLOv8 PCB缺陷检测 - 带日志的训练脚本
关键位置都添加了日志输出，方便调试和调参
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import numpy as np

# ============ 日志配置 ============
def setup_logger(name='PCB_Training', log_dir='logs', level=logging.INFO):
    """配置日志系统"""
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f'train_{timestamp}.log'
    
    # 配置logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    # 文件Handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(level)
    
    # 控制台Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    
    # 格式化
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_file

logger, log_file = setup_logger()
logger.info(f"=" * 60)
logger.info(f"PCB 缺陷检测训练开始")
logger.info(f"=" * 60)

# 检查依赖
try:
    from ultralytics import YOLO
    logger.info(f"✓ Ultralytics 已加载")
except ImportError:
    logger.error("✗ 请先安装: pip install ultralytics")
    sys.exit(1)

# 检查CUDA
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
logger.info(f"✓ 设备: {device_name}")
logger.info(f"✓ PyTorch: {torch.__version__}")
logger.info(f"✓ CUDA: {torch.cuda.is_available()}")


# ============ 配置加载 ============
class ConfigLoader:
    """配置加载器，带日志输出"""
    
    def __init__(self, config_path='configs/train_config.yaml'):
        self.config_path = Path(config_path)
        self.config = self.load()
    
    def load(self):
        """加载配置文件"""
        if not self.config_path.exists():
            logger.warning(f"配置文件不存在，使用默认配置: {self.config_path}")
            return self.get_default_config()
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✓ 配置文件加载成功: {self.config_path}")
        self._log_config(config)
        return config
    
    def _log_config(self, config):
        """输出配置日志"""
        logger.info("-" * 40)
        logger.info("【配置参数】")
        logger.info("-" * 40)
        
        # 模型配置
        model_cfg = config.get('model', {})
        logger.info(f"  模型: {model_cfg.get('name', 'yolov8n')}")
        logger.info(f"  输入尺寸: {model_cfg.get('imgsz', 640)}")
        logger.info(f"  预训练: {model_cfg.get('pretrained', True)}")
        
        # 数据配置
        data_cfg = config.get('data', {})
        logger.info(f"  数据集: {data_cfg.get('dataset_path', 'N/A')}")
        
        # 训练配置
        train_cfg = config.get('train', {})
        logger.info(f"  Epochs: {train_cfg.get('epochs', 100)}")
        logger.info(f"  Batch: {train_cfg.get('batch', 16)}")
        logger.info(f"  早停耐心值: {train_cfg.get('patience', 20)}")
        logger.info(f"  设备: {train_cfg.get('device', 'auto')}")
        
        # 优化器配置
        opt_cfg = config.get('optimizer', {})
        logger.info(f"  优化器: {opt_cfg.get('name', 'SGD')}")
        logger.info(f"  初始学习率: {opt_cfg.get('lr0', 0.01)}")
        logger.info(f"  最终学习率因子: {opt_cfg.get('lrf', 0.01)}")
        logger.info(f"  权重衰减: {opt_cfg.get('weight_decay', 0.0005)}")
        
        # 数据增强配置
        aug_cfg = config.get('augmentation', {})
        logger.info(f"  【数据增强配置】")
        logger.info(f"    色相: ±{aug_cfg.get('hsv_h', 0) * 100:.1f}%")
        logger.info(f"    饱和度: ±{aug_cfg.get('hsv_s', 0) * 100:.1f}%")
        logger.info(f"    亮度: ±{aug_cfg.get('hsv_v', 0) * 100:.1f}%")
        logger.info(f"    旋转: ±{aug_cfg.get('degrees', 0)}°")
        logger.info(f"    平移: ±{aug_cfg.get('translate', 0) * 100:.1f}%")
        logger.info(f"    缩放: {aug_cfg.get('scale', 0) * 100:.1f}%")
        logger.info(f"    水平翻转: {aug_cfg.get('fliplr', 0) * 100:.1f}%")
        logger.info(f"    Mosaic: {aug_cfg.get('mosaic', 1.0)}")
        logger.info(f"    MixUp: {aug_cfg.get('mixup', 0.0)}")
        logger.info(f"    CopyPaste: {aug_cfg.get('copy_paste', 0.0)}")
        logger.info("-" * 40)
        
        return config
    
    def get_default_config(self):
        """默认配置"""
        return {
            'model': {
                'name': 'yolov8n',
                'pretrained': True,
                'imgsz': 640,
            },
            'data': {
                'dataset_path': 'yolo_pcb_dataset',
                'yaml_path': 'models/yolov8/data.yaml',
            },
            'train': {
                'epochs': 100,
                'batch': 16,
                'imgsz': 640,
                'patience': 20,
                'save_period': 10,
                'device': 0,
            },
            'optimizer': {
                'name': 'AdamW',
                'lr0': 0.001,
                'lrf': 0.01,
                'weight_decay': 0.0005,
                'momentum': 0.937,
            },
            'augmentation': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.15,
                'copy_paste': 0.1,
            },
            'inference': {
                'conf': 0.25,
                'iou': 0.45,
            },
            'export': {
                'format': 'onnx',
                'simplify': True,
                'opset': 12,
                'half': False,
            }
        }


# ============ 数据集分析 ============
class DatasetAnalyzer:
    """数据集分析器"""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.stats = {}
    
    def analyze(self):
        """分析数据集并输出统计"""
        logger.info("=" * 40)
        logger.info("【数据集分析】")
        logger.info("=" * 40)
        
        # 检查数据集结构
        train_img = self.dataset_path / 'images' / 'train'
        val_img = self.dataset_path / 'images' / 'val'
        test_img = self.dataset_path / 'images' / 'test'
        
        for split, path in [('Train', train_img), ('Val', val_img), ('Test', test_img)]:
            if path.exists():
                count = len(list(path.glob('*.jpg'))) + len(list(path.glob('*.png')))
                logger.info(f"  {split}: {count} 张图片")
            else:
                logger.warning(f"  {split}: 目录不存在")
        
        # 分析类别分布
        self._analyze_classes()
        
        # 分析图像尺寸
        self._analyze_image_sizes()
        
        logger.info("=" * 40)
    
    def _analyze_classes(self):
        """分析类别分布"""
        labels_dir = self.dataset_path / 'labels'
        if not labels_dir.exists():
            logger.warning("  标签目录不存在")
            return
        
        # 统计各类别数量
        class_counts = Counter()
        total_objects = 0
        
        for label_file in labels_dir.rglob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        total_objects += 1
        
        # 类别名称映射
        class_names = {
            0: 'missing_hole',
            1: 'mouse_bite',
            2: 'open_circuit',
            3: 'short',
            4: 'spur',
            5: 'spurious_copper',
            6: 'normal'
        }
        
        logger.info("  类别分布:")
        for class_id in sorted(class_counts.keys()):
            name = class_names.get(class_id, f'class_{class_id}')
            count = class_counts[class_id]
            pct = count / total_objects * 100 if total_objects > 0 else 0
            logger.info(f"    {name}: {count} ({pct:.1f}%)")
        
        self.stats['class_counts'] = dict(class_counts)
        self.stats['total_objects'] = total_objects
    
    def _analyze_image_sizes(self):
        """分析图像尺寸分布"""
        from PIL import Image
        
        sample_images = list((self.dataset_path / 'images' / 'train').glob('*.jpg'))[:50]
        if not sample_images:
            return
        
        sizes = []
        for img_path in sample_images:
            try:
                img = Image.open(img_path)
                sizes.append(img.size)
            except:
                continue
        
        if sizes:
            widths, heights = zip(*sizes)
            logger.info(f"  图像尺寸 (采样{sample_images.__len__()}张):")
            logger.info(f"    宽度: min={min(widths)}, max={max(widths)}, avg={np.mean(widths):.0f}")
            logger.info(f"    高度: min={min(heights)}, max={max(heights)}, avg={np.mean(heights):.0f}")


# ============ 训练器 ============
class PCBTrainer:
    """PCB检测训练器"""
    
    def __init__(self, config):
        self.config = config
        self.project_dir = Path('models/yolov8')
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.best_model_path = None
    
    def prepare_dataset_config(self):
        """准备数据集配置"""
        config = {
            'path': str(Path(self.config['data']['dataset_path']).absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {
                0: 'missing_hole',
                1: 'mouse_bite',
                2: 'open_circuit',
                3: 'short',
                4: 'spur',
                5: 'spurious_copper',
                6: 'normal'
            },
            'nc': 7
        }
        
        config_path = self.project_dir / 'data.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        
        logger.info(f"✓ 数据集配置已保存: {config_path}")
        return config_path
    
    def load_model(self):
        """加载模型"""
        model_cfg = self.config['model']
        model_name = model_cfg.get('name', 'yolov8n.pt')
        
        # 检查是否有微调过的模型
        best_path = self.project_dir / 'weights' / 'best.pt'
        if best_path.exists():
            logger.info(f"→ 加载已有模型: {best_path}")
            self.model = YOLO(str(best_path))
        else:
            logger.info(f"→ 加载预训练模型: {model_name}")
            self.model = YOLO(model_name)
        
        logger.info(f"✓ 模型加载完成")
        return self.model
    
    def train(self):
        """执行训练"""
        logger.info("=" * 40)
        logger.info("【开始训练】")
        logger.info("=" * 40)
        
        train_cfg = self.config['train']
        aug_cfg = self.config['augmentation']
        opt_cfg = self.config['optimizer']
        
        # 日志关键参数
        logger.info(f"  训练参数:")
        logger.info(f"    Epochs: {train_cfg.get('epochs', 100)}")
        logger.info(f"    Batch size: {train_cfg.get('batch', 16)}")
        logger.info(f"    Image size: {train_cfg.get('imgsz', 640)}")
        logger.info(f"    Device: {train_cfg.get('device', 0)}")
        
        # 设备选择
        device = train_cfg.get('device', 0)
        if device == 'auto':
            device = 0 if torch.cuda.is_available() else 'cpu'
        logger.info(f"  使用设备: {device}")
        
        # 开始训练
        start_time = datetime.now()
        logger.info(f"  训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            results = self.model.train(
                # 数据配置
                data=str(self.prepare_dataset_config()),
                
                # 训练参数
                epochs=train_cfg.get('epochs', 100),
                imgsz=train_cfg.get('imgsz', 640),
                batch=train_cfg.get('batch', 16),
                patience=train_cfg.get('patience', 20),
                save_period=train_cfg.get('save_period', 10),
                
                # 项目配置
                project=str(self.project_dir),
                name='train',
                exist_ok=True,
                
                # 优化器配置
                optimizer=opt_cfg.get('name', 'AdamW'),
                lr0=opt_cfg.get('lr0', 0.001),
                lrf=opt_cfg.get('lrf', 0.01),
                momentum=opt_cfg.get('momentum', 0.937),
                weight_decay=opt_cfg.get('weight_decay', 0.0005),
                
                # 学习率调度
                warmup_epochs=3,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                cos_lr=True,
                
                # 数据增强
                hsv_h=aug_cfg.get('hsv_h', 0.015),
                hsv_s=aug_cfg.get('hsv_s', 0.7),
                hsv_v=aug_cfg.get('hsv_v', 0.4),
                degrees=aug_cfg.get('degrees', 0.0),
                translate=aug_cfg.get('translate', 0.1),
                scale=aug_cfg.get('scale', 0.5),
                shear=aug_cfg.get('shear', 0.0),
                perspective=aug_cfg.get('perspective', 0.0),
                flipud=aug_cfg.get('flipud', 0.0),
                fliplr=aug_cfg.get('fliplr', 0.5),
                mosaic=aug_cfg.get('mosaic', 1.0),
                mixup=aug_cfg.get('mixup', 0.0),
                copy_paste=aug_cfg.get('copy_paste', 0.0),
                
                # 其他配置
                close_mosaic=10,
                workers=0,
                device=device,
                verbose=True,
                amp=True,
                val=True,
                plots=True,
                conf=self.config['inference'].get('conf', 0.25),
                iou=self.config['inference'].get('iou', 0.45),
            )
            
            # 训练完成
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"  训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"  总耗时: {duration}")
            
            # 获取最佳模型路径
            self.best_model_path = self.project_dir / 'train' / 'weights' / 'best.pt'
            if self.best_model_path.exists():
                size_mb = self.best_model_path.stat().st_size / (1024 * 1024)
                logger.info(f"✓ 最佳模型: {self.best_model_path} ({size_mb:.1f}MB)")
            
            return results
            
        except Exception as e:
            logger.error(f"训练出错: {e}")
            raise
    
    def evaluate(self):
        """评估模型"""
        if self.best_model_path is None:
            self.best_model_path = self.project_dir / 'train' / 'weights' / 'best.pt'
        
        if not self.best_model_path.exists():
            logger.warning("模型文件不存在，跳过评估")
            return
        
        logger.info("=" * 40)
        logger.info("【模型评估】")
        logger.info("=" * 40)
        
        self.model = YOLO(str(self.best_model_path))
        
        # 评估
        metrics = self.model.val(
            data=str(self.prepare_dataset_config()),
            imgsz=self.config['train'].get('imgsz', 640),
            batch=self.config['train'].get('batch', 16),
            conf=self.config['inference'].get('conf', 0.25),
            iou=self.config['inference'].get('iou', 0.45),
            verbose=True,
        )
        
        # 输出指标
        logger.info("  评估结果:")
        logger.info(f"    mAP@50: {metrics.box.map50:.4f}")
        logger.info(f"    mAP@50-95: {metrics.box.map:.4f}")
        logger.info(f"    Precision: {metrics.box.mp:.4f}")
        logger.info(f"    Recall: {metrics.box.mr:.4f}")
        
        # 各类别AP
        if hasattr(metrics.box, 'ap_class_index'):
            class_names = ['missing_hole', 'mouse_bite', 'open_circuit', 
                          'short', 'spur', 'spurious_copper', 'normal']
            logger.info("  各类别 AP@50:")
            for i, ap in enumerate(metrics.box.ap50):
                if i < len(class_names):
                    logger.info(f"    {class_names[i]}: {ap:.4f}")
        
        return metrics
    
    def export_model(self):
        """导出模型"""
        if self.best_model_path is None:
            self.best_model_path = self.project_dir / 'train' / 'weights' / 'best.pt'
        
        if not self.best_model_path.exists():
            logger.warning("模型文件不存在，跳过导出")
            return
        
        logger.info("=" * 40)
        logger.info("【模型导出】")
        logger.info("=" * 40)
        
        export_cfg = self.config['export']
        fmt = export_cfg.get('format', 'onnx')
        imgsz = self.config['model'].get('imgsz', 640)
        
        logger.info(f"  导出格式: {fmt}")
        logger.info(f"  输入尺寸: {imgsz}")
        
        self.model = YOLO(str(self.best_model_path))
        
        try:
            exported_path = self.model.export(
                format=fmt,
                imgsz=imgsz,
                simplify=export_cfg.get('simplify', True),
                opset=export_cfg.get('opset', 12),
                half=export_cfg.get('half', False),
            )
            
            size_mb = Path(exported_path).stat().st_size / (1024 * 1024)
            logger.info(f"✓ 导出成功: {exported_path} ({size_mb:.1f}MB)")
            
        except Exception as e:
            logger.error(f"导出失败: {e}")
    
    def benchmark(self):
        """性能基准测试"""
        if self.best_model_path is None:
            self.best_model_path = self.project_dir / 'train' / 'weights' / 'best.pt'
        
        logger.info("=" * 40)
        logger.info("【性能基准测试】")
        logger.info("=" * 40)
        
        import time
        
        # 测试 PyTorch
        try:
            self.model = YOLO(str(self.best_model_path))
            times = []
            for _ in range(100):
                start = time.time()
                self.model.predict(imgsz=640, verbose=False)
                times.append(time.time() - start)
            
            avg_time = np.median(times) * 1000
            fps = 1000 / avg_time
            logger.info(f"  PyTorch: {avg_time:.2f}ms/frame, {fps:.1f} FPS")
        except Exception as e:
            logger.warning(f"  PyTorch: 测试失败 - {e}")
        
        # 测试 ONNX
        onnx_path = str(self.best_model_path).replace('.pt', '.onnx')
        if Path(onnx_path).exists():
            try:
                self.model = YOLO(onnx_path)
                times = []
                for _ in range(100):
                    start = time.time()
                    self.model.predict(imgsz=640, verbose=False)
                    times.append(time.time() - start)
                
                avg_time = np.median(times) * 1000
                fps = 1000 / avg_time
                logger.info(f"  ONNX: {avg_time:.2f}ms/frame, {fps:.1f} FPS")
            except Exception as e:
                logger.warning(f"  ONNX: 测试失败 - {e}")


# ============ 主函数 ============
def main():
    parser = argparse.ArgumentParser(description='YOLOv8 PCB缺陷检测训练 (带日志)')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', 
                       help='配置文件路径')
    parser.add_argument('--data', type=str, default='yolo_pcb_dataset', 
                       help='数据集路径')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch', type=int, default=None, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=None, help='图像尺寸')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--model', type=str, default=None, choices=['n','s','m','l','x'],
                       help='模型大小')
    parser.add_argument('--analyze', action='store_true', help='仅分析数据集')
    parser.add_argument('--train', action='store_true', help='开始训练')
    parser.add_argument('--eval', action='store_true', help='评估模型')
    parser.add_argument('--export', action='store_true', help='导出模型')
    parser.add_argument('--benchmark', action='store_true', help='性能测试')
    parser.add_argument('--all', action='store_true', help='执行全部流程')
    
    args = parser.parse_args()
    
    # 加载配置
    config = ConfigLoader(args.config).config
    
    # 命令行参数覆盖配置
    if args.epochs:
        config['train']['epochs'] = args.epochs
        logger.info(f"→ 命令行覆盖 epochs: {args.epochs}")
    if args.batch:
        config['train']['batch'] = args.batch
        logger.info(f"→ 命令行覆盖 batch: {args.batch}")
    if args.imgsz:
        config['train']['imgsz'] = args.imgsz
        logger.info(f"→ 命令行覆盖 imgsz: {args.imgsz}")
    if args.lr:
        config['optimizer']['lr0'] = args.lr
        logger.info(f"→ 命令行覆盖 lr: {args.lr}")
    if args.model:
        config['model']['name'] = f'yolov8{args.model}.pt'
        logger.info(f"→ 命令行覆盖 model: {config['model']['name']}")
    if args.data:
        config['data']['dataset_path'] = args.data
        logger.info(f"→ 命令行覆盖 data: {args.data}")
    
    # 创建训练器
    trainer = PCBTrainer(config)
    
    # 分析数据集
    if args.analyze or args.all:
        analyzer = DatasetAnalyzer(config['data']['dataset_path'])
        analyzer.analyze()
    
    # 训练
    if args.train or args.all:
        trainer.load_model()
        trainer.train()
    
    # 评估
    if args.eval or args.all:
        trainer.evaluate()
    
    # 导出
    if args.export or args.all:
        trainer.export_model()
    
    # 基准测试
    if args.benchmark or args.all:
        trainer.benchmark()
    
    if args.all:
        logger.info("=" * 60)
        logger.info("全部流程执行完成!")
        logger.info(f"日志文件: {log_file}")
        logger.info("=" * 60)


if __name__ == '__main__':
    main()
