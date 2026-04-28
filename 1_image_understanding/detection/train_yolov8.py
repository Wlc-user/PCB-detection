"""
YOLOv8 PCB缺陷检测模型训练
用于替换旧的MobileNetV2模型，提升检测精度和速度
"""

import os
import yaml
import torch
import argparse
from pathlib import Path

# 检查并安装依赖
def check_install_dependencies():
    """检查并安装必要的依赖"""
    try:
        import ultralytics
    except ImportError:
        print("Installing ultralytics...")
        os.system("pip install ultralytics -q")

check_install_dependencies()

from ultralytics import YOLO
import shutil


class YOLOTrainer:
    """YOLOv8训练器"""
    
    def __init__(self, dataset_path, model_size='n'):
        """
        初始化训练器
        
        Args:
            dataset_path: 数据集路径
            model_size: 模型大小 [n/s/m/l/x]
        """
        self.dataset_path = dataset_path
        self.model_size = model_size
        self.project_dir = Path("models/yolov8")
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_dataset_config(self):
        """准备数据集配置文件"""
        config = {
            'path': str(Path(self.dataset_path).absolute()),
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
        
        print(f"数据集配置已保存: {config_path}")
        return config_path
    
    def train(self, epochs=100, imgsz=640, batch=16, patience=20):
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            imgsz: 输入图像尺寸
            batch: 批次大小
            patience: 早停耐心值
        """
        # 使用现有的best.pt作为预训练模型
        model_path = str(self.project_dir / 'weights' / 'best.pt')
        if Path(model_path).exists():
            print(f"Loading existing model: {model_path}")
            model = YOLO(model_path)
        else:
            # 如果没有现有模型，从头下载
            model_name = f'yolov8{self.model_size}.pt' if self.model_size else 'yolov8n.pt'
            print(f"Loading model: {model_name}")
            model = YOLO(model_name)
        
        # 训练参数
        results = model.train(
            data=str(self.prepare_dataset_config()),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            project=str(self.project_dir),
            name='train',
            exist_ok=True,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=3,
            close_mosaic=10,
            workers=0,
            device=0 if __import__('torch').cuda.is_available() else 'cpu',
            verbose=True,
            amp=True,  # 启用AMP
            # 数据增强
            augment=True,
            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.1,
            # 验证
            val=True,
            plots=True,
            # 推理优化
            conf=0.001,
            iou=0.6,
        )
        
        print("\n" + "="*50)
        print("训练完成!")
        print("="*50)
        
        # 打印最佳模型路径
        best_model = self.project_dir / 'train' / 'weights' / 'best.pt'
        if best_model.exists():
            print(f"\n最佳模型: {best_model}")
        
        last_model = self.project_dir / 'train' / 'weights' / 'last.pt'
        if last_model.exists():
            print(f"最新模型: {last_model}")
        
        return results
    
    def export_onnx(self, model_path=None, imgsz=640):
        """
        导出ONNX模型
        
        Args:
            model_path: 模型路径
            imgsz: 输入尺寸
        """
        if model_path is None:
            model_path = self.project_dir / 'train' / 'weights' / 'best.pt'
        
        model = YOLO(str(model_path))
        
        # 导出ONNX
        onnx_path = model.export(format='onnx', imgsz=imgsz, dynamic=False)
        
        print(f"ONNX模型已导出: {onnx_path}")
        return onnx_path
    
    def export_tensorrt(self, model_path=None, imgsz=640):
        """
        导出TensorRT模型 (需要安装tensorrt)
        
        Args:
            model_path: 模型路径
            imgsz: 输入尺寸
        """
        if model_path is None:
            model_path = self.project_dir / 'train' / 'weights' / 'best.pt'
        
        model = YOLO(str(model_path))
        
        # 导出TensorRT
        trt_path = model.export(format='engine', imgsz=imgsz, device=0)
        
        print(f"TensorRT模型已导出: {trt_path}")
        return trt_path
    
    def evaluate(self, model_path=None, data_path=None):
        """
        评估模型
        
        Args:
            model_path: 模型路径
            data_path: 数据集路径
        """
        if model_path is None:
            model_path = self.project_dir / 'train' / 'weights' / 'best.pt'
        
        if data_path is None:
            data_path = self.prepare_dataset_config()
        
        model = YOLO(str(model_path))
        
        # 评估
        metrics = model.val(data=str(data_path))
        
        print("\n" + "="*50)
        print("评估结果:")
        print("="*50)
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def benchmark(self, model_path=None, imgsz=640):
        """
        性能基准测试
        
        Args:
            model_path: 模型路径
            imgsz: 输入尺寸
        """
        if model_path is None:
            model_path = self.project_dir / 'train' / 'weights' / 'best.pt'
        
        model = YOLO(str(model_path))
        
        print("\n" + "="*50)
        print("性能基准测试")
        print("="*50)
        
        # 测试不同推理后端
        backends = ['onnx', 'engine', 'torch']
        
        for backend in backends:
            try:
                if backend == 'torch':
                    # PyTorch推理
                    import time
                    model = YOLO(str(model_path))
                    times = []
                    for _ in range(100):
                        start = time.time()
                        model.predict(imgsz=imgsz, verbose=False)
                        times.append(time.time() - start)
                    
                    avg_time = sum(times) / len(times)
                    fps = 1 / avg_time
                    print(f"PyTorch: {avg_time*1000:.2f}ms/frame, {fps:.1f} FPS")
                else:
                    # ONNX/TensorRT
                    ext = '.onnx' if backend == 'onnx' else '.engine'
                    alt_model = str(model_path).replace('.pt', ext)
                    if os.path.exists(alt_model):
                        import time
                        model = YOLO(alt_model)
                        times = []
                        for _ in range(100):
                            start = time.time()
                            model.predict(imgsz=imgsz, verbose=False)
                            times.append(time.time() - start)
                        
                        avg_time = sum(times) / len(times)
                        fps = 1 / avg_time
                        print(f"{backend.upper()}: {avg_time*1000:.2f}ms/frame, {fps:.1f} FPS")
            except Exception as e:
                print(f"{backend.upper()}: 未安装或不支持")


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 PCB缺陷检测训练')
    parser.add_argument('--data', type=str, default='./yolo_pcb_dataset', help='数据集路径')
    parser.add_argument('--model', type=str, default='n', choices=['n','s','m','l','x'], help='模型大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--train', action='store_true', help='开始训练')
    parser.add_argument('--export-onnx', action='store_true', help='导出ONNX')
    parser.add_argument('--export-trt', action='store_true', help='导出TensorRT')
    parser.add_argument('--eval', action='store_true', help='评估模型')
    parser.add_argument('--benchmark', action='store_true', help='性能测试')
    
    args = parser.parse_args()
    
    trainer = YOLOTrainer(args.data, args.model)
    
    if args.train:
        trainer.train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
    
    if args.export_onnx:
        trainer.export_onnx(imgsz=args.imgsz)
    
    if args.export_trt:
        trainer.export_tensorrt(imgsz=args.imgsz)
    
    if args.eval:
        trainer.evaluate()
    
    if args.benchmark:
        trainer.benchmark(imgsz=args.imgsz)


if __name__ == '__main__':
    main()
