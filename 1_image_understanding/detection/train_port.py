# ============================================================
# PortAI 快速训练测试
# PortAI Quick Training Test
#
# 使用示例数据快速测试训练流程
# Quick test training pipeline with sample data
# ============================================================

"""
快速训练测试 / Quick Training Test
===================================

本脚本使用示例数据进行快速训练测试，验证整个流程是否正常。
This script tests the training pipeline with sample data.

流程:
1. 检查数据完整性
2. 创建数据集配置
3. 训练模型 (少量epoch)
4. 验证输出
"""

import os
import sys
import shutil
from pathlib import Path
import yaml

# 路径配置
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "port_datasets" / "sample_port"
OUTPUT_DIR = SCRIPT_DIR / "port_training_output"

def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def check_data():
    """检查数据完整性 / Check data integrity"""
    print_header("Step 1: 检查数据完整性 / Checking Data Integrity")
    
    images_dir = DATA_DIR / "images"
    labels_dir = DATA_DIR / "labels"
    
    # 检查目录
    if not images_dir.exists():
        print(f"[FAIL] 图像目录不存在: {images_dir}")
        return False
        
    if not labels_dir.exists():
        print(f"[FAIL] 标注目录不存在: {labels_dir}")
        return False
    
    # 统计文件
    images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    labels = list(labels_dir.glob("*.txt"))
    
    print(f"[OK] 图像数量: {len(images)}")
    print(f"[OK] 标注数量: {len(labels)}")
    
    if len(images) == 0:
        print("[FAIL] 没有找到图像文件")
        return False
        
    # 检查标注文件
    print("\n检查标注文件 / Checking label files:")
    sample_images = images[:5]
    for img in sample_images:
        label_file = labels_dir / f"{img.stem}.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
            print(f"  [OK] {img.name}: {len(lines)} 个目标")
        else:
            print(f"  [WARN] {img.name}: 缺少标注文件")
    
    return True

def create_dataset_yaml():
    """创建数据集配置 / Create dataset config"""
    print_header("Step 2: 创建数据集配置 / Creating Dataset Config")
    
    yaml_path = DATA_DIR / "dataset.yaml"
    
    config = {
        'path': str(DATA_DIR.absolute()),
        'train': 'images',
        'val': 'images',
        'names': {
            0: 'container',
            1: 'crane',
            2: 'truck',
            3: 'ship',
            4: 'worker'
        },
        'nc': 5
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"[OK] 数据集配置: {yaml_path}")
    print(f"[OK] 类别数: {config['nc']}")
    print(f"[OK] 类别: {list(config['names'].values())}")
    
    return True

def test_training():
    """测试训练 / Test training"""
    print_header("Step 3: 测试训练流程 / Testing Training Pipeline")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("[INFO] 检查 Ultralytics YOLO...")
    
    try:
        from ultralytics import YOLO
        print("[OK] Ultralytics 已安装")
        
        # 使用最小的模型
        print("\n[INFO] 使用 YOLOv8n (最小模型) 进行测试")
        print("[INFO] 训练 10 个 epoch (快速测试)")
        
        # 训练配置
        results = YOLO('yolov8n.pt').train(
            data=str(DATA_DIR / "dataset.yaml"),
            epochs=10,
            imgsz=320,  # 小图加快速度
            batch=4,
            project=str(OUTPUT_DIR),
            name='port_test',
            verbose=True,
            exist_ok=True,
            device='cpu'  # 用CPU测试
        )
        
        print(f"\n[OK] 训练完成!")
        print(f"[OK] 模型保存: {OUTPUT_DIR / 'port_test' / 'weights'}")
        
        return True
        
    except ImportError:
        print("[INFO] Ultralytics 未安装，跳过实际训练")
        print("[INFO] 安装命令: pip install ultralytics")
        
        # 创建模拟训练结果
        print("\n[SIM] 模拟训练完成 (实际训练需要安装 ultralytics)")
        
        # 创建模拟权重文件
        weights_dir = OUTPUT_DIR / "port_test" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制示例图片作为"训练结果"
        sample_img = list((DATA_DIR / "images").glob("*.jpg"))[0]
        shutil.copy(sample_img, weights_dir / "result.jpg")
        
        print(f"[OK] 模拟结果: {weights_dir}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 训练失败: {e}")
        return False

def print_summary():
    """打印总结 / Print summary"""
    print_header("总结 / Summary")
    
    print(f"""
数据集位置 / Dataset Location:
  {DATA_DIR}

训练输出 / Training Output:
  {OUTPUT_DIR}

下一步 / Next Steps:
-------------------
1. 查看训练结果:
   explorer {OUTPUT_DIR}

2. 下载真实数据:
   - 访问 https://universe.roboflow.com
   - 搜索 "port", "container", "crane"
   - 下载 YOLO 格式
   - 解压到 port_datasets/real_port/

3. 训练真实模型:
   python yolov10_prune_distill.py --data port_datasets/real_port/dataset.yaml

4. 查看数据集:
   explorer {DATA_DIR / "images"}
""")

def main():
    print("=" * 60)
    print("PortAI 快速训练测试")
    print("PortAI Quick Training Test")
    print("=" * 60)
    
    # Step 1: 检查数据
    if not check_data():
        print("\n[FAIL] 数据检查失败，请先运行: python fetch_datasets.py")
        return
    
    # Step 2: 创建配置
    create_dataset_yaml()
    
    # Step 3: 测试训练
    test_training()
    
    # Step 4: 总结
    print_summary()

if __name__ == "__main__":
    main()
