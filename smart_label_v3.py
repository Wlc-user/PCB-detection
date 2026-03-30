"""
PCB缺陷智能标注工具V3 - 基于同模板对比
使用同一PCB模板的第一张图片作为参考，进行差分检测
"""
import cv2
import numpy as np
from pathlib import Path
import os

class PCBTemplatedLabeler:
    """基于模板对比的智能标注"""
    
    DEFECT_TYPES = {
        'missing_hole': 0,
        'mouse_bite': 1, 
        'open_circuit': 2,
        'short': 3,
        'spur': 4,
        'spurious_copper': 5,
    }
    
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.templates = {}  # 缓存模板图片
        self._load_templates()
    
    def _load_templates(self):
        """加载所有PCB模板"""
        train_dir = self.dataset_dir / 'images' / 'train'
        if not train_dir.exists():
            return
        
        files = sorted(os.listdir(train_dir))
        
        # 按模板分组
        templates = {}
        for f in files:
            if not f.endswith('.jpg'):
                continue
            parts = f.split('_')
            if len(parts) < 2:
                continue
            template_id = parts[0]  # 01, 04, etc.
            if template_id not in templates:
                templates[template_id] = []
            templates[template_id].append(f)
        
        # 对每个模板，取第一张作为参考
        for tid, files in templates.items():
            if files:
                # 使用第一张作为伪模板
                self.templates[tid] = train_dir / files[0]
        
        print(f"加载了 {len(self.templates)} 个模板")
        for tid, path in self.templates.items():
            print(f"  模板 {tid}: {path.name}")
    
    def detect_defects(self, img_path, template_path=None):
        """通过与模板对比检测缺陷"""
        img = cv2.imread(str(img_path))
        if img is None:
            return []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 如果有模板，进行差分
        if template_path and Path(template_path).exists():
            template = cv2.imread(str(template_path))
            if template is not None and template.shape == img.shape:
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                
                # 差分
                diff = cv2.absdiff(gray, template_gray)
                _, binary = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            else:
                # 尺寸不匹配，使用边缘检测
                edges = cv2.Canny(gray, 30, 100)
                binary = edges
        else:
            # 使用边缘检测作为备选
            edges = cv2.Canny(gray, 30, 100)
            binary = edges
        
        # 形态学处理
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < h*w*0.3:  # 过滤太小的噪点和太大的区域
                x, y, cw, ch = cv2.boundingRect(cnt)
                # 扩大一点边界
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                cw = min(cw + padding*2, w - x)
                ch = min(ch + padding*2, h - y)
                regions.append((x, y, cw, ch))
        
        return regions
    
    def parse_defect_type(self, filename):
        """从文件名解析缺陷类型"""
        name = Path(filename).stem.lower()
        
        if 'spurious_copper' in name or 'spurious' in name:
            return 5
        elif 'missing_hole' in name:
            return 0
        elif 'mouse_bite' in name:
            return 1
        elif 'open_circuit' in name:
            return 2
        elif 'short' in name:
            return 3
        elif 'spur' in name:
            return 4
        return 0
    
    def label_single(self, img_path):
        """标注单张图片"""
        # 获取模板ID
        filename = Path(img_path).stem
        template_id = filename.split('_')[0]
        
        # 获取对应的模板
        template_path = self.templates.get(template_id)
        
        # 检测缺陷区域
        regions = self.detect_defects(img_path, template_path)
        
        # 获取缺陷类型
        class_id = self.parse_defect_type(img_path)
        
        img = cv2.imread(str(img_path))
        if img is None:
            return []
        h, w = img.shape[:2]
        
        # 如果没有检测到区域，生成默认标注
        if not regions:
            # 基于位置的默认标注
            idx = int(filename.split('_')[-1]) if filename.split('_')[-1].isdigit() else 1
            row = (idx - 1) // 4
            col = (idx - 1) % 4
            
            default_w = int(w * 0.1)
            default_h = int(h * 0.1)
            cx = w * (0.2 + col * 0.2)
            cy = h * (0.2 + row * 0.2)
            
            regions = [(int(cx - default_w/2), int(cy - default_h/2), default_w, default_h)]
        
        # 转换为YOLO格式
        yolo_labels = []
        for x, y, cw, ch in regions[:2]:  # 最多2个
            cx_norm = (x + cw/2) / w
            cy_norm = (y + ch/2) / h
            w_norm = cw / w
            h_norm = ch / h
            
            # 限制范围
            cx_norm = max(0.01, min(0.99, cx_norm))
            cy_norm = max(0.01, min(0.99, cy_norm))
            w_norm = max(0.01, min(0.99, w_norm))
            h_norm = max(0.01, min(0.99, h_norm))
            
            yolo_labels.append((class_id, cx_norm, cy_norm, w_norm, h_norm))
        
        return yolo_labels
    
    def batch_process(self, split='train'):
        """批量处理"""
        img_dir = self.dataset_dir / 'images' / split
        label_dir = self.dataset_dir / 'labels' / split
        
        if not img_dir.exists():
            print(f"目录不存在: {img_dir}")
            return
        
        os.makedirs(label_dir, exist_ok=True)
        
        image_files = sorted(list(img_dir.glob('*.jpg')))
        
        stats = {'success': 0, 'failed': 0}
        
        for i, img_path in enumerate(image_files):
            try:
                yolo_labels = self.label_single(img_path)
                
                label_file = label_dir / f"{img_path.stem}.txt"
                with open(label_file, 'w') as f:
                    for label in yolo_labels:
                        f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
                stats['success'] += 1
            except Exception as e:
                stats['failed'] += 1
                print(f"错误: {img_path.name} - {e}")
            
            if (i + 1) % 100 == 0:
                print(f"进度: {i+1}/{len(image_files)}")
        
        print(f"\n完成! 成功: {stats['success']}, 失败: {stats['failed']}")
        return stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='./yolo_pcb_dataset')
    parser.add_argument('--split', default='train')
    args = parser.parse_args()
    
    labeler = PCBTemplatedLabeler(args.dataset)
    labeler.batch_process(args.split)
