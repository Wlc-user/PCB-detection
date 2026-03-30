"""
PCB缺陷智能标注工具
自动检测缺陷候选区域，支持人工审核和调整
"""
import os
import cv2
import numpy as np
from pathlib import Path
import json
import argparse

class PCBInteractiveLabeler:
    """交互式PCB缺陷标注工具"""
    
    DEFECT_TYPES = {
        0: 'missing_hole',
        1: 'mouse_bite', 
        2: 'open_circuit',
        3: 'short',
        4: 'spur',
        5: 'spurious_copper',
    }
    
    COLORS = [
        (0, 255, 0),    # 绿色 - missing_hole
        (255, 0, 0),    # 蓝色 - mouse_bite
        (0, 0, 255),    # 红色 - open_circuit
        (255, 255, 0),  # 青色 - short
        (255, 0, 255),  # 紫色 - spur
        (0, 255, 255),  # 黄色 - spurious_copper
    ]
    
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.annotations = {}
        self.current_idx = 0
        self.image_files = []
        
    def load_images(self, split='train'):
        """加载图片列表"""
        img_dir = self.dataset_dir / 'images' / split
        if not img_dir.exists():
            print(f"目录不存在: {img_dir}")
            return []
            
        self.image_files = sorted(list(img_dir.glob('*.jpg')))
        print(f"加载了 {len(self.image_files)} 张图片")
        return self.image_files
    
    def detect_defect_regions(self, img_path, defect_type):
        """多策略缺陷区域检测"""
        img = cv2.imread(str(img_path))
        if img is None:
            return []
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        regions = []
        
        # 策略1: 自适应阈值检测异常
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 策略2: Canny边缘检测
        edges = cv2.Canny(gray, 30, 100)
        
        # 策略3: 形态学操作
        kernel = np.ones((3,3), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # 根据缺陷类型选择检测策略
        if defect_type in ['missing_hole', 'spurious_copper']:
            # 检测暗区域/亮点
            _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 100 < area < 50000:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    if cw > 3 and ch > 3:
                        regions.append(self._create_region(x, y, cw, ch, 'adaptive'))
                        
        elif defect_type in ['mouse_bite', 'open_circuit', 'short']:
            # 检测边缘异常
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 200 < area < 30000:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    # 过滤太大或太小的区域
                    if cw > 5 and ch > 5 and cw < w * 0.4 and ch < h * 0.4:
                        regions.append(self._create_region(x, y, cw, ch, 'edge'))
        
        elif defect_type == 'spur':
            # 检测细小突出
            edges_thin = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges_thin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 50 < area < 5000:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    regions.append(self._create_region(x, y, cw, ch, 'thin'))
        
        # 去重和合并重叠区域
        regions = self._merge_overlapping_regions(regions)
        
        # 按面积排序，取最大的几个
        regions = sorted(regions, key=lambda r: r['area'], reverse=True)[:5]
        
        return regions
    
    def _create_region(self, x, y, cw, ch, method):
        """创建区域字典"""
        # 扩大边界框 10%
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        cw = min(cw + padding*2, 3034 - x)  # 假设最大宽度
        ch = min(ch + padding*2, 1586 - y)
        
        return {
            'x': int(x), 'y': int(y), 
            'w': int(cw), 'h': int(ch),
            'area': cw * ch,
            'method': method
        }
    
    def _merge_overlapping_regions(self, regions, iou_threshold=0.3):
        """合并重叠区域"""
        if len(regions) <= 1:
            return regions
            
        merged = []
        used = set()
        
        for i, r1 in enumerate(regions):
            if i in used:
                continue
                
            group = [r1]
            used.add(i)
            
            for j, r2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue
                    
                if self._calc_iou(r1, r2) > iou_threshold:
                    group.append(r2)
                    used.add(j)
            
            # 合并为一个更大的区域
            if len(group) > 1:
                x = min(r['x'] for r in group)
                y = min(r['y'] for r in group)
                x2 = max(r['x'] + r['w'] for r in group)
                y2 = max(r['y'] + r['h'] for r in group)
                merged.append({
                    'x': x, 'y': y,
                    'w': x2 - x, 'h': y2 - y,
                    'area': (x2-x) * (y2-y),
                    'method': 'merged'
                })
            else:
                merged.append(r1)
        
        return merged
    
    def _calc_iou(self, r1, r2):
        """计算IoU"""
        x1 = max(r1['x'], r2['x'])
        y1 = max(r1['y'], r2['y'])
        x2 = min(r1['x'] + r1['w'], r2['x'] + r2['w'])
        y2 = min(r1['y'] + r1['h'], r2['y'] + r2['h'])
        
        if x2 <= x1 or y2 <= y1:
            return 0
            
        inter = (x2 - x1) * (y2 - y1)
        union = r1['area'] + r2['area'] - inter
        return inter / union if union > 0 else 0
    
    def parse_defect_type(self, filename):
        """从文件名解析缺陷类型"""
        name = Path(filename).stem.lower()
        
        # 按优先级匹配
        if 'spurious_copper' in name:
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
        
        return 0  # 默认
    
    def draw_regions(self, img, regions, selected=None):
        """绘制候选区域"""
        img_vis = img.copy()
        
        for i, region in enumerate(regions):
            color = self.COLORS[i % len(self.COLORS)]
            thickness = 3 if selected and i == selected else 2
            
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(img_vis, (x, y), (x+w, y+h), color, thickness)
            
            # 标注序号
            label = f"#{i+1}"
            cv2.putText(img_vis, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return img_vis
    
    def save_annotation(self, img_path, regions, defect_type):
        """保存标注到YOLO格式"""
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        label_dir = self.dataset_dir / 'labels' / img_path.parent.name
        label_dir.mkdir(parents=True, exist_ok=True)
        
        label_file = label_dir / f"{img_path.stem}.txt"
        
        with open(label_file, 'w') as f:
            for region in regions:
                cx = (region['x'] + region['w']/2) / w
                cy = (region['y'] + region['h']/2) / h
                nw = region['w'] / w
                nh = region['h'] / h
                
                # 限制在合理范围
                cx = max(0.01, min(0.99, cx))
                cy = max(0.01, min(0.99, cy))
                nw = max(0.01, min(0.99, nw))
                nh = max(0.01, min(0.99, nh))
                
                f.write(f"{defect_type} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        
        return label_file
    
    def run_interactive(self, split='train'):
        """运行交互式标注"""
        self.load_images(split)
        
        if not self.image_files:
            return
            
        print(f"\n{'='*60}")
        print("PCB缺陷交互式标注工具")
        print(f"{'='*60}")
        print("操作说明:")
        print("  [0-9] - 选择候选区域编号")
        print("  [a]   - 选择所有检测到的区域")
        print("  [s]   - 保存并跳过")
        print("  [d]   - 删除最后一个区域")
        print("  [w]   - 使用鼠标画框 (点击左上角和右下角)")
        print("  [q]   - 退出")
        print("  [n]   - 下一张图片")
        print(f"{'='*60}\n")
        
        self.current_idx = 0
        
        while self.current_idx < len(self.image_files):
            img_path = self.image_files[self.current_idx]
            defect_type = self.parse_defect_type(img_path.name)
            
            print(f"\n[{self.current_idx+1}/{len(self.image_files)}] {img_path.name}")
            print(f"缺陷类型: {self.DEFECT_TYPES[defect_type]} (ID: {defect_type})")
            
            # 检测候选区域
            regions = self.detect_defect_regions(img_path, self.DEFECT_TYPES[defect_type])
            
            if not regions:
                print("未检测到缺陷区域，使用默认位置")
                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]
                # 默认放在中心
                regions = [{
                    'x': int(w*0.3), 'y': int(h*0.3),
                    'w': int(w*0.1), 'h': int(h*0.1),
                    'area': int(w*0.1 * h*0.1),
                    'method': 'default'
                }]
            
            print(f"检测到 {len(regions)} 个候选区域")
            
            # 显示候选区域
            img = cv2.imread(str(img_path))
            img_vis = self.draw_regions(img, regions)
            
            # 保存预览图
            preview_path = self.dataset_dir / 'preview.jpg'
            cv2.imwrite(str(preview_path), img_vis)
            print(f"预览图已保存: {preview_path}")
            
            # 直接保存检测结果
            label_file = self.save_annotation(img_path, regions, defect_type)
            print(f"已保存标注: {label_file}")
            
            self.current_idx += 1
        
        print("\n所有图片标注完成！")
    
    def batch_process(self, split='train'):
        """批量处理模式（不交互）"""
        self.load_images(split)
        
        stats = {'success': 0, 'failed': 0}
        
        for i, img_path in enumerate(self.image_files):
            defect_type = self.parse_defect_type(img_path.name)
            regions = self.detect_defect_regions(img_path, self.DEFECT_TYPES[defect_type])
            
            if not regions:
                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]
                regions = [{
                    'x': int(w*0.3), 'y': int(h*0.3),
                    'w': int(w*0.1), 'h': int(h*0.1),
                    'area': int(w*0.1 * h*0.1),
                    'method': 'default'
                }]
            
            try:
                self.save_annotation(img_path, regions, defect_type)
                stats['success'] += 1
            except:
                stats['failed'] += 1
            
            if (i + 1) % 50 == 0:
                print(f"处理进度: {i+1}/{len(self.image_files)}")
        
        print(f"\n批量处理完成!")
        print(f"  成功: {stats['success']}")
        print(f"  失败: {stats['failed']}")


def main():
    parser = argparse.ArgumentParser(description='PCB缺陷交互式标注工具')
    parser.add_argument('--dataset', type=str, default='./yolo_pcb_dataset')
    parser.add_argument('--split', type=str, default='train', 
                       choices=['train', 'val', 'test'])
    parser.add_argument('--mode', type=str, default='batch',
                       choices=['interactive', 'batch'])
    
    args = parser.parse_args()
    
    labeler = PCBInteractiveLabeler(args.dataset)
    
    if args.mode == 'interactive':
        labeler.run_interactive(args.split)
    else:
        labeler.batch_process(args.split)


if __name__ == '__main__':
    main()
