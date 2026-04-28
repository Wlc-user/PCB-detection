"""
PCB缺陷检测器 - 带ROI区域约束
功能:
1. 支持PCB区域ROI约束，抑制背景误检
2. 使用语义先验提高检测精度
3. 集成置信度阈值优化
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict
import yaml
import argparse
from cv_features import CVFeatures


class ROI:
    """ROI区域定义"""
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def contains(self, x: int, y: int) -> bool:
        """检查点是否在ROI内"""
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h
    
    def get_rect(self) -> Tuple[int, int, int, int]:
        """获取矩形区域 (x, y, w, h)"""
        return (self.x, self.y, self.w, self.h)


class DetectorWithROI:
    """带ROI约束的PCB缺陷检测器"""
    
    # PCB缺陷类别
    CLASS_NAMES = {
        0: 'missing_hole',
        1: 'mouse_bite', 
        2: 'open_circuit',
        3: 'short',
        4: 'spur',
        5: 'spurious_copper',
        6: 'normal'
    }
    
    def __init__(
        self, 
        model_path: str = "models/yolov8/train/weights/best.pt",
        roi_config_path: Optional[str] = None,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.50,
        min_defect_area: int = 50,
        max_defect_area: int = 50000,
    ):
        """
        初始化检测器
        
        Args:
            model_path: 模型路径
            roi_config_path: ROI配置文件路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
            min_defect_area: 最小缺陷面积(像素)
            max_defect_area: 最大缺陷面积(像素)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_defect_area = min_defect_area
        self.max_defect_area = max_defect_area
        
        # 加载模型
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        
        # 加载ROI配置
        self.roi_regions: List[ROI] = []
        if roi_config_path and Path(roi_config_path).exists():
            self.load_roi_config(roi_config_path)
        
        # 初始化特征提取器
        self.features_extractor = CVFeatures()
        
        # 统计信息
        self.stats = {
            'total_detections': 0,
            'roi_filtered': 0,
            'area_filtered': 0,
            'defects_found': 0,
        }
    
    def load_roi_config(self, config_path: str):
        """加载ROI配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'roi_regions' in config:
            for roi_data in config['roi_regions']:
                roi = ROI(
                    x=roi_data['x'],
                    y=roi_data['y'],
                    w=roi_data['w'],
                    h=roi_data['h']
                )
                self.roi_regions.append(roi)
                print(f"Loaded ROI: ({roi.x}, {roi.y}, {roi.w}, {roi.h})")
    
    def add_roi(self, x: int, y: int, w: int, h: int):
        """添加ROI区域"""
        self.roi_regions.append(ROI(x, y, w, h))
    
    def clear_roi(self):
        """清除所有ROI区域"""
        self.roi_regions.clear()
    
    def _filter_by_roi(self, boxes: np.ndarray) -> np.ndarray:
        """根据ROI过滤检测框"""
        if not self.roi_regions:
            return boxes
        
        filtered_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            in_roi = False
            for roi in self.roi_regions:
                if roi.contains(int(cx), int(cy)):
                    in_roi = True
                    break
            
            if in_roi:
                filtered_boxes.append(box)
            else:
                self.stats['roi_filtered'] += 1
        
        return np.array(filtered_boxes) if filtered_boxes else np.array([]).reshape(0, 6)
    
    def _filter_by_area(self, boxes: np.ndarray) -> np.ndarray:
        """根据面积过滤检测框"""
        if len(boxes) == 0:
            return boxes
        
        filtered_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            area = (x2 - x1) * (y2 - y1)
            
            if self.min_defect_area <= area <= self.max_defect_area:
                filtered_boxes.append(box)
            else:
                self.stats['area_filtered'] += 1
        
        return np.array(filtered_boxes) if filtered_boxes else np.array([]).reshape(0, 6)
    
    def detect(
        self, 
        image: np.ndarray,
        use_roi: bool = True,
        use_area_filter: bool = True,
    ) -> Dict:
        """
        检测图像中的缺陷
        
        Args:
            image: 输入图像 (BGR格式)
            use_roi: 是否使用ROI过滤
            use_area_filter: 是否使用面积过滤
        
        Returns:
            检测结果字典
        """
        # 推理
        results = self.model(
            image, 
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        result = results[0]
        
        # 提取检测框
        if result.boxes is None or len(result.boxes) == 0:
            return {
                'boxes': [],
                'scores': [],
                'classes': [],
                'labels': [],
                'defects': [],
                'defect_count': 0,
            }
        
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        # 合并为 [N, 6] 格式
        detections = np.column_stack([boxes, scores, classes])
        
        self.stats['total_detections'] += len(detections)
        
        # 应用过滤
        if use_roi:
            detections = self._filter_by_roi(detections)
        
        if use_area_filter:
            detections = self._filter_by_area(detections)
        
        # 分离缺陷和正常
        defects = []
        for det in detections:
            cls_id = int(det[5])
            if cls_id != 6:  # 非normal类别
                defects.append({
                    'bbox': det[:4].tolist(),
                    'confidence': float(det[4]),
                    'class_id': cls_id,
                    'class_name': self.CLASS_NAMES.get(cls_id, 'unknown'),
                })
                self.stats['defects_found'] += 1
        
        self.stats['defects_found'] = len(defects)
        
        return {
            'boxes': detections[:, :4].tolist() if len(detections) > 0 else [],
            'scores': detections[:, 4].tolist() if len(detections) > 0 else [],
            'classes': detections[:, 5].tolist() if len(detections) > 0 else [],
            'labels': [self.CLASS_NAMES.get(int(c), 'unknown') for c in detections[:, 5]] if len(detections) > 0 else [],
            'defects': defects,
            'defect_count': len(defects),
            'raw_result': result,
        }
    
    def detect_image_path(self, image_path: str, **kwargs) -> Dict:
        """检测图像文件"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self.detect(image, **kwargs)
    
    def draw_detections(
        self, 
        image: np.ndarray, 
        detections: Dict,
        show_roi: bool = True,
        thickness: int = 2,
    ) -> np.ndarray:
        """在图像上绘制检测结果"""
        result = image.copy()
        
        # 绘制ROI
        if show_roi:
            for roi in self.roi_regions:
                cv2.rectangle(
                    result, 
                    (roi.x, roi.y), 
                    (roi.x + roi.w, roi.y + roi.h),
                    (0, 255, 0), 1
                )
        
        # 绘制检测框
        colors = {
            0: (255, 0, 0),      # missing_hole - 蓝色
            1: (0, 255, 255),    # mouse_bite - 黄色
            2: (0, 165, 255),    # open_circuit - 橙色
            3: (0, 0, 255),      # short - 红色
            4: (255, 0, 255),   # spur - 紫色
            5: (0, 255, 0),      # spurious_copper - 绿色
            6: (128, 128, 128), # normal - 灰色
        }
        
        for defect in detections['defects']:
            x1, y1, x2, y2 = [int(v) for v in defect['bbox']]
            cls_id = defect['class_id']
            color = colors.get(cls_id, (255, 255, 255))
            
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{defect['class_name']}: {defect['confidence']:.2f}"
            cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 统计信息
        info_text = f"Defects: {detections['defect_count']}"
        cv2.putText(result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return result
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()
    
    def process_with_features(
        self, 
        image: np.ndarray, 
        features: List[str],
        **kwargs
    ) -> Dict:
        """
        处理图像并提取特征
        
        Args:
            image: 输入图像 (BGR格式)
            features: 要提取的特征列表
            **kwargs: 传递给detect方法的参数
        
        Returns:
            包含检测结果和特征的字典
        """
        # 先进行缺陷检测
        detection_result = self.detect(image, **kwargs)
        
        # 提取特征
        features_result = self.features_extractor.process_image(image, features)
        
        # 合并结果
        result = {
            'detection': detection_result,
            'features': features_result
        }
        
        return result


def create_demo_roi_config(output_path: str = "configs/roi_config.yaml"):
    """创建示例ROI配置文件"""
    config = {
        'roi_regions': [
            {'x': 50, 'y': 50, 'w': 1700, 'h': 1300},
        ],
        'min_defect_area': 50,
        'max_defect_area': 50000,
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    print(f"ROI配置已保存: {output_path}")
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='带ROI的PCB缺陷检测器')
    parser.add_argument('--model', type=str, default='models/yolov8/train/weights/best.pt', help='模型路径')
    parser.add_argument('--roi', type=str, default=None, help='ROI配置文件')
    parser.add_argument('--conf', type=float, default=0.30, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.50, help='NMS IoU阈值')
    parser.add_argument('--image', type=str, required=True, help='检测图像路径')
    parser.add_argument('--output', type=str, default='result.jpg', help='输出图像路径')
    parser.add_argument('--features', type=str, nargs='+', default=[], help='要提取的特征')
    args = parser.parse_args()
    
    # 初始化检测器
    detector = DetectorWithROI(
        model_path=args.model,
        roi_config_path=args.roi,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
    
    print(f"\n检测图像: {args.image}")
    
    if args.features:
        # 处理图像并提取特征
        result = detector.process_with_features(
            cv2.imread(args.image),
            features=args.features
        )
        
        # 打印检测结果
        detection_result = result['detection']
        print(f"\n检测到 {detection_result['defect_count']} 个缺陷:")
        for d in detection_result['defects']:
            print(f"  - {d['class_name']}: {d['confidence']:.2f}")
        
        # 处理并保存特征结果
        for feature_name, feature_result in result['features'].items():
            if isinstance(feature_result, np.ndarray):
                feature_output = f"{args.output.rsplit('.', 1)[0]}_{feature_name}.jpg"
                cv2.imwrite(feature_output, feature_result)
                print(f"\n特征 {feature_name} 已保存: {feature_output}")
    else:
        # 仅进行缺陷检测
        result = detector.detect_image_path(args.image)
        
        print(f"\n检测到 {result['defect_count']} 个缺陷:")
        for d in result['defects']:
            print(f"  - {d['class_name']}: {d['confidence']:.2f}")
        
        # 绘制并保存结果
        if 'raw_result' in result:
            # 使用模型自带绘图
            annotated = result['raw_result'].plot()
            cv2.imwrite(args.output, annotated)
            print(f"\n结果已保存: {args.output}")
    
    # 统计信息
    stats = detector.get_stats()
    print(f"\n统计: {stats}")
