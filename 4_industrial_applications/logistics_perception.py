"""
无人物流感知系统 - Logistics Perception System
================================================
功能: 目标分割、分类、检测、追踪
场景: 仓库环境、无人车环境

Author: Vision Team
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SceneType(Enum):
    """场景类型"""
    WAREHOUSE = "warehouse"        # 仓库环境
    UNMANNED_VEHICLE = "vehicle"  # 无人车环境
    MIXED = "mixed"               # 混合场景


class ObjectCategory(Enum):
    """物体类别"""
    # 仓库场景
    PERSON = "person"              # 人员
    FORKLIFT = "forklift"          # 叉车
    PALLET = "pallet"              # 托盘
    BOX = "box"                    # 货物箱
    SHELF = "shelf"                # 货架
    CONVEYOR = "conveyor"          # 传送带
    ROBOT = "robot"               # AGV/AMR机器人
    
    # 无人车场景
    VEHICLE = "vehicle"            # 车辆
    OBSTACLE = "obstacle"          # 障碍物
    TRAFFIC_SIGN = "traffic_sign"  # 交通标识
    LANE = "lane"                 # 车道线
    DANGER_ZONE = "danger_zone"    # 危险区域
    
    # 通用
    UNKNOWN = "unknown"


@dataclass
class DetectedObject:
    """检测到的物体"""
    track_id: int = -1
    category: ObjectCategory = ObjectCategory.UNKNOWN
    class_name: str = "unknown"
    confidence: float = 0.0
    
    # 边界框
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2
    
    # 分割掩码
    mask: Optional[np.ndarray] = None
    
    # 追踪信息
    position: Tuple[float, float, float] = (0, 0, 0)  # x, y, z (世界坐标)
    velocity: Tuple[float, float] = (0, 0)  # vx, vy
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    
    # 时间戳
    timestamp: float = 0.0
    age: int = 0  # 连续帧数
    visible: bool = True
    
    # 特征向量
    features: Optional[np.ndarray] = None
    
    @property
    def bbox_area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, (x2 - x1) * (y2 - y1))
    
    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def mask_area(self) -> float:
        if self.mask is not None:
            return np.sum(self.mask > 0)
        return 0.0


@dataclass
class SceneUnderstanding:
    """场景理解结果"""
    scene_type: SceneType = SceneType.MIXED
    
    # 检测到的物体列表
    objects: List[DetectedObject] = field(default_factory=list)
    
    # 空间理解
    occupancy_grid: Optional[np.ndarray] = None  # 占用栅格图
    traversable_area: Optional[np.ndarray] = None  # 可通行区域
    danger_zones: List[Tuple[int, int, int, int]] = field(default_factory=list)
    
    # 语义分割
    semantic_mask: Optional[np.ndarray] = None
    
    # 统计信息
    stats: Dict[str, Any] = field(default_factory=dict)
    
    # 时间戳
    timestamp: float = 0.0
    frame_id: int = 0
    
    def get_objects_by_category(self, category: ObjectCategory) -> List[DetectedObject]:
        """按类别获取物体"""
        return [obj for obj in self.objects if obj.category == category]
    
    def get_persons(self) -> List[DetectedObject]:
        """获取所有人员"""
        return self.get_objects_by_category(ObjectCategory.PERSON)
    
    def get_obstacles(self) -> List[DetectedObject]:
        """获取所有障碍物"""
        return self.get_objects_by_category(ObjectCategory.OBSTACLE)
    
    def get_vehicles(self) -> List[DetectedObject]:
        """获取所有车辆"""
        return self.get_objects_by_category(ObjectCategory.FORKLIFT) + \
               self.get_objects_by_category(ObjectCategory.VEHICLE) + \
               self.get_objects_by_category(ObjectCategory.ROBOT)


class LogisticsPerception:
    """
    无人物流感知系统主类
    
    整合: 分割 + 检测 + 分类 + 追踪
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # 模型加载
        self._detector = None      # YOLO检测器
        self._segmenter = None     # SAM分割器
        self._tracker = None       # ByteTrack追踪器
        
        # 状态
        self.frame_count = 0
        self.track_ids = set()
        self.next_track_id = 0
        
        # 类别映射
        self._class_mapping = self._build_class_mapping()
        
        logger.info("[OK] LogisticsPerception initialized")
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            # 检测配置
            "detector": {
                "model": "yolov8n.pt",
                "conf_threshold": 0.4,
                "iou_threshold": 0.5,
                "device": "cuda" if self._check_cuda() else "cpu",
            },
            # 分割配置
            "segmenter": {
                "enabled": True,
                "model_type": "sam",  # sam, sam2, samvit
            },
            # 追踪配置
            "tracker": {
                "enabled": True,
                "max_time_lost": 30,
                "track_thresh": 0.5,
                "match_thresh": 0.8,
            },
            # 场景配置
            "scene": {
                "type": "mixed",
                "classes": list(ObjectCategory),
            },
        }
    
    def _check_cuda(self) -> bool:
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _build_class_mapping(self) -> Dict[int, ObjectCategory]:
        """构建COCO类别到物流类别的映射"""
        # COCO 80类 -> 物流类别
        mapping = {
            0: ObjectCategory.PERSON,      # person
            1: ObjectCategory.UNKNOWN,     # bicycle
            2: ObjectCategory.VEHICLE,      # car
            3: ObjectCategory.VEHICLE,     # motorcycle
            4: ObjectCategory.VEHICLE,      # airplane
            5: ObjectCategory.VEHICLE,      # bus
            6: ObjectCategory.VEHICLE,      # train
            7: ObjectCategory.VEHICLE,      # truck
            8: ObjectCategory.BOX,          # boat
            9: ObjectCategory.UNKNOWN,      # traffic light
            10: ObjectCategory.UNKNOWN,     # fire hydrant
            11: ObjectCategory.UNKNOWN,     # stop sign
            12: ObjectCategory.UNKNOWN,      # parking meter
            13: ObjectCategory.UNKNOWN,     # bench
            14: ObjectCategory.UNKNOWN,     # bird
            15: ObjectCategory.UNKNOWN,     # cat
            16: ObjectCategory.UNKNOWN,     # dog
            17: ObjectCategory.UNKNOWN,     # horse
            18: ObjectCategory.UNKNOWN,     # sheep
            19: ObjectCategory.UNKNOWN,     # cow
            20: ObjectCategory.UNKNOWN,     # elephant
            21: ObjectCategory.UNKNOWN,     # bear
            22: ObjectCategory.UNKNOWN,     # zebra
            23: ObjectCategory.UNKNOWN,     # giraffe
            24: ObjectCategory.UNKNOWN,     # backpack
            25: ObjectCategory.UNKNOWN,     # umbrella
            26: ObjectCategory.UNKNOWN,     # handbag
            27: ObjectCategory.UNKNOWN,     # tie
            28: ObjectCategory.UNKNOWN,     # suitcase
            29: ObjectCategory.BOX,         # frisbee
            30: ObjectCategory.UNKNOWN,     # skis
            31: ObjectCategory.UNKNOWN,     # snowboard
            32: ObjectCategory.UNKNOWN,     # sports ball
            33: ObjectCategory.UNKNOWN,     # kite
            34: ObjectCategory.UNKNOWN,     # baseball bat
            35: ObjectCategory.UNKNOWN,     # baseball glove
            36: ObjectCategory.UNKNOWN,     # skateboard
            37: ObjectCategory.UNKNOWN,     # surfboard
            38: ObjectCategory.UNKNOWN,     # tennis racket
            39: ObjectCategory.BOX,         # bottle
            40: ObjectCategory.BOX,         # wine glass
            41: ObjectCategory.BOX,         # cup
            42: ObjectCategory.BOX,         # fork
            43: ObjectCategory.BOX,         # knife
            44: ObjectCategory.BOX,         # spoon
            45: ObjectCategory.BOX,         # bowl
            46: ObjectCategory.BOX,         # banana
            47: ObjectCategory.BOX,         # apple
            48: ObjectCategory.BOX,         # sandwich
            49: ObjectCategory.BOX,         # orange
            50: ObjectCategory.BOX,         # broccoli
            51: ObjectCategory.BOX,         # carrot
            52: ObjectCategory.BOX,         # hot dog
            53: ObjectCategory.BOX,         # pizza
            54: ObjectCategory.BOX,         # donut
            55: ObjectCategory.BOX,         # cake
            56: ObjectCategory.UNKNOWN,     # chair
            57: ObjectCategory.UNKNOWN,      # couch
            58: ObjectCategory.UNKNOWN,      # potted plant
            59: ObjectCategory.UNKNOWN,      # bed
            60: ObjectCategory.UNKNOWN,      # dining table
            61: ObjectCategory.UNKNOWN,      # toilet
            62: ObjectCategory.UNKNOWN,      # tv
            63: ObjectCategory.UNKNOWN,      # laptop
            64: ObjectCategory.UNKNOWN,      # mouse
            65: ObjectCategory.UNKNOWN,      # remote
            66: ObjectCategory.UNKNOWN,      # keyboard
            67: ObjectCategory.UNKNOWN,      # cell phone
            68: ObjectCategory.UNKNOWN,      # microwave
            69: ObjectCategory.UNKNOWN,      # oven
            70: ObjectCategory.UNKNOWN,      # toaster
            71: ObjectCategory.UNKNOWN,      # sink
            72: ObjectCategory.UNKNOWN,      # refrigerator
            73: ObjectCategory.UNKNOWN,      # book
            74: ObjectCategory.UNKNOWN,      # clock
            75: ObjectCategory.UNKNOWN,      # vase
            76: ObjectCategory.UNKNOWN,      # scissors
            77: ObjectCategory.UNKNOWN,      # teddy bear
            78: ObjectCategory.UNKNOWN,      # hair drier
            79: ObjectCategory.UNKNOWN,      # toothbrush
        }
        return mapping
    
    def load_models(self):
        """加载所有模型"""
        self._load_detector()
        self._load_segmenter()
        self._load_tracker()
        logger.info("[OK] All models loaded")
    
    def _load_detector(self):
        """加载YOLO检测器"""
        try:
            from ultralytics import YOLO
            model_path = self.config["detector"].get("model", "yolov8n.pt")
            self._detector = YOLO(model_path)
            logger.info(f"[OK] Detector loaded: {model_path}")
        except Exception as e:
            logger.warning(f"[!] Detector load failed: {e}")
            self._detector = None
    
    def _load_segmenter(self):
        """加载分割器"""
        # 可集成 SAM
        # try:
        #     from segment_anything import sam_model_registry
        #     ...
        # except:
        pass
    
    def _load_tracker(self):
        """加载追踪器"""
        try:
            import supervision as sv
            self._tracker = sv.ByteTrack
            logger.info("[OK] Tracker loaded: ByteTrack")
        except Exception as e:
            logger.warning(f"[!] Tracker load failed: {e}")
            self._tracker = None
    
    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """检测物体"""
        if self._detector is None:
            logger.warning("[!] Detector not loaded")
            return []
        
        results = self._detector(frame, verbose=False)[0]
        objects = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, classes)):
                obj = DetectedObject()
                obj.bbox = tuple(map(int, box))
                obj.confidence = float(conf)
                obj.category = self._class_mapping.get(cls_id, ObjectCategory.UNKNOWN)
                obj.class_name = results.names[cls_id]
                
                objects.append(obj)
        
        return objects
    
    def segment(self, frame: np.ndarray, bboxes: List[Tuple]) -> List[np.ndarray]:
        """分割物体"""
        # TODO: 集成 SAM
        return [None] * len(bboxes)
    
    def track(self, frame: np.ndarray, objects: List[DetectedObject]) -> List[DetectedObject]:
        """追踪物体"""
        if self._tracker is None:
            # 简单ID分配
            for obj in objects:
                if obj.track_id < 0:
                    obj.track_id = self.next_track_id
                    self.next_track_id += 1
            return objects
        
        # 使用 ByteTrack
        try:
            import supervision as sv
            from ultralytics import YOLO
            
            # 重新检测
            results = self._detector(frame, verbose=False)[0]
            
            # 转换为 supervision 格式
            detections = sv.Detections.from_ultralytics(results)
            
            # 追踪
            tracker = sv.ByteTrack()
            tracked = tracker.update_with_detections(detections)
            
            # 更新对象
            for i, obj in enumerate(objects):
                if i < len(tracked):
                    obj.track_id = tracked[i].tracker_id
                    obj.class_name = tracked[i].class_name
            
            return objects
        except Exception as e:
            logger.warning(f"[!] Tracking failed: {e}")
            return objects
    
    def understand_scene(self, frame: np.ndarray) -> SceneUnderstanding:
        """
        完整场景理解
        分割 -> 检测 -> 分类 -> 追踪 -> 场景分析
        """
        self.frame_count += 1
        timestamp = time.time()
        
        result = SceneUnderstanding(
            timestamp=timestamp,
            frame_id=self.frame_count
        )
        
        # 1. 检测
        objects = self.detect(frame)
        
        # 2. 追踪
        objects = self.track(frame, objects)
        
        # 3. 分割 (可选)
        if self.config["segmenter"]["enabled"]:
            bboxes = [obj.bbox for obj in objects]
            masks = self.segment(frame, bboxes)
            for obj, mask in zip(objects, masks):
                obj.mask = mask
        
        result.objects = objects
        
        # 4. 场景分析
        result.stats = self._analyze_scene(objects)
        
        # 5. 场景类型判断
        result.scene_type = self._infer_scene_type(objects)
        
        return result
    
    def _analyze_scene(self, objects: List[DetectedObject]) -> Dict[str, Any]:
        """分析场景统计"""
        stats = {
            "total_objects": len(objects),
            "by_category": {},
            "person_count": 0,
            "vehicle_count": 0,
            "obstacle_count": 0,
            "avg_confidence": 0,
        }
        
        if not objects:
            return stats
        
        confs = []
        for obj in objects:
            cat = obj.category.value
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
            confs.append(obj.confidence)
            
            if obj.category == ObjectCategory.PERSON:
                stats["person_count"] += 1
            elif obj.category in [ObjectCategory.VEHICLE, ObjectCategory.FORKLIFT, ObjectCategory.ROBOT]:
                stats["vehicle_count"] += 1
            elif obj.category == ObjectCategory.OBSTACLE:
                stats["obstacle_count"] += 1
        
        stats["avg_confidence"] = sum(confs) / len(confs) if confs else 0
        
        return stats
    
    def _infer_scene_type(self, objects: List[DetectedObject]) -> SceneType:
        """推断场景类型"""
        categories = [obj.category for obj in objects]
        
        if ObjectCategory.FORKLIFT in categories or ObjectCategory.SHELF in categories:
            return SceneType.WAREHOUSE
        
        if ObjectCategory.LANE in categories or ObjectCategory.TRAFFIC_SIGN in categories:
            return SceneType.UNMANNED_VEHICLE
        
        return SceneType.MIXED
    
    def visualize(self, frame: np.ndarray, result: SceneUnderstanding) -> np.ndarray:
        """可视化结果"""
        vis = frame.copy()
        
        # 颜色映射
        colors = {
            ObjectCategory.PERSON: (255, 0, 0),       # 蓝色
            ObjectCategory.VEHICLE: (0, 255, 0),       # 绿色
            ObjectCategory.FORKLIFT: (0, 255, 255),    # 黄色
            ObjectCategory.ROBOT: (0, 128, 255),       # 橙色
            ObjectCategory.BOX: (128, 0, 255),        # 紫色
            ObjectCategory.OBSTACLE: (0, 0, 255),      # 红色
            ObjectCategory.UNKNOWN: (128, 128, 128),   # 灰色
        }
        
        for obj in result.objects:
            x1, y1, x2, y2 = obj.bbox
            color = colors.get(obj.category, (255, 255, 255))
            
            # 绘制边界框
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{obj.class_name} ID:{obj.track_id} {obj.confidence:.2f}"
            cv2.putText(vis, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 绘制分割掩码
            if obj.mask is not None:
                mask_color = (*color, 128)
                vis = self._draw_mask(vis, obj.mask, mask_color)
            
            # 绘制轨迹
            if len(obj.trajectory) > 1:
                for i in range(len(obj.trajectory) - 1):
                    cv2.line(vis, obj.trajectory[i], obj.trajectory[i+1], color, 2)
        
        # 绘制统计信息
        stats_text = [
            f"Frame: {result.frame_id}",
            f"Objects: {result.stats.get('total_objects', 0)}",
            f"Persons: {result.stats.get('person_count', 0)}",
            f"Vehicles: {result.stats.get('vehicle_count', 0)}",
            f"Scene: {result.scene_type.value}",
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(vis, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis
    
    def _draw_mask(self, frame: np.ndarray, mask: np.ndarray, color: Tuple) -> np.ndarray:
        """绘制掩码"""
        if mask is None:
            return frame
        
        # 创建彩色掩码
        mask_vis = np.zeros_like(frame)
        mask_vis[mask > 0] = color[:3]
        
        # 混合
        alpha = 0.4
        frame = cv2.addWeighted(frame, 1 - alpha, mask_vis, alpha, 0)
        
        return frame


def demo():
    """演示"""
    print("=" * 60)
    print("   无人物流感知系统 - Logistics Perception Demo")
    print("=" * 60)
    
    # 初始化
    perception = LogisticsPerception()
    
    # 尝试加载模型
    try:
        perception.load_models()
    except Exception as e:
        print(f"[!] Model load failed: {e}")
        print("[*] Using mock mode")
    
    # 使用摄像头或视频
    print("\n[*] Starting camera...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Camera not available")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 场景理解
        result = perception.understand_scene(frame)
        
        # 可视化
        vis = perception.visualize(frame, result)
        
        cv2.imshow("Logistics Perception", vis)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo()
