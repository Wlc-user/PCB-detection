"""
无人物流 - 统一感知+SLAM系统
融合视觉感知和SLAM，提供完整的环境理解
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

from logistics_perception import LogisticsPerception, ObjectCategory, SceneType, DetectedObject, SceneUnderstanding
from slam_module import (
    SLAMType, BaseSLAM, VisualSLAM, LidarSLAM, FusionSLAM,
    Pose2D, OccupancyGrid, SLAMState, create_slam
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionLevel(Enum):
    """融合级别"""
    LOOSE = "loose"      # 松耦合: 分别运行，结果组合
    TIGHT = "tight"      # 紧耦合: SLAM特征参与感知
    DEEP = "deep"        # 深度融合: 语义地图


@dataclass
class Obstacle:
    """障碍物"""
    position: Tuple[float, float]  # 世界坐标 (x, y)
    size: Tuple[float, float]     # 尺寸 (width, height)
    category: str                  # 类别
    confidence: float = 0.0
    source: str = "perception"     # 'perception' or 'slam'
    id: int = -1                   # 追踪ID


@dataclass
class LocalMap:
    """局部地图"""
    pose: Pose2D = field(default_factory=Pose2D)
    obstacles: List[Obstacle] = field(default_factory=list)
    traversable: List[Tuple[float, float]] = field(default_factory=list)  # 可通行区域
    dynamic_objects: List[Obstacle] = field(default_factory=list)  # 动态物体
    static_map: Optional[OccupancyGrid] = None
    timestamp: float = 0.0


@dataclass
class AGVPerceptionResult:
    """AGV完整感知结果"""
    # 感知信息
    objects: List[DetectedObject] = field(default_factory=list)
    scene_type: SceneType = SceneType.MIXED
    
    # SLAM信息
    pose: Pose2D = field(default_factory=Pose2D)
    local_map: LocalMap = field(default_factory=LocalMap)
    global_map: Optional[OccupancyGrid] = None
    
    # 导航信息
    obstacles: List[Obstacle] = field(default_factory=list)
    danger_zones: List[Tuple[float, float, float]] = field(default_factory=list)  # (x, y, radius)
    path_clear: bool = True
    
    # 元信息
    timestamp: float = 0.0
    processing_time_ms: float = 0.0
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        """转换为字典 (用于API响应)"""
        return {
            # 感知
            "detections": [
                {
                    "class": obj.class_name,
                    "category": obj.category.value,
                    "bbox": list(obj.bbox),
                    "confidence": float(obj.confidence),
                    "track_id": obj.track_id
                }
                for obj in self.objects
            ],
            "scene_type": self.scene_type.value,
            
            # SLAM
            "pose": {
                "x": float(self.pose.x),
                "y": float(self.pose.y),
                "theta": float(self.pose.theta),
                "timestamp": self.timestamp
            },
            
            # 导航
            "obstacles": [
                {
                    "x": o.position[0],
                    "y": o.position[1],
                    "size": list(o.size),
                    "category": o.category,
                    "dangerous": self._is_dangerous(o)
                }
                for o in self.obstacles
            ],
            "danger_zones": [
                {"x": d[0], "y": d[1], "radius": d[2]}
                for d in self.danger_zones
            ],
            "path_clear": self.path_clear,
            
            # 元信息
            "timestamp": self.timestamp,
            "processing_time_ms": self.processing_time_ms,
            "confidence": self.confidence
        }
    
    def _is_dangerous(self, obstacle: Obstacle) -> bool:
        """判断障碍物是否危险"""
        # 基于类别判断
        dangerous_categories = ["person", "vehicle", "forklift", "obstacle"]
        if obstacle.category.lower() in dangerous_categories:
            # 基于距离判断
            dist = np.sqrt(obstacle.position[0]**2 + obstacle.position[1]**2)
            return dist < 2.0  # 2米内危险
        return False


class UnifiedPerceptionSLAM:
    """
    统一感知+SLAM系统
    
    功能:
    1. 视觉感知: 目标检测/分割/追踪
    2. SLAM定位: 视觉/激光/融合定位
    3. 地图构建: 占据栅格/语义地图
    4. 融合感知: 将检测目标投影到地图
    """
    
    def __init__(self, 
                 perception_config: Optional[Dict] = None,
                 slam_config: Optional[Dict] = None,
                 fusion_level: FusionLevel = FusionLevel.LOOSE):
        
        self.fusion_level = fusion_level
        
        # 1. 初始化感知模块
        logger.info("[1/3] Initializing perception module...")
        self.perception = LogisticsPerception(perception_config)
        try:
            self.perception.load_models()
        except Exception as e:
            logger.warning(f"[!] Perception model load failed: {e}")
        
        # 2. 初始化SLAM模块
        logger.info("[2/3] Initializing SLAM module...")
        _config = slam_config or SLAMConfig.FUSION
        slam_type = _config.get("slam_type", SLAMType.FUSION)
        self.slam = create_slam(slam_type, slam_config)
        self.slam.start()
        
        # 3. 初始化状态
        self.result = AGVPerceptionResult()
        self.local_map = LocalMap()
        
        # 相机内参 (用于3D投影)
        self.K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
        self.depth_scale = 1000.0  # 假设深度图单位为mm
        
        logger.info("[3/3] UnifiedPerceptionSLAM ready!")
        logger.info(f"    Fusion level: {fusion_level.value}")
        logger.info(f"    SLAM type: {slam_type.value}")
    
    def update(self,
               frame: Optional[np.ndarray] = None,
               depth: Optional[np.ndarray] = None,
               scan: Optional[np.ndarray] = None,
               odometry: Optional[Tuple] = None) -> AGVPerceptionResult:
        """
        统一更新感知和SLAM
        
        Args:
            frame: RGB图像
            depth: 深度图像 (可选)
            scan: 激光扫描 (可选)
            odometry: 里程计 (vx, vy, vtheta)
        
        Returns:
            AGVPerceptionResult: 完整感知结果
        """
        t_start = time.time()
        timestamp = time.time()
        
        # 1. 视觉感知
        if frame is not None:
            self._update_perception(frame)
        
        # 2. SLAM更新
        self._update_slam(frame, scan, odometry)
        
        # 3. 融合感知和SLAM
        self._fuse_perception_slam(depth)
        
        # 4. 生成导航信息
        self._generate_navigation_info()
        
        # 更新结果
        self.result.timestamp = timestamp
        self.result.processing_time_ms = (time.time() - t_start) * 1000
        
        return self.result
    
    def _update_perception(self, frame: np.ndarray):
        """更新视觉感知"""
        # 目标检测
        objects = self.perception.detect(frame)
        
        # 场景理解
        scene = self.perception.understand_scene(frame)
        
        self.result.objects = scene.objects
        self.result.scene_type = scene.scene_type
    
    def _update_slam(self, 
                     frame: Optional[np.ndarray],
                     scan: Optional[np.ndarray],
                     odometry: Optional[Tuple]):
        """更新SLAM"""
        if isinstance(self.slam, VisualSLAM) and frame is not None:
            state = self.slam.update(frame)
        elif isinstance(self.slam, LidarSLAM) and scan is not None:
            state = self.slam.update(scan, odometry)
        elif isinstance(self.slam, FusionSLAM):
            state = self.slam.update(frame, scan, odometry)
        else:
            return
        
        self.result.pose = state.pose
        self.result.global_map = state.map
        self.local_map.pose = state.pose
        self.local_map.static_map = state.map
        self.local_map.timestamp = state.timestamp
    
    def _fuse_perception_slam(self, depth: Optional[np.ndarray]):
        """融合感知和SLAM"""
        pose = self.result.pose
        
        obstacles = []
        
        for obj in self.result.objects:
            # 计算物体中心
            x1, y1, x2, y2 = obj.bbox
            
            # 像素坐标
            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            
            # 估计距离 (如果有深度图)
            if depth is not None:
                z = depth[int(v), int(u)] / self.depth_scale
            else:
                # 假设固定高度
                z = 2.0  # 2米
            
            # 从像素坐标转换到世界坐标
            world_x = pose.x + z * np.cos(pose.theta)
            world_y = pose.y + z * np.sin(pose.theta)
            
            # 创建障碍物
            obstacle = Obstacle(
                position=(world_x, world_y),
                size=(x2 - x1, y2 - y1),
                category=obj.class_name,
                confidence=obj.confidence,
                source="perception",
                id=obj.track_id
            )
            obstacles.append(obstacle)
        
        self.result.obstacles = obstacles
        self.local_map.obstacles = obstacles
    
    def _generate_navigation_info(self):
        """生成导航信息"""
        pose = self.result.pose
        
        # 危险区域
        danger_zones = []
        for obs in self.result.obstacles:
            if obs.confidence > 0.7:
                # 人员周围1.5米为危险区
                if obs.category.lower() == "person":
                    danger_zones.append((
                        obs.position[0], 
                        obs.position[1], 
                        1.5
                    ))
        
        self.result.danger_zones = danger_zones
        
        # 路径是否通畅
        self.path_clear = len([
            d for d in self.result.danger_zones 
            if np.sqrt(d[0]**2 + d[1]**2) < 1.0
        ]) == 0
        
        self.local_map.traversable = self._compute_traversable()
        self.local_map.dynamic_objects = [
            o for o in self.result.obstacles 
            if o.category.lower() in ["person", "vehicle", "forklift"]
        ]
    
    def _compute_traversable(self) -> List[Tuple[float, float]]:
        """计算可通行区域"""
        if self.result.global_map is None:
            return []
        
        pose = self.result.pose
        traversable = []
        
        # 检查前方扇形区域
        for r in np.arange(0.5, 5.0, 0.5):  # 0.5-5米
            for angle in np.arange(-np.pi/3, np.pi/3, np.pi/12):  # ±60度
                wx = pose.x + r * np.cos(pose.theta + angle)
                wy = pose.y + r * np.sin(pose.theta + angle)
                
                gx, gy = self.result.global_map.world_to_grid(wx, wy)
                
                if self.result.global_map.is_free(gx, gy):
                    traversable.append((wx, wy))
        
        return traversable
    
    def reset(self):
        """重置系统"""
        self.perception = LogisticsPerception()
        try:
            self.perception.load_models()
        except:
            pass
        
        self.slam.reset()
        self.result = AGVPerceptionResult()
        logger.info("[OK] UnifiedPerceptionSLAM reset")
    
    def get_status(self) -> Dict:
        """获取系统状态"""
        return {
            "slam_running": self.slam.is_running,
            "slam_type": type(self.slam).__name__,
            "fusion_level": self.fusion_level.value,
            "last_update": self.result.timestamp,
            "objects_tracked": len(self.result.objects),
            "processing_time_ms": self.result.processing_time_ms
        }


# ==================== 工厂函数 ====================

class SLAMConfig:
    """SLAM配置"""
    
    # 视觉SLAM配置
    VISUAL = {
        "slam_type": SLAMType.VISUAL,
        "map_size": 100,
        "resolution": 0.05,
    }
    
    # 激光SLAM配置
    LIDAR = {
        "slam_type": SLAMType.LIDAR,
        "map_size": 200,
        "resolution": 0.05,
        "num_particles": 100,
        "max_range": 10.0,
    }
    
    # 融合SLAM配置
    FUSION = {
        "slam_type": SLAMType.FUSION,
        "viz_weight": 0.3,
        "lidar_weight": 0.7,
        "map_size": 200,
        "resolution": 0.05,
    }


def create_unified_system(system_type: str = "fusion") -> UnifiedPerceptionSLAM:
    """创建统一感知SLAM系统"""
    
    configs = {
        "visual": (SLAMConfig.VISUAL, FusionLevel.TIGHT),
        "lidar": (SLAMConfig.LIDAR, FusionLevel.LOOSE),
        "fusion": (SLAMConfig.FUSION, FusionLevel.TIGHT),
    }
    
    config, fusion = configs.get(system_type, SLAMConfig.FUSION)
    
    return UnifiedPerceptionSLAM(
        slam_config=config,
        fusion_level=fusion
    )


# ==================== 简易封装 ====================

class SimpleAGVPerception:
    """
    简化版AGV感知 (只用摄像头)
    适用于没有激光雷达的场景
    """
    
    def __init__(self):
        self.perception = LogisticsPerception()
        try:
            self.perception.load_models()
        except:
            pass
        
        # 简化的里程计估计
        self.prev_pose = Pose2D()
        self.velocity = [0.0, 0.0]
        self.frame_count = 0
        
        logger.info("[OK] SimpleAGVPerception initialized")
    
    def update(self, frame: np.ndarray) -> AGVPerceptionResult:
        """更新感知"""
        t_start = time.time()
        
        # 1. 目标检测
        objects = self.perception.detect(frame)
        scene = self.perception.understand_scene(frame)
        
        # 2. 简单运动估计 (基于检测框移动)
        self._estimate_motion(scene.objects)
        
        # 3. 生成结果
        result = AGVPerceptionResult(
            objects=scene.objects,
            scene_type=scene.scene_type,
            pose=self.prev_pose,
            local_map=LocalMap(pose=self.prev_pose),
            obstacles=[
                Obstacle(
                    position=(0.5 * i, 0.5 * i),
                    size=(100, 100),
                    category=obj.class_name,
                    confidence=obj.confidence,
                    source="perception",
                    id=obj.track_id
                )
                for i, obj in enumerate(scene.objects[:5])
            ],
            timestamp=time.time(),
            processing_time_ms=(time.time() - t_start) * 1000
        )
        
        # 4. 判断路径是否通畅
        result.path_clear = not any(
            obj.category == ObjectCategory.PERSON and obj.confidence > 0.8
            for obj in scene.objects
        )
        
        return result
    
    def _estimate_motion(self, objects: List[DetectedObject]):
        """估计运动"""
        # 简化: 基于检测框中心的变化估计运动
        if len(objects) > 0 and self.frame_count > 1:
            avg_dx = np.mean([obj.bbox[0] for obj in objects]) - 320
            self.velocity[0] = 0.95 * self.velocity[0] + 0.05 * avg_dx * 0.001
            
            self.prev_pose.x += self.velocity[0]
            self.prev_pose.theta += self.velocity[1]
        
        self.frame_count += 1


if __name__ == "__main__":
    print("="*60)
    print("Testing Unified Perception + SLAM")
    print("="*60)
    
    # 测试统一系统
    print("\n1. Testing UnifiedPerceptionSLAM...")
    system = create_unified_system("fusion")
    
    # 模拟数据
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    scan = np.random.uniform(0.5, 8.0, 360)
    
    result = system.update(frame=frame, scan=scan)
    
    print(f"   Pose: x={result.pose.x:.2f}, y={result.pose.y:.2f}")
    print(f"   Detections: {len(result.objects)}")
    print(f"   Obstacles: {len(result.obstacles)}")
    print(f"   Path clear: {result.path_clear}")
    print(f"   Processing time: {result.processing_time_ms:.1f}ms")
    
    # 测试简单版本
    print("\n2. Testing SimpleAGVPerception...")
    simple = SimpleAGVPerception()
    simple_result = simple.update(frame)
    
    print(f"   Detections: {len(simple_result.objects)}")
    print(f"   Path clear: {simple_result.path_clear}")
    
    print("\n" + "="*60)
    print("[OK] All tests passed!")
    print("="*60)
