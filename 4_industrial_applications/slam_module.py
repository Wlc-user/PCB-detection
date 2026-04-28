"""
无人物流 SLAM 模块
支持: 视觉SLAM / 激光SLAM / 视觉-激光融合SLAM
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SLAMType(Enum):
    """SLAM类型"""
    VISUAL = "visual"           # 纯视觉SLAM
    LIDAR = "lidar"             # 激光SLAM
    FUSION = "fusion"           # 视觉-激光融合


class MapType(Enum):
    """地图类型"""
    GRID = "grid"               # 2D占据栅格地图
    POINT_CLOUD = "pointcloud"   # 3D点云地图
    MESH = "mesh"               # 3D mesh
    SEMANTIC = "semantic"       # 语义地图


@dataclass
class Pose2D:
    """2D位姿"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # 弧度
    timestamp: float = 0.0
    
    def to_matrix(self) -> np.ndarray:
        """转换为齐次变换矩阵"""
        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        return np.array([
            [cos_t, -sin_t, self.x],
            [sin_t,  cos_t, self.y],
            [0,      0,     1]
        ])
    
    def distance_to(self, other: 'Pose2D') -> float:
        """到另一个位姿的距离"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class Pose3D:
    """3D位姿"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0   # 弧度
    pitch: float = 0.0  # 弧度
    yaw: float = 0.0    # 弧度
    timestamp: float = 0.0
    
    @staticmethod
    def from_se3(matrix: np.ndarray, timestamp: float = 0.0) -> 'Pose3D':
        """从SE(3)矩阵创建"""
        pose = Pose3D()
        pose.x = matrix[0, 3]
        pose.y = matrix[1, 3]
        pose.z = matrix[2, 3]
        # 提取旋转矩阵
        R = matrix[:3, :3]
        # yaw (Z轴旋转)
        pose.yaw = np.arctan2(R[1, 0], R[0, 0])
        # pitch (Y轴旋转)
        pose.pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        # roll (X轴旋转)
        pose.roll = np.arctan2(R[2, 1], R[2, 2])
        pose.timestamp = timestamp
        return pose


@dataclass
class OccupancyGrid:
    """2D占据栅格地图"""
    width: int = 0
    height: int = 0
    resolution: float = 0.05  # 米/像素
    origin: Tuple[float, float] = (0.0, 0.0)  # 左下角世界坐标
    data: np.ndarray = None   # 占据概率 (-1=未知, 0=空闲, 1=占据)
    
    def __post_init__(self):
        if self.data is None:
            self.data = np.full((self.height, self.width), -1, dtype=np.float32)
    
    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """世界坐标转栅格坐标"""
        gx = int((wx - self.origin[0]) / self.resolution)
        gy = int((wy - self.origin[1]) / self.resolution)
        return gx, gy
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """栅格坐标转世界坐标"""
        wx = gx * self.resolution + self.origin[0]
        wy = gy * self.resolution + self.origin[1]
        return wx, wy
    
    def is_free(self, gx: int, gy: int) -> bool:
        """判断栅格是否可通行"""
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return self.data[gy, gx] < 0.5
        return False


@dataclass
class PointCloudMap:
    """3D点云地图"""
    points: np.ndarray = None  # Nx3 点云
    colors: np.ndarray = None  # Nx3 RGB颜色
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.points is None:
            self.points = np.zeros((0, 3), dtype=np.float32)
        if self.colors is None:
            self.colors = np.zeros((0, 3), dtype=np.float32)


@dataclass
class Landmark:
    """地图地标"""
    id: int
    type: str  # 'corner', 'door', 'landmark'
    position: Tuple[float, float, float]
    descriptor: Optional[np.ndarray] = None
    observations: int = 0


@dataclass
class SLAMState:
    """SLAM状态"""
    pose: Pose2D = field(default_factory=Pose2D)
    velocity: Tuple[float, float] = (0.0, 0.0)  # vx, vy
    map: Optional[OccupancyGrid] = None
    pointcloud_map: Optional[PointCloudMap] = None
    landmarks: List[Landmark] = field(default_factory=list)
    trajectory: List[Pose2D] = field(default_factory=list)
    is_tracking_lost: bool = False
    processing_time_ms: float = 0.0
    timestamp: float = 0.0


class BaseSLAM:
    """SLAM基类"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.state = SLAMState()
        self.is_running = False
        self._lock = threading.Lock()
        
    def _default_config(self) -> Dict:
        return {
            "slam_type": SLAMType.VISUAL,
            "map_size": 100,  # 栅格地图大小
            "resolution": 0.05,  # 5cm/pixel
            "max_range": 10.0,  # 最大感知距离
            "min_range": 0.1,   # 最小感知距离
            "loop_closure": True,
            "relocalization": True,
        }
    
    def start(self):
        """启动SLAM"""
        self.is_running = True
        logger.info(f"[OK] {self.__class__.__name__} started")
    
    def stop(self):
        """停止SLAM"""
        self.is_running = False
        logger.info(f"[OK] {self.__class__.__name__} stopped")
    
    def update(self, *args, **kwargs) -> SLAMState:
        """更新SLAM (子类实现)"""
        raise NotImplementedError
    
    def get_pose(self) -> Pose2D:
        """获取当前位姿"""
        with self._lock:
            return self.state.pose
    
    def get_map(self) -> Optional[OccupancyGrid]:
        """获取地图"""
        with self._lock:
            return self.state.map
    
    def reset(self):
        """重置SLAM"""
        with self._lock:
            self.state = SLAMState()
            logger.info("[OK] SLAM reset")


class VisualSLAM(BaseSLAM):
    """
    视觉SLAM (简化版ORB-SLAM)
    支持: 单目 / 双目 / RGB-D
    """
    
    def __init__(self, camera_type: str = "monocular", config: Optional[Dict] = None):
        super().__init__(config)
        self.camera_type = camera_type
        self.feature_detector = None  # 可用ORB/FAST/SIFT
        self.feature_matcher = None
        self.map_points = {}  # map_id -> 3D point
        self.keyframes = []   # 关键帧列表
        self.frame_count = 0
        self.last_pose = Pose2D()
        
        # 简化的特征跟踪
        self.prev_features = None
        self.tracked_features = {}
        self.next_feature_id = 0
        
        logger.info(f"[OK] VisualSLAM initialized ({camera_type})")
    
    def update(self, frame: np.ndarray, timestamp: Optional[float] = None) -> SLAMState:
        """
        更新视觉SLAM
        Args:
            frame: 图像 (H, W, 3)
            timestamp: 时间戳
        Returns:
            SLAMState: 当前状态
        """
        if timestamp is None:
            timestamp = time.time()
        
        t_start = time.time()
        
        with self._lock:
            self.state.timestamp = time.time()
            # 1. 特征提取 (简化版)
            features = self._extract_features(frame)
            
            # 2. 特征匹配/跟踪
            if self.prev_features is not None:
                matches = self._match_features(self.prev_features, features)
                # 3. 运动估计
                motion = self._estimate_motion(matches)
                # 4. 更新位姿
                self._update_pose(motion)
            else:
                # 第一帧，初始化
                self.state.pose = Pose2D(x=0, y=0, theta=0, timestamp=timestamp)
            
            # 5. 更新地图
            self._update_map(features)
            
            # 6. 关键帧检测
            if self._is_keyframe():
                self._add_keyframe(frame, features)
            
            self.prev_features = features
            self.frame_count += 1
            self.state.processing_time_ms = (time.time() - t_start) * 1000
        
        return self.state
    
    def _extract_features(self, frame: np.ndarray) -> List[Tuple]:
        """提取特征点"""
        # 简化: 使用Shi-Tomasi角点
        gray = frame if len(frame.shape) == 2 else np.mean(frame, axis=2).astype(np.uint8)
        
        # 检测角点
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=200, qualityLevel=0.01, minDistance=10
        )
        
        features = []
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                features.append((float(x), float(y)))
        
        return features
    
    def _match_features(self, prev: List, curr: List) -> List:
        """特征匹配"""
        # 简化: 距离匹配
        matches = []
        for i, (px, py) in enumerate(prev):
            best_dist = float('inf')
            best_idx = -1
            for j, (cx, cy) in enumerate(curr):
                dist = np.sqrt((px-cx)**2 + (py-cy)**2)
                if dist < best_dist and dist < 50:  # 50像素阈值
                    best_dist = dist
                    best_idx = j
            if best_idx >= 0:
                matches.append((i, best_idx, best_dist))
        return matches
    
    def _estimate_motion(self, matches: List) -> Tuple[float, float, float]:
        """估计运动 (简化)"""
        if len(matches) < 8:
            return (0.0, 0.0, 0.0)
        
        # 提取匹配点
        prev_pts = np.array([self.prev_features[m[0]] for m in matches])
        curr_pts = np.array([self.features[m[1]] for m in matches])
        
        # 计算光流中心
        dx = np.mean(curr_pts[:, 0] - prev_pts[:, 0])
        dy = np.mean(curr_pts[:, 1] - prev_pts[:, 1])
        
        # 简化的运动模型
        vx = dx * 0.05  # 假设30fps, 像素->米
        vy = dy * 0.05
        dtheta = 0.0
        
        return (vx, vy, dtheta)
    
    @property
    def features(self):
        return self.prev_features
    
    def _update_pose(self, motion: Tuple):
        """更新位姿"""
        vx, vy, dtheta = motion
        
        # 积分
        self.state.pose.x += vx
        self.state.pose.y += vy
        self.state.pose.theta += dtheta
        
        # 归一化角度
        self.state.pose.theta = np.arctan2(np.sin(self.state.pose.theta), 
                                           np.cos(self.state.pose.theta))
        
        self.state.trajectory.append(Pose2D(
            x=self.state.pose.x,
            y=self.state.pose.y,
            theta=self.state.pose.theta,
            timestamp=self.state.pose.timestamp
        ))
    
    def _update_map(self, features: List):
        """更新地图点"""
        for i, (x, y) in enumerate(features):
            if i not in self.tracked_features:
                self.tracked_features[i] = self.next_feature_id
                self.next_feature_id += 1
    
    def _is_keyframe(self) -> bool:
        """判断是否为关键帧"""
        # 简化的关键帧策略
        return self.frame_count % 30 == 0 and self.prev_features is not None and len(self.prev_features) > 50
    
    def _add_keyframe(self, frame: np.ndarray, features: List):
        """添加关键帧"""
        self.keyframes.append({
            'id': len(self.keyframes),
            'frame': frame.copy(),
            'features': features.copy(),
            'pose': Pose2D(
                x=self.state.pose.x,
                y=self.state.pose.y,
                theta=self.state.pose.theta
            )
        })
        logger.info(f"[VisualSLAM] Keyframe {len(self.keyframes)} added")


class LidarSLAM(BaseSLAM):
    """
    激光SLAM (简化版GMapping/Cartographer)
    使用粒子滤波 + 扫描匹配
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        # 地图参数
        self.map_size = self.config.get("map_size", 100)
        self.resolution = self.config.get("resolution", 0.05)
        
        # 初始化栅格地图
        self.state.map = OccupancyGrid(
            width=self.map_size,
            height=self.map_size,
            resolution=self.resolution,
            origin=(-self.map_size * self.resolution / 2, -self.map_size * self.resolution / 2)
        )
        
        # 粒子滤波参数
        self.num_particles = self.config.get("num_particles", 100)
        self.particles = [self._create_particle() for _ in range(self.num_particles)]
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # 扫描匹配参数
        self.max_range = self.config.get("max_range", 10.0)
        self.min_range = self.config.get("min_range", 0.1)
        
        # 位姿历史
        self.odom_history = []
        
        logger.info(f"[OK] LidarSLAM initialized ({self.num_particles} particles)")
    
    def _create_particle(self) -> Pose2D:
        """创建粒子"""
        return Pose2D(
            x=np.random.uniform(-1, 1),
            y=np.random.uniform(-1, 1),
            theta=np.random.uniform(-np.pi, np.pi),
            timestamp=time.time()
        )
    
    def update(self, scan: np.ndarray, odometry: Optional[Tuple] = None) -> SLAMState:
        """
        更新激光SLAM
        Args:
            scan: 激光扫描数据 (N,) 距离数组, 角度范围假设-PI~PI
            odometry: 里程计数据 (vx, vy, vtheta)
        Returns:
            SLAMState: 当前状态
        """
        t_start = time.time()
        
        with self._lock:
            self.state.timestamp = time.time()
            # 1. 预测 (运动模型)
            if odometry:
                self._motion_update(odometry)
            
            # 2. 扫描匹配
            best_pose = self._scan_matching(scan)
            
            # 3. 更新粒子权重
            self._weight_update(scan, best_pose)
            
            # 4. 重采样
            if self._needs_resampling():
                self._resample()
            
            # 5. 估计最终位姿 (加权平均)
            self._estimate_pose()
            
            # 6. 更新地图
            self._update_map(scan)
            
            self.state.processing_time_ms = (time.time() - t_start) * 1000
        
        return self.state
    
    def _motion_update(self, odometry: Tuple):
        """运动模型更新"""
        vx, vy, vtheta = odometry
        
        for i, p in enumerate(self.particles):
            # 添加噪声的运动模型
            noise = 0.1
            p.x += vx + np.random.randn() * noise
            p.y += vy + np.random.randn() * noise
            p.theta += vtheta + np.random.randn() * noise * 0.1
    
    def _scan_matching(self, scan: np.ndarray) -> Pose2D:
        """扫描匹配 (ICP简化版)"""
        # 简化的扫描匹配: 使用scan-to-map
        best_pose = Pose2D()
        best_score = float('inf')
        
        # 测试几个候选位姿
        for _ in range(9):  # 3x3网格搜索
            test_x = best_pose.x + np.random.uniform(-0.2, 0.2)
            test_y = best_pose.y + np.random.uniform(-0.2, 0.2)
            test_theta = best_pose.theta + np.random.uniform(-0.1, 0.1)
            
            score = self._evaluate_pose(scan, test_x, test_y, test_theta)
            if score < best_score:
                best_score = score
                best_pose = Pose2D(x=test_x, y=test_y, theta=test_theta)
        
        return best_pose
    
    def _evaluate_pose(self, scan: np.ndarray, x: float, y: float, theta: float) -> float:
        """评估位姿得分"""
        score = 0.0
        num_beams = len(scan)
        
        for i, r in enumerate(scan):
            if self.min_range < r < self.max_range:
                # 计算端点
                angle = theta + (i - num_beams/2) * (2*np.pi / num_beams)
                gx = int((x + r * np.cos(angle) - self.state.map.origin[0]) / self.resolution)
                gy = int((y + r * np.sin(angle) - self.state.map.origin[1]) / self.resolution)
                
                # 检查地图
                if 0 <= gx < self.state.map.width and 0 <= gy < self.state.map.height:
                    if self.state.map.data[gy, gx] > 0.5:
                        score += 1.0
        
        return score
    
    def _weight_update(self, scan: np.ndarray, pose: Pose2D):
        """权重更新"""
        # 简化的似然计算
        for i, p in enumerate(self.particles):
            dist = p.distance_to(pose)
            self.weights[i] = np.exp(-dist * 5.0)
        
        # 归一化
        self.weights += 1e-10
        self.weights /= np.sum(self.weights)
    
    def _needs_resampling(self) -> bool:
        """判断是否需要重采样"""
        n_eff = 1.0 / np.sum(self.weights ** 2)
        return n_eff < self.num_particles / 2
    
    def _resample(self):
        """系统重采样"""
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = [Pose2D(
            x=self.particles[i].x,
            y=self.particles[i].y,
            theta=self.particles[i].theta
        ) for i in indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def _estimate_pose(self):
        """估计最终位姿"""
        # 加权平均
        x = np.sum([p.x * w for p, w in zip(self.particles, self.weights)])
        y = np.sum([p.y * w for p, w in zip(self.particles, self.weights)])
        
        # 角度用加权平均
        sin_theta = np.sum([np.sin(p.theta) * w for p, w in zip(self.particles, self.weights)])
        cos_theta = np.sum([np.cos(p.theta) * w for p, w in zip(self.particles, self.weights)])
        theta = np.arctan2(sin_theta, cos_theta)
        
        self.state.pose = Pose2D(x=x, y=y, theta=theta, timestamp=time.time())
        self.state.trajectory.append(Pose2D(x=x, y=y, theta=theta))
    
    def _update_map(self, scan: np.ndarray):
        """更新占据栅格地图"""
        pose = self.state.pose
        num_beams = len(scan)
        
        for i, r in enumerate(scan):
            if self.min_range < r < self.max_range:
                angle = pose.theta + (i - num_beams/2) * (2*np.pi / num_beams)
                
                # 射线起点和终点
                x1, y1 = pose.x, pose.y
                x2 = x1 + r * np.cos(angle)
                y2 = y1 + r * np.sin(angle)
                
                # 标记栅格
                self._mark_line(x1, y1, x2, y2)
    
    def _mark_line(self, x1: float, y1: float, x2: float, y2: float):
        """Bresenham射线标记"""
        gx1, gy1 = self.state.map.world_to_grid(x1, y1)
        gx2, gy2 = self.state.map.world_to_grid(x2, y2)
        
        # Bresenham算法
        dx = abs(gx2 - gx1)
        dy = abs(gy2 - gy1)
        sx = 1 if gx1 < gx2 else -1
        sy = 1 if gy1 < gy2 else -1
        err = dx - dy
        
        cx, cy = gx1, gy1
        while True:
            if 0 <= cx < self.state.map.width and 0 <= cy < self.state.map.height:
                # 射线上的点标记为free
                if cx != gx2 or cy != gy2:
                    if self.state.map.data[cy, cx] < 0:
                        self.state.map.data[cy, cx] = 0
                # 端点标记为occupied
                elif cx == gx2 and cy == gy2:
                    self.state.map.data[cy, cx] = 1
            
            if cx == gx2 and cy == gy2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy


class FusionSLAM(BaseSLAM):
    """
    视觉-激光融合SLAM (简化版LIO-SAM)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        # 初始化子模块
        self.visual_slam = VisualSLAM("stereo", config)
        self.lidar_slam = LidarSLAM(config)
        
        # 融合参数
        self.viz_weight = self.config.get("viz_weight", 0.3)
        self.lidar_weight = self.config.get("lidar_weight", 0.7)
        
        # 初始化地图
        self.state.map = OccupancyGrid(width=100, height=100, resolution=0.05)
        
        logger.info("[OK] FusionSLAM initialized (Visual + Lidar)")
    
    def update(self, 
               frame: Optional[np.ndarray] = None,
               scan: Optional[np.ndarray] = None,
               odometry: Optional[Tuple] = None) -> SLAMState:
        """
        融合更新
        """
        t_start = time.time()
        
        with self._lock:
            self.state.timestamp = time.time()
            viz_pose = None
            lidar_pose = None
            
            # 视觉SLAM更新
            if frame is not None:
                self.visual_slam.update(frame)
                viz_pose = self.visual_slam.get_pose()
            
            # 激光SLAM更新
            if scan is not None:
                self.lidar_slam.update(scan, odometry)
                lidar_pose = self.lidar_slam.get_pose()
            
            # 融合位姿
            if viz_pose is not None and lidar_pose is not None:
                self.state.pose.x = self.viz_weight * viz_pose.x + self.lidar_weight * lidar_pose.x
                self.state.pose.y = self.viz_weight * viz_pose.y + self.lidar_weight * lidar_pose.y
                self.state.pose.theta = self.viz_weight * viz_pose.theta + self.lidar_weight * lidar_pose.theta
            elif viz_pose is not None:
                self.state.pose = viz_pose
            elif lidar_pose is not None:
                self.state.pose = lidar_pose
            
            # 更新轨迹
            self.state.trajectory.append(Pose2D(
                x=self.state.pose.x,
                y=self.state.pose.y,
                theta=self.state.pose.theta,
                timestamp=time.time()
            ))
            
            # 更新地图
            if scan is not None:
                self.state.map = self.lidar_slam.get_map()
            
            self.state.processing_time_ms = (time.time() - t_start) * 1000
        
        return self.state
    
    def start(self):
        self.visual_slam.start()
        self.lidar_slam.start()
        super().start()
    
    def stop(self):
        self.visual_slam.stop()
        self.lidar_slam.stop()
        super().stop()
    
    def reset(self):
        self.visual_slam.reset()
        self.lidar_slam.reset()
        super().reset()


# ==================== SLAM工厂 ====================

def create_slam(slam_type: SLAMType, config: Optional[Dict] = None) -> BaseSLAM:
    """创建SLAM实例 (无IMU版本)"""
    if slam_type == SLAMType.VISUAL:
        return VisualSLAM(config=config)
    elif slam_type == SLAMType.LIDAR:
        return LidarSLAM(config=config)
    elif slam_type == SLAMType.FUSION:
        return FusionSLAM(config=config)
    else:
        raise ValueError(f"Unknown SLAM type: {slam_type}")


if __name__ == "__main__":
    import cv2
    
    # 测试视觉SLAM
    print("Testing VisualSLAM...")
    vslam = VisualSLAM()
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    state = vslam.update(frame)
    print(f"  Pose: x={state.pose.x:.2f}, y={state.pose.y:.2f}, theta={state.pose.theta:.2f}")
    print(f"  Tracking lost: {state.is_tracking_lost}")
    
    # 测试激光SLAM
    print("\nTesting LidarSLAM...")
    lslam = LidarSLAM()
    scan = np.random.uniform(0.5, 8.0, 360)  # 360度激光
    state = lslam.update(scan)
    print(f"  Pose: x={state.pose.x:.2f}, y={state.pose.y:.2f}, theta={state.pose.theta:.2f}")
    print(f"  Map size: {state.map.width}x{state.map.height}")
    
    # 测试融合SLAM
    print("\nTesting FusionSLAM...")
    fslam = FusionSLAM()
    state = fslam.update(frame=frame, scan=scan)
    print(f"  Pose: x={state.pose.x:.2f}, y={state.pose.y:.2f}, theta={state.pose.theta:.2f}")
    
    print("\n[OK] All SLAM modules work!")
