"""
SLAM 算法核心模块集合
包含: EKF-SLAM, Particle SLAM, Visual Odometry, Graph SLAM 等
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


@dataclass
class Landmark:
    """路标点"""
    x: float
    y: float
    id: int
    observed: bool = False


@dataclass
class RobotState:
    """机器人状态"""
    x: float
    y: float
    theta: float  # 偏航角
    
    def to_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])


class EKFSLAM:
    """
    扩展卡尔曼滤波 SLAM
    基于 EKF 的同步定位与地图构建
    """
    
    def __init__(self, initial_pose: RobotState, landmark_init_cov: float = 1.0):
        self.state = initial_pose
        self.dim_state = 3  # x, y, theta
        
        # 状态向量 [x, y, theta, l1_x, l1_y, l2_x, l2_y, ...]
        self.state_vec = initial_pose.to_vector()
        self.covariance = np.eye(self.dim_state) * 0.1
        
        # 路标点列表
        self.landmarks: List[Landmark] = []
        self.landmark_dim = 2
        
        # 过程噪声和观测噪声
        self.Q = np.diag([0.05, 0.05, 0.02])  # 过程噪声
        self.R = np.diag([0.1, 0.1])           # 观测噪声
        
        self.landmark_init_cov = landmark_init_cov
        
    def predict(self, u: np.ndarray, dt: float):
        """
        运动预测
        u: [v, omega] 线速度和角速度
        """
        v, omega = u
        
        # 更新状态
        if abs(omega) < 1e-6:
            delta_x = v * dt * np.cos(self.state_vec[2])
            delta_y = v * dt * np.sin(self.state_vec[2])
        else:
            delta_x = v / omega * (np.sin(self.state_vec[2] + omega * dt) - np.sin(self.state_vec[2]))
            delta_y = v / omega * (-np.cos(self.state_vec[2] + omega * dt) + np.cos(self.state_vec[2]))
        delta_theta = omega * dt
        
        # 状态更新
        self.state_vec[:3] += np.array([delta_x, delta_y, delta_theta])
        self.state_vec[2] = np.arctan2(np.sin(self.state_vec[2]), np.cos(self.state_vec[2]))
        
        # 雅可比矩阵
        F = np.eye(len(self.state_vec))
        F[:3, :3] = self._jacobian_motion(self.state_vec, u, dt)
        
        # 协方差更新
        self.covariance = F @ self.covariance @ F.T
        self.covariance[:3, :3] += self.Q
        
    def _jacobian_motion(self, state: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """运动模型雅可比矩阵"""
        v, omega = u
        theta = state[2]
        
        if abs(omega) < 1e-6:
            J = np.array([
                [1, 0, -v * dt * np.sin(theta)],
                [0, 1,  v * dt * np.cos(theta)],
                [0, 0, 1]
            ])
        else:
            J = np.array([
                [1, 0, v/omega * (np.cos(theta + omega*dt) - np.cos(theta))],
                [0, 1, v/omega * (np.sin(theta + omega*dt) - np.sin(theta))],
                [0, 0, 1]
            ])
        return J
    
    def add_landmark(self, range bearing: float, angle: float) -> int:
        """
        添加新路标点
        range: 距离
        bearing: 观测角度
        返回路标点索引
        """
        # 全局坐标
        lx = self.state_vec[0] + range * np.cos(self.state_vec[2] + bearing)
        ly = self.state_vec[1] + range * np.sin(self.state_vec[2] + bearing)
        
        landmark = Landmark(lx, ly, len(self.landmarks))
        self.landmarks.append(landmark)
        
        # 扩展状态向量
        self.state_vec = np.append(self.state_vec, [lx, ly])
        
        # 扩展协方差矩阵
        n = len(self.covariance)
        new_cov = np.zeros((n + 2, n + 2))
        new_cov[:n, :n] = self.covariance
        new_cov[n:, :n] = self._landmark_jacobian(range, angle)
        new_cov[:n, n:] = new_cov[n:, :n].T
        new_cov[n:, n:] = np.eye(2) * self.landmark_init_cov
        self.covariance = new_cov
        
        return len(self.landmarks) - 1
    
    def _landmark_jacobian(self, r: float, b: float) -> np.ndarray:
        """路标点相关雅可比"""
        theta = self.state_vec[2]
        dx = r * np.cos(theta + b)
        dy = r * np.sin(theta + b)
        d = np.sqrt(dx**2 + dy**2) + 1e-6
        
        J = np.zeros((2, 3))
        J[0, 0] = -dx / d
        J[0, 1] = -dy / d
        J[0, 2] = -r * np.sin(theta + b)
        J[1, 0] = -dy / d**2
        J[1, 1] =  dx / d**2
        J[1, 2] =  r * np.cos(theta + b)
        return J
    
    def update(self, observations: List[Tuple[int, float, float]]):
        """
        观测更新
        observations: [(landmark_id, range, bearing), ...]
        """
        for lm_id, z_range, z_bearing in observations:
            if lm_id >= len(self.landmarks):
                continue
                
            # 预测观测
            lm = self.landmarks[lm_id]
            dx = lm.x - self.state_vec[0]
            dy = lm.y - self.state_vec[1]
            
            z_pred_range = np.sqrt(dx**2 + dy**2)
            z_pred_bearing = np.arctan2(dy, dx) - self.state_vec[2]
            z_pred_bearing = np.arctan2(np.sin(z_pred_bearing), np.cos(z_pred_bearing))
            
            # 观测残差
            y = np.array([z_range - z_pred_range, z_bearing - z_pred_bearing])
            y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))
            
            # 雅可比矩阵 H
            d = z_pred_range**2 + 1e-6
            H = np.array([
                [-dx/np.sqrt(d), -dy/np.sqrt(d), 0],
                [dy/d, -dx/d, -1]
            ])
            
            # 扩展 H 到全状态维度
            n = 3 + lm_id * 2
            H_full = np.zeros((2, len(self.state_vec)))
            H_full[:, :3] = H
            H_full[:, n:n+2] = H[:, :2]
            
            #卡尔曼增益
            S = H_full @ self.covariance @ H_full.T + self.R
            K = self.covariance @ H_full.T @ np.linalg.inv(S)
            
            # 更新
            self.state_vec += (K @ y).flatten()
            self.covariance = (np.eye(len(self.state_vec)) - K @ H_full) @ self.covariance
    
    def get_pose(self) -> RobotState:
        """获取当前位姿"""
        return RobotState(self.state_vec[0], self.state_vec[1], self.state_vec[2])
    
    def get_map(self) -> List[Landmark]:
        """获取地图"""
        for i, lm in enumerate(self.landmarks):
            if i * 2 + 2 < len(self.state_vec):
                lm.x = self.state_vec[3 + i * 2]
                lm.y = self.state_vec[3 + i * 2 + 1]
        return self.landmarks


class ParticleSLAM:
    """
    粒子滤波 SLAM (FastSLAM 算法)
    使用粒子滤波进行定位和地图构建
    """
    
    def __init__(self, num_particles: int = 100):
        self.num_particles = num_particles
        self.particles = []
        self.map = []
        
        # 初始化粒子
        for _ in range(num_particles):
            self.particles.append({
                'pose': np.array([0.0, 0.0, 0.0]),
                'weight': 1.0 / num_particles,
                'map': []  # 路标点列表
            })
    
    def predict(self, u: np.ndarray, dt: float):
        """运动预测"""
        v, omega = u
        
        for p in self.particles:
            theta = p['pose'][2]
            
            # 添加噪声
            v_noise = v + np.random.randn() * 0.05
            omega_noise = omega + np.random.randn() * 0.02
            
            if abs(omega_noise) < 1e-6:
                delta_x = v_noise * dt * np.cos(theta)
                delta_y = v_noise * dt * np.sin(theta)
            else:
                delta_x = v_noise / omega_noise * (np.sin(theta + omega_noise * dt) - np.sin(theta))
                delta_y = v_noise / omega_noise * (-np.cos(theta + omega_noise * dt) + np.cos(theta))
            
            delta_theta = omega_noise * dt
            
            p['pose'] += np.array([delta_x, delta_y, delta_theta])
            p['pose'][2] = np.arctan2(np.sin(p['pose'][2]), np.cos(p['pose'][2]))
    
    def update(self, observations: List[Tuple[float, float]], weights: Optional[np.ndarray] = None):
        """观测更新"""
        if weights is None:
            weights = np.ones(self.num_particles) / self.num_particles
        
        # 重要性采样
        indices = np.random.choice(self.num_particles, self.num_particles, p=weights)
        self.particles = [self.particles[i].copy() for i in indices]
        
        # 重置权重
        for p in self.particles:
            p['weight'] = 1.0 / self.num_particles
    
    def resample(self):
        """重采样"""
        weights = np.array([p['weight'] for p in self.particles])
        weights /= weights.sum()
        
        indices = np.random.choice(self.num_particles, self.num_particles, p=weights)
        self.particles = [self.particles[i].copy() for i in indices]
        
        for p in self.particles:
            p['weight'] = 1.0 / self.num_particles
    
    def get_best_pose(self) -> np.ndarray:
        """获取最优粒子位姿"""
        best_idx = np.argmax([p['weight'] for p in self.particles])
        return self.particles[best_idx]['pose']
    
    def get_mean_pose(self) -> np.ndarray:
        """获取平均位姿"""
        poses = np.array([p['pose'] for p in self.particles])
        mean_pose = poses.mean(axis=0)
        mean_pose[2] = np.arctan2(
            np.sin(poses[:, 2]).mean(),
            np.cos(poses[:, 2]).mean()
        )
        return mean_pose


class VisualOdometry:
    """
    视觉里程计
    基于特征点的单目视觉里程计实现
    """
    
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray = None):
        self.K = camera_matrix
        self.D = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        
        self.prev_image = None
        self.prev_features = None
        self.prev_descriptors = None
        self.prev_pose = np.eye(4)
        self.trajectory = []
        
        # 内参
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]
    
    def detect_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测图像特征点
        返回: (特征点坐标, 描述子)
        """
        import cv2
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # ORB 特征检测
        orb = cv2.ORB_create(nfeatures=1000)
        features, descriptors = orb.detectAndCompute(gray, None)
        
        # 转换为 numpy 数组
        if features:
            points = np.array([[f.pt[0], f.pt[1]] for f in features], dtype=np.float32)
        else:
            points = np.array([]).reshape(0, 2)
        
        return points, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
        """特征匹配"""
        import cv2
        
        if desc1 is None or desc2 is None:
            return np.array([])
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        
        return np.array([m.queryIdx for m in matches]), np.array([m.trainIdx for m in matches])
    
    def estimate_motion(self, curr_pts: np.ndarray, prev_pts: np.ndarray) -> np.ndarray:
        """
        估计相机运动
        curr_pts: 当前帧特征点
        prev_pts: 上一帧特征点
        返回: 4x4 变换矩阵
        """
        import cv2
        
        if len(curr_pts) < 8:
            return np.eye(4)
        
        # 恢复相对姿态
        E, mask = cv2.findEssentialMat(curr_pts, prev_pts, self.K, cv2.RANSAC, 0.999, 1.0)
        
        if E is None:
            return np.eye(4)
        
        _, R, t, mask = cv2.recoverPose(E, curr_pts, prev_pts, self.K)
        
        # 构建变换矩阵
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        
        return T
    
    def process_frame(self, image: np.ndarray) -> np.ndarray:
        """处理单帧图像"""
        curr_pts, curr_desc = self.detect_features(image)
        
        if self.prev_features is not None:
            # 特征匹配
            q_idx, t_idx = self.match_features(self.prev_descriptors, curr_desc)
            
            if len(q_idx) > 0:
                prev_matched = self.prev_features[q_idx]
                curr_matched = curr_pts[t_idx]
                
                # 估计运动
                T = self.estimate_motion(curr_matched, prev_matched)
                
                # 更新位姿
                self.prev_pose = self.prev_pose @ np.linalg.inv(T)
                self.trajectory.append(self.prev_pose[:3, 3].copy())
        
        # 更新上一帧
        self.prev_image = image.copy()
        self.prev_features = curr_pts
        self.prev_descriptors = curr_desc
        
        return self.prev_pose
    
    def get_trajectory(self) -> np.ndarray:
        """获取轨迹"""
        return np.array(self.trajectory) if self.trajectory else np.array([]).reshape(0, 3)


class GraphSLAM:
    """
    基于图优化的 SLAM
    使用 g2o 风格的核心实现
    """
    
    def __init__(self):
        self.nodes = []      # 图节点: (pose_id, x, y, theta)
        self.edges = []      # 边: (id1, id2, dx, dy, dtheta, info_matrix)
        self.dimension = 3
        
    def add_node(self, pose_id: int, x: float, y: float, theta: float):
        """添加节点"""
        self.nodes.append((pose_id, x, y, theta))
    
    def add_edge(self, id1: int, id2: int, dx: float, dy: float, dtheta: float, 
                 info_matrix: np.ndarray = None):
        """添加边(约束)"""
        if info_matrix is None:
            info_matrix = np.eye(3) * 10  # 默认信息矩阵
        
        self.edges.append((id1, id2, dx, dy, dtheta, info_matrix))
    
    def linearize(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        线性化图
        返回: (残差向量, 雅可比矩阵块)
        """
        residuals = []
        J_blocks = []
        
        for id1, id2, dx, dy, dtheta, info in self.edges:
            # 找到节点
            node1 = self.nodes[id1] if id1 < len(self.nodes) else None
            node2 = self.nodes[id2] if id2 < len(self.nodes) else None
            
            if node1 is None or node2 is None:
                continue
            
            # 计算预测
            pred_dx = node2[1] - node1[1]
            pred_dy = node2[2] - node1[2]
            pred_dtheta = node2[3] - node1[3]
            
            # 残差
            r = np.array([
                dx - pred_dx,
                dy - pred_dy,
                dtheta - pred_dtheta
            ])
            r[2] = np.arctan2(np.sin(r[2]), np.cos(r[2]))
            
            residuals.append(r)
            
            # 雅可比
            J1 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
            J2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            
            J_blocks.append((id1, id2, J1, J2))
        
        return np.array(residuals).flatten(), J_blocks
    
    def optimize(self, max_iterations: int = 100, tolerance: float = 1e-6) -> np.ndarray:
        """
        优化图
        返回优化后的状态向量
        """
        # 构建初始状态向量
        state = np.zeros(len(self.nodes) * self.dimension)
        for i, (node_id, x, y, theta) in enumerate(self.nodes):
            state[i * self.dimension:(i + 1) * self.dimension] = [x, y, theta]
        
        for iteration in range(max_iterations):
            residuals, J_blocks = self.linearize(state)
            
            # 计算增量
            H = np.zeros((len(state), len(state)))
            b = np.zeros(len(state))
            
            for i, ((id1, id2, J1, J2), r) in enumerate(zip(J_blocks, residuals.reshape(-1, 3))):
                H[id1*3:(id1+1)*3, id1*3:(id1+1)*3] += J1.T @ J1
                H[id1*3:(id1+1)*3, id2*3:(id2+1)*3] += J1.T @ J2
                H[id2*3:(id2+1)*3, id1*3:(id1+1)*3] += J2.T @ J1
                H[id2*3:(id2+1)*3, id2*3:(id2+1)*3] += J2.T @ J2
                b[id1*3:(id1+1)*3] += J1.T @ r
                b[id2*3:(id2+1)*3] += J2.T @ r
            
            # 求解
            try:
                delta = -np.linalg.solve(H + np.eye(len(state)) * 0.1, b)
            except np.linalg.LinAlgError:
                break
            
            state += delta
            
            # 检查收敛
            if np.linalg.norm(delta) < tolerance:
                break
        
        return state


if __name__ == "__main__":
    # 测试 EKF SLAM
    ekf = EKFSLAM(RobotState(0, 0, 0))
    
    # 模拟运动
    for i in range(10):
        u = np.array([1.0, 0.1])  # v, omega
        ekf.predict(u, dt=0.1)
        print(f"Step {i}: pose = {ekf.get_pose()}")
    
    # 测试视觉里程计
    K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]])
    vo = VisualOdometry(K)
    
    print("\nVisual Odometry initialized")
    print(f"Camera Matrix:\n{K}")
