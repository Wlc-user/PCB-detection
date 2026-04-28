# 摔倒检测（基于人体关键点时序）
class FallDetector:
    \"\"\"基于姿态关键点变化的摔倒检测\"\"\"
    def __init__(self, velocity_threshold=500, angle_threshold=45):
        self.v_thresh = velocity_threshold
        self.a_thresh = angle_threshold
    
    def detect(self, keypoints_sequence):
        \"\"\"
        keypoints_sequence: [T, 17, 2]  # T帧，17个关键点，(x,y)
        核心逻辑：头部关键点垂直速度 > 阈值 + 躯干角度变化 > 阈值 → 摔倒
        \"\"\"
        head_y = keypoints_sequence[:, 0, 1]  # 头部y坐标
        velocity = abs(head_y[-1] - head_y[0]) / len(keypoints_sequence)
        
        # 躯干角度（肩-髋连线与垂直方向夹角）
        shoulder = keypoints_sequence[:, 5, :]
        hip = keypoints_sequence[:, 11, :]
        angle = np.abs(np.arctan2(shoulder[:,0]-hip[:,0], shoulder[:,1]-hip[:,1]))
        angle_change = angle[-1] - angle[0]
        
        is_fall = velocity > self.v_thresh and angle_change > self.a_thresh
        return is_fall, {'velocity': velocity, 'angle_change': angle_change}
