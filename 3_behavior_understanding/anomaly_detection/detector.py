# 异常行为检测（基于时序特征）
import numpy as np

class AnomalyDetector:
    \"\"\"基于重建误差的异常行为检测\"\"\"
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.normal_patterns = []
    
    def fit(self, normal_sequences):
        \"\"\"记忆正常行为模式\"\"\"
        self.normal_patterns = [s.mean(0) for s in normal_sequences]
    
    def detect(self, sequence):
        \"\"\"检测异常：重建误差 > 阈值 → 异常\"\"\"
        errors = [np.linalg.norm(sequence - p) for p in self.normal_patterns]
        return min(errors) > self.threshold, min(errors)
