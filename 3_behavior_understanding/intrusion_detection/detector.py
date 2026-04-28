# 闯入检测（基于ROI规则）
class IntrusionDetector:
    \"\"\"ROI区域闯入检测\"\"\"
    def __init__(self, roi_polygon):
        self.roi = roi_polygon  # [(x1,y1), (x2,y2), ...]
    
    def is_inside(self, point):
        \"\"\"射线法判断点是否在多边形内\"\"\"
        x, y = point
        n = len(self.roi)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.roi[i]
            xj, yj = self.roi[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    def detect(self, detections):
        \"\"\"检测目标是否闯入禁区\"\"\"
        intruders = []
        for det in detections:
            center = ((det['bbox'][0]+det['bbox'][2])/2, (det['bbox'][1]+det['bbox'][3])/2)
            if self.is_inside(center):
                intruders.append(det)
        return intruders
