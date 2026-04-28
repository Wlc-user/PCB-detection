"""
物流感知演示脚本
测试分割、检测、分类、追踪功能
"""

import cv2
import numpy as np
from logistics_perception import LogisticsPerception, SceneType, ObjectCategory

def test_detection():
    """测试检测功能"""
    print("\n" + "="*60)
    print("测试: 目标检测")
    print("="*60)
    
    perception = LogisticsPerception()
    
    # 尝试加载模型
    try:
        perception.load_models()
    except Exception as e:
        print(f"[!] 模型加载失败: {e}")
        print("[*] 使用模拟数据演示")
        
        # 创建模拟检测结果
        objects = []
        for i in range(5):
            from logistics_perception import DetectedObject
            obj = DetectedObject()
            obj.track_id = i + 1
            obj.class_name = ["person", "car", "truck", "box", "forklift"][i]
            obj.confidence = 0.7 + np.random.random() * 0.25
            obj.bbox = (100 + i*150, 100, 200 + i*150, 300)
            obj.category = ObjectCategory.UNKNOWN
            objects.append(obj)
        
        return objects
    
    # 使用摄像头测试
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] 无法打开摄像头")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("[!] 无法读取帧")
        return None
    
    # 检测
    result = perception.understand_scene(frame)
    
    print(f"检测到 {len(result.objects)} 个物体:")
    for obj in result.objects:
        print(f"  - {obj.class_name} (ID:{obj.track_id}) 置信度:{obj.confidence:.2f}")
    
    return result.objects


def test_scene_understanding():
    """测试场景理解"""
    print("\n" + "="*60)
    print("测试: 场景理解")
    print("="*60)
    
    perception = LogisticsPerception()
    
    # 模拟仓库场景
    print("\n仓库场景:")
    warehouse_objects = []
    from logistics_perception import DetectedObject
    
    # 添加一些物体
    for i in range(3):
        obj = DetectedObject()
        obj.class_name = "person"
        obj.category = ObjectCategory.PERSON
        warehouse_objects.append(obj)
    
    for i in range(2):
        obj = DetectedObject()
        obj.class_name = "forklift"
        obj.category = ObjectCategory.FORKLIFT
        warehouse_objects.append(obj)
    
    for i in range(10):
        obj = DetectedObject()
        obj.class_name = "box"
        obj.category = ObjectCategory.BOX
        warehouse_objects.append(obj)
    
    stats = perception._analyze_scene(warehouse_objects)
    scene_type = perception._infer_scene_type(warehouse_objects)
    
    print(f"  场景类型: {scene_type.value}")
    print(f"  统计: {stats}")
    
    # 模拟无人车场景
    print("\n无人车场景:")
    vehicle_objects = []
    
    for i in range(2):
        obj = DetectedObject()
        obj.class_name = "car"
        obj.category = ObjectCategory.VEHICLE
        vehicle_objects.append(obj)
    
    obj = DetectedObject()
    obj.class_name = "obstacle"
    obj.category = ObjectCategory.OBSTACLE
    vehicle_objects.append(obj)
    
    stats = perception._analyze_scene(vehicle_objects)
    scene_type = perception._infer_scene_type(vehicle_objects)
    
    print(f"  场景类型: {scene_type.value}")
    print(f"  统计: {stats}")


def demo_video_processing():
    """演示视频处理"""
    print("\n" + "="*60)
    print("演示: 视频流处理")
    print("="*60)
    print("""
功能说明:
1. 分割 (Segmentation)
   - 使用 SAM (Segment Anything Model)
   - 精确分割物体边界
   
2. 检测 (Detection)  
   - YOLOv8/YOLOv10 目标检测
   - 支持 80+ 类别
   
3. 分类 (Classification)
   - 物流专用类别映射
   - 人员、车辆、货物等
   
4. 追踪 (Tracking)
   - ByteTrack 多目标追踪
   - ID 分配和轨迹记录
   
启动服务器:
  python logistics_server.py
  
打开浏览器:
  http://localhost:8000/demo
""")


def print_supported_classes():
    """打印支持的类别"""
    print("\n" + "="*60)
    print("支持的物体类别")
    print("="*60)
    
    print("\n仓库场景:")
    warehouse_classes = [
        ("person", "人员"),
        ("forklift", "叉车"),
        ("pallet", "托盘"),
        ("box", "货物箱"),
        ("shelf", "货架"),
        ("conveyor", "传送带"),
        ("robot", "AGV/AMR机器人"),
    ]
    for en, cn in warehouse_classes:
        print(f"  {en:15} - {cn}")
    
    print("\n无人车场景:")
    vehicle_classes = [
        ("vehicle", "车辆"),
        ("obstacle", "障碍物"),
        ("traffic_sign", "交通标识"),
        ("lane", "车道线"),
        ("danger_zone", "危险区域"),
    ]
    for en, cn in vehicle_classes:
        print(f"  {en:15} - {cn}")
    
    print("\n通用类别:")
    general_classes = [
        ("person", "人员"),
        ("box", "货物"),
        ("robot", "机器人"),
    ]
    for en, cn in general_classes:
        print(f"  {en:15} - {cn}")


if __name__ == "__main__":
    print("="*60)
    print("   无人物流感知系统 - 测试演示")
    print("="*60)
    
    # 打印支持的类别
    print_supported_classes()
    
    # 测试检测
    test_detection()
    
    # 测试场景理解
    test_scene_understanding()
    
    # 演示
    demo_video_processing()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
