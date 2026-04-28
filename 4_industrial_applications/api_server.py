"""
PCB缺陷检测 - 高并发API服务
支持批量推理 + REST API
"""

from ultralytics import YOLO
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass, asdict
from typing import List
import base64
from flask import Flask, request, jsonify
import io

app = Flask(__name__)

# 全局检测器
detector = None
executor = ThreadPoolExecutor(max_workers=4)


@dataclass
class DetectionResult:
    """检测结果"""
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    area: float


class PCBBatchDetector:
    """PCB批量检测器"""
    
    def __init__(self, model_path: str = 'models/yolov8/train/weights/best.pt', 
                 conf: float = 0.2, imgsz: int = 1280):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.class_names = list(self.model.names.values())
        print(f"[Detector] 模型: {model_path}, conf={conf}, imgsz={imgsz}")
    
    def detect(self, image: np.ndarray, conf: float = None) -> List[DetectionResult]:
        """检测单张图片"""
        results = self.model(image, conf=conf or self.conf, imgsz=self.imgsz, verbose=False)
        result = results[0]
        
        detections = []
        if len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, score, cls_id in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                detections.append(DetectionResult(
                    class_name=self.class_names[cls_id],
                    confidence=float(score),
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    area=float((x2-x1)*(y2-y1))
                ))
        
        return detections
    
    def detect_batch(self, images: List[np.ndarray], conf: float = None) -> List[List[DetectionResult]]:
        """批量检测"""
        results = self.model(images, conf=conf or self.conf, imgsz=self.imgsz, verbose=False)
        
        all_detections = []
        for result in results:
            detections = []
            if len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, score, cls_id in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = box
                    detections.append(DetectionResult(
                        class_name=self.class_names[cls_id],
                        confidence=float(score),
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        area=float((x2-x1)*(y2-y1))
                    ))
            all_detections.append(detections)
        
        return all_detections


# ============ API 路由 ============

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({'status': 'ok', 'model': 'PCB Detector'})


@app.route('/detect', methods=['POST'])
def detect():
    """单张图片检测"""
    start = time.time()
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    conf = float(request.form.get('conf', 0.2))
    detections = detector.detect(image, conf=conf)
    
    return jsonify({
        'success': True,
        'detections': [asdict(d) for d in detections],
        'count': len(detections),
        'time': time.time() - start
    })


@app.route('/detect_batch', methods=['POST'])
def detect_batch():
    """批量检测"""
    start = time.time()
    
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No images provided'}), 400
    
    images = []
    for file in files:
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is not None:
            images.append(image)
    
    if not images:
        return jsonify({'error': 'No valid images'}), 400
    
    conf = float(request.form.get('conf', 0.2))
    all_results = detector.detect_batch(images, conf=conf)
    
    return jsonify({
        'success': True,
        'results': [
            {'detections': [asdict(d) for d in detections], 'count': len(detections)}
            for detections in all_results
        ],
        'total_images': len(images),
        'total_detections': sum(len(r) for r in all_results),
        'time': time.time() - start
    })


@app.route('/detect_base64', methods=['POST'])
def detect_base64():
    """Base64图片检测"""
    start = time.time()
    data = request.get_json()
    
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    
    # 解码base64
    img_data = base64.b64decode(data['image'])
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    conf = float(data.get('conf', 0.2))
    detections = detector.detect(image, conf=conf)
    
    return jsonify({
        'success': True,
        'detections': [asdict(d) for d in detections],
        'count': len(detections),
        'time': time.time() - start
    })


# ============ 性能测试 ============

def benchmark():
    """性能测试"""
    print("="*60)
    print("性能测试")
    print("="*60)
    
    # 使用随机图片测试
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    images = [test_img.copy() for _ in range(32)]
    
    # 单张测试
    print("\n[单张检测]")
    times = []
    for img in images:
        start = time.time()
        detector.detect(img)
        times.append(time.time() - start)
    print(f"  平均: {np.mean(times)*1000:.1f}ms, FPS: {1/np.mean(times):.1f}")
    
    # 批量测试
    print("\n[批量检测]")
    for batch_size in [4, 8, 16, 32]:
        start = time.time()
        detector.detect_batch(images[:batch_size])
        elapsed = time.time() - start
        fps = batch_size / elapsed
        print(f"  batch={batch_size}: {elapsed:.2f}s, FPS: {fps:.1f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/yolov8/train/weights/best.pt')
    parser.add_argument('--conf', type=float, default=0.2)
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    
    # 初始化检测器
    detector = PCBBatchDetector(args.model, args.conf, args.imgsz)
    
    if args.benchmark:
        benchmark()
    else:
        print(f"\nAPI服务启动: http://0.0.0.0:{args.port}")
        print("  GET  /health         - 健康检查")
        print("  POST /detect         - 单张检测 (form-data: image)")
        print("  POST /detect_batch   - 批量检测 (form-data: images)")
        print("  POST /detect_base64  - Base64检测 (JSON: {image: 'base64...'})")
        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
