"""测试REST API"""
import requests
import json
import base64
import numpy as np
import cv2

print("="*50)
print("REST API 测试")
print("="*50)

# 测试健康检查
print("\n1. 健康检查...")
try:
    r = requests.get('http://localhost:8080/health', timeout=5)
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
except Exception as e:
    print(f"   Error: {e}")

# 测试检测API
print("\n2. 检测API...")
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 0), -1)
cv2.putText(img, 'TEST', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
_, buffer = cv2.imencode('.jpg', img)
img_base64 = base64.b64encode(buffer).decode()

payload = {
    'image_base64': img_base64,
    'camera_id': 'test_cam_01'
}

try:
    r = requests.post('http://localhost:8080/api/detect', json=payload, timeout=30)
    print(f"   Status: {r.status_code}")
    result = r.json()
    print(f"   Detections: {len(result.get('detections', []))}")
    print(f"   Scene type: {result.get('scene_type')}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*50)
print("API测试完成!")
