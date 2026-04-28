"""HTTP 测试脚本"""
import base64
import json
import requests

print('='*60)
print('HTTP 检测测试')
print('='*60)

# 读取测试图像
with open('yolo_pcb_dataset/images/test/01_missing_hole_02.jpg', 'rb') as f:
    img_data = base64.b64encode(f.read()).decode('utf-8')

print(f'图片 Base64 长度: {len(img_data)} 字符')

# 发送请求
response = requests.post('http://localhost:8080/detect', 
                        json={'data': img_data, 'confidence': 0.25},
                        timeout=120)
                        
print(f'HTTP Status: {response.status_code}')
if response.status_code == 200:
    result = response.json()
    print()
    print('检测结果:')
    print(f'  成功: {result.get("success")}')
    print(f'  检测数量: {result.get("count", len(result.get("detections", [])))}')
    print(f'  处理时间: {result.get("processing_time_ms", 0):.1f}ms')
    print()
    for det in result.get('detections', [])[:5]:
        cls = det.get('class', 'unknown')
        conf = det.get('confidence', 0)
        print(f'  - {cls}: {conf:.2%}')
