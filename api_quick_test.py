"""快速测试API检测"""
import requests

# 测试1: 低置信度检测
print("测试低置信度检测 (conf=0.15)...")
with open('yolo_pcb_dataset/images/test/01_missing_hole_02.jpg', 'rb') as f:
    resp = requests.post(
        'http://localhost:8000/detect',
        files={'file': f},
        data={'conf_threshold': 0.15, 'return_image': 'true'}
    )

result = resp.json()
print(f'状态码: {resp.status_code}')
print(f'检测到 {len(result.get("detections", []))} 个缺陷:')
for d in result.get('detections', []):
    print(f"  - {d['class']}: 置信度={d['confidence']:.2f}")
print(f'推理时间: {result.get("inference_time", 0):.3f}s')

# 测试2: 批量测试集
print('\n测试批量检测...')
resp = requests.post(
    'http://localhost:8000/detect/batch',
    json={
        'image_dir': 'yolo_pcb_dataset/images/test',
        'conf_threshold': 0.15
    }
)
result = resp.json()
print(f'状态码: {resp.status_code}')
print(f'处理图片: {result.get("total_images", 0)} 张')
print(f'总缺陷数: {result.get("total_defects", 0)} 个')

print('\nAPI测试完成!')
