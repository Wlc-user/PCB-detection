"""
API测试脚本
用于测试PCB缺陷检测API接口
"""

import requests
import base64
import json
import time

API_URL = "http://localhost:8000"


def test_health():
    """测试健康检查"""
    print("=" * 50)
    print("1. 测试健康检查 /health")
    print("=" * 50)
    resp = requests.get(f"{API_URL}/health")
    print(f"状态码: {resp.status_code}")
    print(f"响应: {json.dumps(resp.json(), indent=2)}")
    print()


def test_model_info():
    """测试模型信息"""
    print("=" * 50)
    print("2. 测试模型信息 /model/info")
    print("=" * 50)
    resp = requests.get(f"{API_URL}/model/info")
    print(f"状态码: {resp.status_code}")
    print(f"响应: {json.dumps(resp.json(), indent=2)}")
    print()


def test_detect_file():
    """测试文件上传检测"""
    print("=" * 50)
    print("3. 测试图片文件检测 /detect")
    print("=" * 50)

    img_path = "yolo_pcb_dataset/images/test/01_missing_hole_02.jpg"

    with open(img_path, "rb") as f:
        files = {"file": f}
        data = {"conf_threshold": 0.25, "return_image": True}

        resp = requests.post(f"{API_URL}/detect", files=files, data=data)

    print(f"状态码: {resp.status_code}")
    result = resp.json()
    print(f"检测到 {len(result.get('detections', []))} 个缺陷:")
    for d in result.get("detections", []):
        print(f"  - {d['class']}: 置信度={d['confidence']:.2f}")
    print(f"推理时间: {result.get('inference_time', 0):.3f}s")
    print()


def test_detect_base64():
    """测试Base64图片检测"""
    print("=" * 50)
    print("4. 测试Base64图片检测 /detect/base64")
    print("=" * 50)

    # 读取图片转Base64
    with open("yolo_pcb_dataset/images/test/01_missing_hole_02.jpg", "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()

    resp = requests.post(
        f"{API_URL}/detect/base64",
        json={
            "image_data": img_base64,
            "conf_threshold": 0.25
        }
    )

    print(f"状态码: {resp.status_code}")
    result = resp.json()
    print(f"检测到 {len(result.get('detections', []))} 个缺陷")
    print()


def test_batch():
    """测试批量检测"""
    print("=" * 50)
    print("5. 测试批量检测 /detect/batch")
    print("=" * 50)

    resp = requests.post(
        f"{API_URL}/detect/batch",
        json={
            "image_dir": "yolo_pcb_dataset/images/test",
            "conf_threshold": 0.25,
            "save_results": True
        }
    )

    print(f"状态码: {resp.status_code}")
    result = resp.json()
    print(f"响应: {json.dumps(result, indent=2)}")
    print()


def test_stats():
    """测试统计信息"""
    print("=" * 50)
    print("6. 测试统计信息 /stats")
    print("=" * 50)

    resp = requests.get(f"{API_URL}/stats")
    print(f"状态码: {resp.status_code}")
    print(f"响应: {json.dumps(resp.json(), indent=2)}")
    print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("   PCB缺陷检测API - 测试脚本")
    print("=" * 50 + "\n")

    try:
        test_health()
        test_model_info()
        test_detect_file()
        test_detect_base64()
        test_stats()

        print("=" * 50)
        print("所有测试完成!")
        print("=" * 50)

    except Exception as e:
        print(f"测试失败: {e}")
        print("请确保API服务正在运行: python defect_detection_api.py")


if __name__ == "__main__":
    main()
