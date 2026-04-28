"""WebSocket 测试"""
import asyncio
import base64
import json
import websockets

async def test_websocket():
    uri = "ws://localhost:8080/ws"
    print("="*60)
    print(f"WebSocket 测试: {uri}")
    print("="*60)
    
    # 读取测试图像
    with open('yolo_pcb_dataset/images/test/01_missing_hole_02.jpg', 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    print(f"图片 Base64 长度: {len(img_data)} 字符")
    
    try:
        async with websockets.connect(uri) as ws:
            print("连接成功!")
            
            # 发送图像
            message = {
                "type": "image",
                "data": img_data,
                "confidence": 0.25
            }
            
            print("发送图像...")
            await ws.send(json.dumps(message))
            
            # 接收结果
            print("等待结果...")
            response = await ws.recv()
            result = json.loads(response)
            
            print()
            print("检测结果:")
            print(f"  成功: {result.get('success')}")
            print(f"  检测数量: {result.get('count', len(result.get('detections', [])))}")
            print()
            for det in result.get('detections', [])[:5]:
                cls = det.get('class', 'unknown')
                conf = det.get('confidence', 0)
                print(f"  - {cls}: {conf:.2%}")
            
    except Exception as e:
        print(f"错误: {e}")

if __name__ == '__main__':
    asyncio.run(test_websocket())
