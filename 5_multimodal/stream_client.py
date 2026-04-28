"""
WebSocket 客户端 - 发送图像流
用于测试实时流处理服务

使用方式:
    python stream_client.py --image test.jpg
    python stream_client.py --folder images/
    python stream_client.py --camera 0 --fps 10
"""

import asyncio
import websockets
import json
import base64
import time
import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import aiofiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamClient:
    """流发送客户端"""
    
    def __init__(self, uri: str = "ws://localhost:8765"):
        self.uri = uri
        self.ws = None
        self.frame_id = 0
    
    async def connect(self):
        """连接服务器"""
        try:
            self.ws = await websockets.connect(self.uri)
            logger.info(f"✓ 已连接: {self.uri}")
            
            # 接收欢迎消息
            welcome = await self.ws.recv()
            logger.info(f"服务器响应: {welcome}")
            
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False
    
    async def send_image(self, image: np.ndarray, metadata: dict = None) -> dict:
        """发送图像"""
        self.frame_id += 1
        
        # 编码为Base64
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode()
        
        message = {
            'type': 'image',
            'frame_id': f"client_{self.frame_id}",
            'timestamp': time.time(),
            'data': img_base64,
            'metadata': metadata or {}
        }
        
        await self.ws.send(json.dumps(message))
        
        # 等待结果
        try:
            response = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
            return json.loads(response)
        except asyncio.TimeoutError:
            return {'error': 'timeout'}
    
    async def send_binary(self, image: np.ndarray) -> dict:
        """发送二进制图像"""
        self.frame_id += 1
        
        # 编码为JPEG
        _, buffer = cv2.imencode('.jpg', image)
        
        # JSON元数据 + 二进制图像
        metadata = json.dumps({
            'type': 'image',
            'frame_id': f"client_{self.frame_id}",
            'timestamp': time.time(),
        })
        
        # 发送元数据
        await self.ws.send(metadata)
        
        # 发送二进制
        await self.ws.send(buffer.tobytes())
        
        return {'status': 'sent'}
    
    async def receive_results(self):
        """接收结果流"""
        try:
            while True:
                message = await self.ws.recv()
                data = json.loads(message)
                
                if data.get('type') == 'result':
                    yield data
                elif data.get('type') == 'annotated_image':
                    yield data
                elif data.get('type') == 'pong':
                    pass  # 心跳响应
                else:
                    logger.info(f"收到: {data.get('type')}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("连接已断开")
    
    async def update_config(self, **kwargs):
        """更新配置"""
        message = {'type': 'config', **kwargs}
        await self.ws.send(json.dumps(message))
        response = await self.ws.recv()
        return json.loads(response)
    
    async def get_stats(self):
        """获取统计"""
        await self.ws.send(json.dumps({'type': 'stats'}))
        response = await self.ws.recv()
        return json.loads(response)
    
    async def close(self):
        """关闭连接"""
        if self.ws:
            await self.ws.close()
            logger.info("连接已关闭")


async def stream_images(client: StreamClient, image_paths: list):
    """流式发送多张图像"""
    for i, path in enumerate(image_paths):
        image = cv2.imread(str(path))
        if image is None:
            logger.warning(f"无法读取: {path}")
            continue
        
        logger.info(f"发送 [{i+1}/{len(image_paths)}]: {Path(path).name}")
        
        result = await client.send_image(image, {'filename': str(path)})
        
        if 'detections' in result:
            count = len(result.get('detections', []))
            logger.info(f"  → 检测到 {count} 个目标")
            for det in result.get('detections', [])[:3]:
                logger.info(f"    - {det.get('class_name')}: {det.get('confidence'):.3f}")
        
        await asyncio.sleep(0.1)  # 控制发送速率


async def stream_camera(client: StreamClient, camera_id: int = 0, fps: int = 10):
    """从摄像头流式发送"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"无法打开摄像头: {camera_id}")
        return
    
    interval = 1 / fps
    
    logger.info(f"开始摄像头捕获 (FPS: {fps})")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = await client.send_image(frame)
            
            if 'detections' in result:
                count = len(result.get('detections', []))
                print(f"\rFPS: {fps} | 检测: {count}", end='')
            
            await asyncio.sleep(interval)
    
    except KeyboardInterrupt:
        logger.info("停止捕获")
    finally:
        cap.release()


async def async_main(args):
    """异步主函数"""
    client = StreamClient(args.uri)
    
    if not await client.connect():
        return
    
    try:
        if args.image:
            # 单张图像
            image = cv2.imread(args.image)
            if image is None:
                logger.error(f"无法读取图像: {args.image}")
                return
            
            result = await client.send_image(image)
            logger.info(f"结果: {result}")
        
        elif args.folder:
            # 文件夹图像
            image_paths = list(Path(args.folder).glob('*.jpg'))
            image_paths += list(Path(args.folder).glob('*.png'))
            logger.info(f"找到 {len(image_paths)} 张图像")
            
            await stream_images(client, image_paths)
        
        elif args.camera is not None:
            # 摄像头
            await stream_camera(client, args.camera, args.fps)
        
        else:
            logger.info("使用 --image 或 --folder 或 --camera 指定输入")
    
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser(description='WebSocket流发送客户端')
    
    # 连接配置
    parser.add_argument('--uri', type=str, default='ws://localhost:8765',
                       help='WebSocket服务器地址')
    
    # 输入模式 (三选一)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--image', type=str, help='单张图像路径')
    input_group.add_argument('--folder', type=str, help='图像文件夹')
    input_group.add_argument('--camera', type=int, const=0, nargs='?', help='摄像头ID')
    
    # 摄像头配置
    parser.add_argument('--fps', type=int, default=10, help='摄像头FPS')
    
    # 测试模式
    parser.add_argument('--test', action='store_true', help='发送测试图像')
    
    args = parser.parse_args()
    
    # 测试模式
    if args.test:
        # 创建测试图像
        logger.info("创建测试图像...")
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_img, "Test Image", (200, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 保存临时文件
        temp_path = 'temp_test.jpg'
        cv2.imwrite(temp_path, test_img)
        args.image = temp_path
    
    # 运行
    asyncio.run(async_main(args))


if __name__ == '__main__':
    main()
