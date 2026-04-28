"""
视频流 + LLM + MCP 客户端

功能:
    - WebSocket 连接服务端
    - 接收视频流并显示
    - 接收 MCP 消息并展示
    - 发送音频进行语音对话

使用方式:
    python video_llm_mcp_client.py --server ws://localhost:8765
"""

import os
import sys
import time
import asyncio
import json
import base64
import queue
import threading
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import websockets
from websockets.client import WebSocketClientProtocol

# ============ 日志配置 ============
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('VideoLLMClient')


# ============ MCP 消息解析 ============
class MCPMessageParser:
    """MCP 消息解析器"""
    
    TYPE_LABELS = {
        'detection_report': '🔍 检测报告',
        'chat_response': '💬 聊天回复',
        'audio_transcript': '🎤 语音转写',
        'stream_frame': '📹 视频帧',
        'system_status': '📊 系统状态'
    }
    
    @staticmethod
    def parse(raw: str) -> Dict:
        """解析 MCP 消息"""
        try:
            return json.loads(raw)
        except:
            return {'raw': raw}
    
    @staticmethod
    def format(msg: Dict) -> str:
        """格式化消息显示"""
        msg_type = msg.get('msg_type', 'unknown')
        label = MCPMessageParser.TYPE_LABELS.get(msg_type, msg_type)
        content = msg.get('content', '')
        
        lines = [
            f"\n{'='*60}",
            f"{label}",
            f"{'='*60}",
            content,
            ""
        ]
        
        # 添加元数据
        metadata = msg.get('metadata', {})
        if metadata:
            if 'count' in metadata:
                lines.append(f"检测数量: {metadata['count']}")
            if 'detections' in metadata:
                lines.append("\n检测详情:")
                for det in metadata['detections'][:5]:
                    lines.append(f"  • {det.get('class_name')}: {det.get('confidence', 0):.2%}")
        
        return "\n".join(lines)


# ============ 视频显示窗口 ============
class VideoWindow:
    """视频显示窗口"""
    
    def __init__(self, title: str = 'Video LLM MCP Client'):
        self.title = title
        self.frame = None
        self.running = False
    
    def show(self, frame: np.ndarray):
        """显示帧"""
        if frame is not None:
            self.frame = frame
            cv2.imshow(self.title, frame)
    
    def wait_key(self, delay: int = 1) -> int:
        """等待按键"""
        return cv2.waitKey(delay) & 0xFF
    
    def close(self):
        """关闭窗口"""
        cv2.destroyWindow(self.title)


# ============ 音频录制器 ============
class AudioRecorder:
    """音频录制器 (简化版)"""
    
    def __init__(self):
        self.recording = False
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
    
    def start(self):
        """开始录制"""
        self.recording = True
        logger.info("音频录制开始 (模拟)")
    
    def stop(self) -> bytes:
        """停止录制并返回音频数据"""
        self.recording = False
        logger.info("音频录制停止")
        return b'audio_data_placeholder'
    
    def is_recording(self) -> bool:
        return self.recording


# ============ 主客户端 ============
class VideoLLMClient:
    """视频流 + LLM + MCP 客户端"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.ws: WebSocketClientProtocol = None
        self.running = False
        self.video_window = VideoWindow()
        self.audio_recorder = AudioRecorder()
        self.message_queue = queue.Queue()
        self.stats = {
            'connected': False,
            'frames_received': 0,
            'messages_received': 0,
            'start_time': time.time()
        }
    
    async def connect(self) -> bool:
        """连接服务器"""
        try:
            logger.info(f"连接服务器: {self.server_url}")
            self.ws = await websockets.connect(self.server_url)
            self.running = True
            self.stats['connected'] = True
            logger.info("✓ 连接成功")
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False
    
    async def disconnect(self):
        """断开连接"""
        self.running = False
        if self.ws:
            await self.ws.close()
        self.stats['connected'] = False
        logger.info("已断开连接")
    
    async def send_image(self, image: np.ndarray, frame_id: str = None):
        """发送图像"""
        if not self.ws or not self.stats['connected']:
            return
        
        try:
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_base64 = base64.b64encode(buffer).decode()
            
            message = {
                'type': 'image',
                'frame_id': frame_id or f"client_{time.time()}",
                'timestamp': time.time(),
                'data': img_base64
            }
            
            await self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"发送图像失败: {e}")
    
    async def send_audio(self, audio_data: bytes):
        """发送音频"""
        if not self.ws or not self.stats['connected']:
            return
        
        try:
            audio_base64 = base64.b64encode(audio_data).decode()
            
            message = {
                'type': 'audio',
                'timestamp': time.time(),
                'data': audio_base64,
                'sample_rate': self.audio_recorder.sample_rate
            }
            
            await self.ws.send(json.dumps(message))
            logger.info("✓ 音频已发送")
        except Exception as e:
            logger.error(f"发送音频失败: {e}")
    
    async def receive_messages(self):
        """接收消息循环"""
        while self.running:
            try:
                message = await asyncio.wait_for(self.ws.recv(), timeout=1.0)
                
                if isinstance(message, str):
                    msg_data = MCPMessageParser.parse(message)
                    self.message_queue.put(msg_data)
                    self.stats['messages_received'] += 1
                    await self._handle_message(msg_data)
                
                elif isinstance(message, bytes):
                    np_arr = np.frombuffer(message, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self.video_window.show(frame)
                        self.stats['frames_received'] += 1
                
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.warning("服务器断开")
                break
            except Exception as e:
                logger.error(f"接收消息错误: {e}")
    
    async def _handle_message(self, msg: Dict):
        """处理接收到的消息"""
        msg_type = msg.get('msg_type', '')
        
        if msg_type == 'detection_report':
            print(MCPMessageParser.format(msg))
        elif msg_type == 'chat_response':
            print(MCPMessageParser.format(msg))
        elif msg_type == 'audio_transcript':
            print(MCPMessageParser.format(msg))
        elif msg_type == 'annotated_image':
            img_data = msg.get('image', '')
            if img_data:
                img_bytes = base64.b64decode(img_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    self.video_window.show(frame)
        elif msg_type == 'connected':
            logger.info(f"✓ 服务器已连接: {msg.get('client_id')}")
        elif msg_type == 'error':
            logger.error(f"服务器错误: {msg.get('message')}")
    
    async def run_loop(self):
        """主循环"""
        if not await self.connect():
            return
        
        recv_task = asyncio.create_task(self.receive_messages())
        
        while self.running:
            key = self.video_window.wait_key(1)
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                if self.video_window.frame is not None:
                    await self.send_image(self.video_window.frame)
                    logger.info("✓ 截图已发送")
        
        await self.disconnect()
        recv_task.cancel()
        self.video_window.close()
    
    def print_stats(self):
        """打印统计"""
        elapsed = time.time() - self.stats['start_time']
        logger.info(f"\n{'='*40}")
        logger.info("客户端统计:")
        logger.info(f"  运行时间: {elapsed:.1f}秒")
        logger.info(f"  连接状态: {'已连接' if self.stats['connected'] else '未连接'}")
        logger.info(f"  接收帧数: {self.stats['frames_received']}")
        logger.info(f"  接收消息: {self.stats['messages_received']}")
        logger.info(f"{'='*40}")


# ============ 主函数 ============
async def main():
    parser = argparse.ArgumentParser(description='Video LLM MCP 客户端')
    
    parser.add_argument('--server', type=str, 
                        default='ws://localhost:8765',
                        help='服务器地址')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='本地摄像头ID')
    parser.add_argument('--image', type=str, default=None,
                        help='发送单张图片')
    parser.add_argument('--no-display', action='store_true',
                        help='不显示视频窗口')
    
    args = parser.parse_args()
    
    client = VideoLLMClient(args.server)
    
    try:
        if args.image:
            if await client.connect():
                image = cv2.imread(args.image)
                if image is not None:
                    await client.send_image(image)
                    logger.info(f"✓ 图片已发送: {args.image}")
                    await asyncio.sleep(5)
                await client.disconnect()
        
        elif args.camera_id >= 0:
            logger.info(f"启动本地摄像头 (ID: {args.camera_id})")
            
            if await client.connect():
                cap = cv2.VideoCapture(args.camera_id)
                if not cap.isOpened():
                    logger.error(f"无法打开摄像头: {args.camera_id}")
                    await client.disconnect()
                    return
                
                logger.info("按 'q' 退出, 空格发送帧")
                
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        client.video_window.show(frame)
                        key = cv2.waitKey(1) & 0xFF
                        
                        if key == ord('q'):
                            break
                        elif key == ord(' '):
                            await client.send_image(frame)
                        
                        try:
                            msg = client.message_queue.get_nowait()
                            print(MCPMessageParser.format(msg))
                        except queue.Empty:
                            pass
                
                finally:
                    cap.release()
                    await client.disconnect()
            else:
                logger.error("无法连接到服务器")
    
    except KeyboardInterrupt:
        logger.info("收到中断信号")
    finally:
        client.print_stats()


if __name__ == '__main__':
    asyncio.run(main())
