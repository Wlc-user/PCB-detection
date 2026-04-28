"""
音视频流模块 - Audio/Video Streaming Module

功能:
    - 音视频同步采集
    - RTMP/WebRTC 流推送
    - 实时音频播放
    - 音视频编解码

使用方式:
    python av_streaming.py --mode publish --url rtmp://localhost/live
    python av_streaming.py --mode subscribe --url rtmp://server/live
"""

import os
import sys
import time
import asyncio
import threading
import queue
import logging
import argparse
import base64
import json
from pathlib import Path
from datetime import datetime
import wave
import struct
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass

import cv2
import numpy as np

# ============ 日志配置 ============
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('AVStreaming')


# ============ 音频编解码器 ============
class AudioCodec:
    """音频编解码器"""
    
    # 采样率
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = 1024
    
    # 编码格式
    SUPPORTED_FORMATS = ['pcm', 'opus', 'aac', 'mp3', 'wav']
    
    @staticmethod
    def pcm_to_wav(pcm_data: bytes, output_file: str = None) -> str:
        """PCM 转 WAV"""
        if output_file is None:
            output_file = f"output/audio_{int(time.time())}.wav"
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(AudioCodec.CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(AudioCodec.SAMPLE_RATE)
            wf.writeframes(pcm_data)
        
        return output_file
    
    @staticmethod
    def resample_audio(audio_data: bytes, orig_rate: int, target_rate: int) -> bytes:
        """重采样音频"""
        try:
            import librosa
            audio = np.frombuffer(audio_data, dtype=np.int16)
            resampled = librosa.resample(audio, orig_sr=orig_rate, target_sr=target_rate)
            return resampled.astype(np.int16).tobytes()
        except ImportError:
            logger.warning("librosa 未安装，使用原始音频")
            return audio_data
    
    @staticmethod
    def normalize_audio(audio_data: bytes) -> bytes:
        """音频归一化"""
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9
        return (audio * 32768).astype(np.int16).tobytes()


# ============ 视频编解码器 ============
class VideoCodec:
    """视频编解码器"""
    
    # 编解码器
    CODECS = {
        'h264': cv2.VideoWriter_fourcc(*'avc1'),
        'h265': cv2.VideoWriter_fourcc(*'hev1'),
        'vp8': cv2.VideoWriter_fourcc(*'VP80'),
        'vp9': cv2.VideoWriter_fourcc(*'VP90'),
        'mjpeg': cv2.VideoWriter_fourcc(*'MJPG'),
    }
    
    # 预设质量
    QUALITY_PRESETS = {
        'low': {'quality': 50, 'fps': 15},
        'medium': {'quality': 70, 'fps': 24},
        'high': {'quality': 85, 'fps': 30},
    }
    
    @staticmethod
    def encode_frame(frame: np.ndarray, quality: int = 70, format: str = '.jpg') -> bytes:
        """编码视频帧"""
        if format == '.jpg' or format == '.jpeg':
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
        elif format == '.png':
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9 - quality // 12]
            _, buffer = cv2.imencode('.png', frame, encode_param)
        else:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        return buffer.tobytes()
    
    @staticmethod
    def decode_frame(data: bytes) -> Optional[np.ndarray]:
        """解码视频帧"""
        nparr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    @staticmethod
    def create_video_writer(output_file: str, fps: int = 30, 
                           codec: str = 'h264', frame_size: tuple = None) -> cv2.VideoWriter:
        """创建视频写入器"""
        fourcc = VideoCodec.CODECS.get(codec, VideoCodec.CODECS['h264'])
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        writer = cv2.VideoWriter(
            output_file, fourcc, fps,
            frame_size or (640, 480)
        )
        
        if not writer.isOpened():
            logger.error(f"无法创建视频写入器: {output_file}")
            return None
        
        logger.info(f"✓ 视频写入器创建: {output_file} ({codec}, {fps}fps)")
        return writer


# ============ 音频播放器 ============
class AudioPlayer:
    """音频播放器"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._pyaudio = None
        self._stream = None
        self._playing = False
        self._init_player()
    
    def _init_player(self):
        """初始化播放器"""
        try:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()
            logger.info("✓ 音频播放器初始化")
        except ImportError:
            logger.warning("PyAudio 未安装，音频播放不可用")
    
    def play(self, audio_data: bytes):
        """播放音频"""
        if not self._pyaudio:
            logger.info(f"[播放] {len(audio_data)} bytes")
            return
        
        try:
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True
            )
            
            self._stream.write(audio_data)
            self._stream.stop_stream()
            self._stream.close()
            
        except Exception as e:
            logger.error(f"音频播放失败: {e}")
    
    def play_file(self, audio_file: str):
        """播放音频文件"""
        try:
            if audio_file.endswith('.mp3'):
                self._play_mp3(audio_file)
            elif audio_file.endswith('.wav'):
                self._play_wav(audio_file)
            elif audio_file.endswith('.mp4'):
                self._play_mp4(audio_file)
        except Exception as e:
            logger.error(f"播放文件失败: {e}")
    
    def _play_wav(self, wav_file: str):
        """播放 WAV"""
        if not self._pyaudio:
            logger.info(f"[WAV] {wav_file}")
            return
        
        with wave.open(wav_file, 'rb') as wf:
            data = wf.readframes(wf.getnframes())
            stream = self._pyaudio.open(
                format=self._pyaudio.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            stream.write(data)
            stream.close()
    
    def _play_mp3(self, mp3_file: str):
        """播放 MP3"""
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(mp3_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except ImportError:
            logger.warning("pygame 未安装，使用系统播放")
            if sys.platform == 'win32':
                os.system(f'start "" "{mp3_file}"')
            else:
                os.system(f'afplay "{mp3_file}"')
    
    def _play_mp4(self, mp4_file: str):
        """播放 MP4 视频"""
        if sys.platform == 'win32':
            os.system(f'start "" "{mp4_file}"')
        else:
            os.system(f'xdg-open "{mp4_file}"')


# ============ RTMP 流推送器 ============
class RTMPStreamer:
    """RTMP 流推送器"""
    
    def __init__(self, url: str, fps: int = 24, codec: str = 'h264'):
        self.url = url
        self.fps = fps
        self.codec = codec
        self.writer = None
        self._connected = False
    
    def connect(self, frame_size: tuple = (640, 480)) -> bool:
        """连接 RTMP 服务器"""
        try:
            fourcc = VideoCodec.CODECS.get(self.codec, VideoCodec.CODECS['h264'])
            
            # 使用 OpenCV 的 VideoWriter 推流
            pipeline = f'appsrc ! video/x-raw,format=BGR ! videoconvert ! video/x-h264,profile=baseline ! flvmux ! rtmpsink location="{self.url}"'
            
            # 简化版本
            self.writer = cv2.VideoWriter(
                self.url,
                cv2.CAP_FFMPEG,
                fourcc,
                self.fps,
                frame_size
            )
            
            if self.writer.isOpened():
                self._connected = True
                logger.info(f"✓ RTMP 连接成功: {self.url}")
                return True
            else:
                logger.error("RTMP 连接失败")
                return False
                
        except Exception as e:
            logger.error(f"RTMP 连接异常: {e}")
            return False
    
    def push_frame(self, frame: np.ndarray):
        """推送视频帧"""
        if self._connected and self.writer:
            self.writer.write(frame)
    
    def push_audio_frame(self, audio_data: bytes):
        """推送音频帧 (需要 GStreamer 支持)"""
        # 需要 GStreamer 或 FFmpeg wrapper
        pass
    
    def close(self):
        """关闭连接"""
        if self.writer:
            self.writer.release()
        self._connected = False
        logger.info("RTMP 流已关闭")


# ============ 音视频同步器 ============
@dataclass
class AVFrame:
    """音视频帧"""
    timestamp: float
    video_frame: Optional[np.ndarray] = None
    audio_data: Optional[bytes] = None
    frame_id: int = 0


class AVSynchronizer:
    """音视频同步器"""
    
    def __init__(self, video_fps: int = 24, audio_sample_rate: int = 16000):
        self.video_fps = video_fps
        self.audio_sample_rate = audio_sample_rate
        self.video_interval = 1.0 / video_fps
        self.audio_samples_per_frame = audio_sample_rate // video_fps
        
        self.frame_queue = queue.Queue(maxsize=30)
        self.last_video_time = 0
        self.last_audio_time = 0
    
    def add_video_frame(self, frame: np.ndarray, timestamp: float = None):
        """添加视频帧"""
        if timestamp is None:
            timestamp = time.time()
        
        self.last_video_time = timestamp
        
        av_frame = AVFrame(
            timestamp=timestamp,
            video_frame=frame,
            frame_id=int(timestamp * self.video_fps)
        )
        
        try:
            self.frame_queue.put_nowait(av_frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(av_frame)
            except:
                pass
    
    def add_audio_data(self, audio_data: bytes, timestamp: float = None):
        """添加音频数据"""
        if timestamp is None:
            timestamp = time.time()
        
        self.last_audio_time = timestamp
        
        av_frame = AVFrame(
            timestamp=timestamp,
            audio_data=audio_data
        )
        
        try:
            self.frame_queue.put_nowait(av_frame)
        except queue.Full:
            pass
    
    def get_next_frame(self, timeout: float = 1.0) -> Optional[AVFrame]:
        """获取下一帧"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def sync_to_audio(self, av_frame: AVFrame, audio_pos: float) -> bool:
        """音频同步"""
        if av_frame.video_frame is None:
            return False
        
        expected_video_pos = audio_pos
        current_video_pos = av_frame.timestamp
        
        # 允许 50ms 的误差
        return abs(expected_video_pos - current_video_pos) < 0.05


# ============ 音频录制器 ============
class AudioRecorder:
    """音频录制器"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.frames = []
        self.thread = None
        self._pyaudio = None
        self._stream = None
    
    def start(self) -> bool:
        """开始录制"""
        try:
            import pyaudio
            
            self._pyaudio = pyaudio.PyAudio()
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024
            )
            
            self.recording = True
            self.frames = []
            
            self.thread = threading.Thread(target=self._record_loop, daemon=True)
            self.thread.start()
            
            logger.info(f"✓ 录音开始 ({self.sample_rate}Hz, {self.channels}ch)")
            return True
            
        except Exception as e:
            logger.error(f"录音启动失败: {e}")
            return False
    
    def _record_loop(self):
        """录制循环"""
        while self.recording:
            try:
                data = self._stream.read(1024, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                logger.error(f"录音错误: {e}")
                break
    
    def stop(self) -> bytes:
        """停止录制"""
        self.recording = False
        
        if self.thread:
            self.thread.join(timeout=2)
        
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        
        if self._pyaudio:
            self._pyaudio.terminate()
        
        audio_data = b''.join(self.frames)
        logger.info(f"✓ 录音结束 ({len(audio_data)} bytes)")
        return audio_data
    
    def save(self, output_file: str) -> str:
        """保存录音"""
        audio_data = b''.join(self.frames)
        
        if output_file.endswith('.wav'):
            AudioCodec.pcm_to_wav(audio_data, output_file)
        else:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'wb') as f:
                f.write(audio_data)
        
        return output_file


# ============ 视频捕获器 ============
class VideoCapture:
    """视频捕获器"""
    
    def __init__(self, source: int = 0, fps: int = 30):
        self.source = source
        self.fps = fps
        self.capture = None
        self.running = False
        self.frame_callbacks = []
        self.thread = None
    
    def open(self, width: int = None, height: int = None) -> bool:
        """打开视频源"""
        try:
            if isinstance(self.source, str):
                self.capture = cv2.VideoCapture(self.source)
            else:
                self.capture = cv2.VideoCapture(int(self.source))
            
            if not self.capture.isOpened():
                logger.error(f"无法打开视频源: {self.source}")
                return False
            
            if width and height:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"✓ 视频源已打开 ({actual_width}x{actual_height} @ {actual_fps}fps)")
            return True
            
        except Exception as e:
            logger.error(f"视频源打开失败: {e}")
            return False
    
    def register_callback(self, callback: Callable):
        """注册帧回调"""
        self.frame_callbacks.append(callback)
    
    def start(self):
        """开始捕获"""
        if not self.capture or not self.capture.isOpened():
            logger.error("视频源未打开")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        logger.info("视频捕获开始")
    
    def _capture_loop(self):
        """捕获循环"""
        frame_interval = 1.0 / self.fps
        last_time = time.time()
        
        while self.running:
            ret, frame = self.capture.read()
            
            if not ret:
                continue
            
            # 节流
            current_time = time.time()
            if current_time - last_time < frame_interval:
                continue
            last_time = current_time
            
            # 调用回调
            for callback in self.frame_callbacks:
                try:
                    callback(frame, current_time)
                except Exception as e:
                    logger.error(f"回调错误: {e}")
    
    def stop(self):
        """停止捕获"""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=2)
        
        if self.capture:
            self.capture.release()
        
        logger.info("视频捕获停止")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """读取单帧"""
        if not self.capture:
            return None
        
        ret, frame = self.capture.read()
        return frame if ret else None


# ============ WebRTC 流 ============
class WebRTCStreamer:
    """WebRTC 流推送器"""
    
    def __init__(self, signaling_url: str = 'ws://localhost:8080'):
        self.signaling_url = signaling_url
        self.peer_connection = None
        self.data_channel = None
        self._connected = False
    
    async def connect(self) -> bool:
        """连接 WebRTC 信令服务器"""
        try:
            import websockets
            
            async with websockets.connect(self.signaling_url) as ws:
                logger.info(f"✓ WebRTC 信令连接: {self.signaling_url}")
                
                # 简化的信令流程
                # 实际需要完整的 ICE/STUN/TURN 配置
                
                self._connected = True
                return True
                
        except Exception as e:
            logger.error(f"WebRTC 连接失败: {e}")
            return False
    
    async def push_frame(self, frame: np.ndarray):
        """推送视频帧"""
        if not self._connected:
            return
        
        # 编码帧
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_data = base64.b64encode(buffer).decode()
        
        # 通过 DataChannel 发送
        # 实际需要 RTCDataChannel
        pass
    
    async def close(self):
        """关闭连接"""
        self._connected = False
        if self.peer_connection:
            await self.peer_connection.close()


# ============ 演示函数 ============
async def demo_streaming():
    """流媒体演示"""
    print("=" * 60)
    print(" 音视频流媒体演示")
    print("=" * 60)
    
    # 1. 音频录制演示
    print("\n[1] 音频录制...")
    recorder = AudioRecorder()
    if recorder.start():
        await asyncio.sleep(3)  # 录制3秒
        audio_data = recorder.stop()
        recorder.save("output/demo_recording.wav")
        print(f"   录制完成: {len(audio_data)} bytes")
    
    # 2. 音频播放演示
    print("\n[2] 音频播放...")
    player = AudioPlayer()
    if os.path.exists("output/deepseek_response.mp3"):
        player.play_file("output/deepseek_response.mp3")
        print("   播放完成")
    
    # 3. 视频编码演示
    print("\n[3] 视频编码...")
    frame = cv2.imread("yolo_pcb_dataset/images/test/01_missing_hole_02.jpg")
    if frame is not None:
        # 编码为不同格式
        for quality in [50, 70, 90]:
            jpg_data = VideoCodec.encode_frame(frame, quality=quality, format='.jpg')
            print(f"   JPEG Q{quality}: {len(jpg_data)} bytes")
    
    # 4. 视频流捕获演示
    print("\n[4] 视频流捕获...")
    capture = VideoCapture(source=0, fps=10)
    frames_captured = []
    
    def on_frame(f, t):
        frames_captured.append(f)
        if len(frames_captured) >= 5:
            capture.running = False
    
    capture.register_callback(on_frame)
    
    if capture.open(width=640, height=480):
        capture.start()
        await asyncio.sleep(2)
        capture.stop()
        print(f"   捕获帧数: {len(frames_captured)}")
    
    # 5. 音视频同步演示
    print("\n[5] 音视频同步...")
    syncer = AVSynchronizer(video_fps=24, audio_sample_rate=16000)
    
    for i in range(10):
        ts = time.time()
        syncer.add_video_frame(np.zeros((480, 640, 3), dtype=np.uint8), ts)
        syncer.add_audio_data(b'\x00' * 640, ts)
    
    frame = syncer.get_next_frame()
    print(f"   同步帧: {frame}")
    
    print("\n" + "=" * 60)
    print(" 演示完成!")
    print("=" * 60)


# ============ 主函数 ============
def main():
    parser = argparse.ArgumentParser(description='音视频流模块')
    
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'publish', 'subscribe', 'record', 'play'],
                       help='运行模式')
    
    # 发布模式
    parser.add_argument('--url', type=str, default=None,
                       help='流地址 (RTMP/WebRTC)')
    
    # 录制模式
    parser.add_argument('--source', type=str, default='0',
                       help='视频源 (0=摄像头, 文件路径, RTSP URL)')
    parser.add_argument('--duration', type=int, default=10,
                       help='录制时长(秒)')
    parser.add_argument('--output', type=str, default='output/recording',
                       help='输出文件')
    
    # 播放模式
    parser.add_argument('--file', type=str, default=None,
                       help='播放文件')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        asyncio.run(demo_streaming())
    
    elif args.mode == 'record':
        print(f"录制模式: {args.source}, 时长: {args.duration}s")
        
        # 录制视频
        recorder = AudioRecorder()
        capture = VideoCapture(source=args.source, fps=24)
        
        if capture.open(width=640, height=480):
            capture.start()
            recorder.start()
            
            time.sleep(args.duration)
            
            capture.stop()
            audio_data = recorder.stop()
            
            # 保存
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            recorder.save(f"{args.output}.wav")
            
            print(f"✓ 录制完成: {args.output}.wav")
    
    elif args.mode == 'play':
        if args.file:
            player = AudioPlayer()
            player.play_file(args.file)
    
    elif args.mode == 'publish':
        print(f"发布模式: {args.url}")
        # 需要配置 RTMP 服务器
        logger.error("需要配置 RTMP 服务器")


if __name__ == '__main__':
    main()
