#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PortAI 通信引擎 - 状态机 + 线程安全 + 内存管理
工业级通信框架
"""
import socket
import struct
import threading
import queue
import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List
from collections import deque
import weakref
import gc

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 状态机定义
# ============================================================

class CommState(Enum):
    """通信状态"""
    IDLE = auto()           # 空闲
    CONNECTING = auto()     # 连接中
    CONNECTED = auto()      # 已连接
    SENDING = auto()        # 发送中
    WAITING = auto()        # 等待响应
    RECEIVING = auto()      # 接收中
    ERROR = auto()          # 错误
    DISCONNECTING = auto()  # 断开中


class CommEvent(Enum):
    """通信事件"""
    CONNECT = auto()         # 连接
    CONNECT_SUCCESS = auto() # 连接成功
    CONNECT_FAIL = auto()   # 连接失败
    SEND = auto()           # 发送
    SEND_COMPLETE = auto()  # 发送完成
    RESPONSE = auto()       # 收到响应
    TIMEOUT = auto()        # 超时
    ERROR = auto()          # 错误
    DISCONNECT = auto()     # 断开
    RECONNECT = auto()      # 重连


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ModbusFrame:
    """Modbus数据帧"""
    transaction_id: int = 0
    protocol_id: int = 0
    slave_id: int = 1
    function_code: int = 0x03
    start_address: int = 0
    quantity: int = 8
    data: bytes = field(default_factory=bytes)
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    
    def to_bytes(self) -> bytes:
        """序列化"""
        frame = struct.pack('>HH', self.transaction_id, self.protocol_id)
        frame += struct.pack('>H', 6 + len(self.data))
        frame += bytes([self.slave_id, self.function_code])
        if self.data:
            frame += self.data
        return frame
    
    @staticmethod
    def read_request(slave_id: int = 1, address: int = 0, quantity: int = 8, 
                      tid: int = 1) -> 'ModbusFrame':
        """创建读请求帧"""
        return ModbusFrame(
            transaction_id=tid,
            slave_id=slave_id,
            function_code=0x03,
            start_address=address,
            quantity=quantity,
            data=struct.pack('>HH', address, quantity)
        )


@dataclass 
class CommResult:
    """通信结果"""
    success: bool
    state: CommState
    data: Optional[bytes] = None
    error_msg: str = ""
    timestamp: float = field(default_factory=time.time)


# ============================================================
# 线程安全的通信引擎
# ============================================================

class CommEngine(threading.Thread):
    """
    通信引擎 - 状态机驱动
    线程安全，支持TCP/RTU
    """
    
    # 最大队列长度
    MAX_QUEUE_SIZE = 100
    # 最大历史记录
    MAX_HISTORY = 1000
    # 默认超时
    DEFAULT_TIMEOUT = 5.0
    # 重试次数
    MAX_RETRY = 3
    
    def __init__(self, protocol: str = "TCP", 
                 host: str = "127.0.0.1", port: int = 5000,
                 serial_port: str = None, baudrate: int = 9600):
        super().__init__(daemon=True)
        
        # 连接参数
        self.protocol = protocol.upper()
        self.host = host
        self.port = port
        self.serial_port = serial_port
        self.baudrate = baudrate
        
        # 状态机
        self._state = CommState.IDLE
        self._state_lock = threading.RLock()
        
        # Socket/Serial
        self._socket: Optional[socket.socket] = None
        self._serial = None
        
        # 线程安全队列
        self._send_queue: queue.Queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self._recv_queue: queue.Queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self._event_queue: queue.Queue = queue.Queue(maxsize=50)
        
        # 回调函数
        self._callbacks: Dict[str, List[Callable]] = {
            'on_state_change': [],
            'on_data': [],
            'on_error': [],
            'on_log': []
        }
        
        # 统计
        self._stats = {
            'tx_count': 0,
            'rx_count': 0,
            'error_count': 0,
            'retry_count': 0,
            'last_tx_time': 0,
            'last_rx_time': 0
        }
        self._stats_lock = threading.Lock()
        
        # 历史记录 (内存保护)
        self._history: deque = deque(maxlen=self.MAX_HISTORY)
        self._history_lock = threading.Lock()
        
        # Transaction ID
        self._tid_lock = threading.Lock()
        self._transaction_id = 0
        
        # 运行标志
        self._running = False
        self._stop_event = threading.Event()
        
        # 超时管理
        self._timeout = self.DEFAULT_TIMEOUT
        self._last_send_time = 0
        
        # 内存管理 - 弱引用
        self._weak_refs: List = []
        
        logger.info(f"CommEngine initialized: {protocol} {host}:{port}")
        
    # --------------------------------------------------------
    # 属性
    # --------------------------------------------------------
    
    @property
    def state(self) -> CommState:
        with self._state_lock:
            return self._state
    
    @state.setter
    def state(self, new_state: CommState):
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            if old_state != new_state:
                logger.info(f"State: {old_state.name} -> {new_state.name}")
                self._emit_callback('on_state_change', old_state, new_state)
    
    @property
    def stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            return self._stats.copy()
    
    # --------------------------------------------------------
    # 线程管理
    # --------------------------------------------------------
    
    def run(self):
        """主循环"""
        self._running = True
        logger.info("CommEngine started")
        
        while self._running:
            try:
                self._main_loop()
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                self.state = CommState.ERROR
                
        logger.info("CommEngine stopped")
        
    def stop(self):
        """安全停止"""
        logger.info("Stopping CommEngine...")
        self._running = False
        self._stop_event.set()
        self._cleanup()
        
    def _main_loop(self):
        """主循环逻辑"""
        state = self.state
        
        if state == CommState.IDLE:
            self._handle_idle()
            
        elif state == CommState.CONNECTING:
            self._handle_connecting()
            
        elif state == CommState.CONNECTED:
            self._handle_connected()
            
        elif state == CommState.WAITING:
            self._handle_waiting()
            
        elif state == CommState.ERROR:
            self._handle_error()
            
        # 短暂休眠防止CPU满载
        time.sleep(0.001)
        
    # --------------------------------------------------------
    # 状态处理
    # --------------------------------------------------------
    
    def _handle_idle(self):
        """空闲状态 - 处理连接请求"""
        try:
            event = self._event_queue.get_nowait()
            
            if event == CommEvent.CONNECT:
                self._do_connect()
            elif event == CommEvent.RECONNECT:
                self._do_connect()
                
        except queue.Empty:
            pass
            
    def _handle_connecting(self):
        """连接中状态"""
        try:
            event = self._event_queue.get_nowait()
            if event == CommEvent.DISCONNECT:
                self.state = CommState.DISCONNECTING
                return
        except queue.Empty:
            pass
            
        # 连接超时检测
        if time.time() - self._last_send_time > self._timeout:
            logger.warning("Connection timeout")
            self.state = CommState.ERROR
            self._emit_callback('on_error', "Connection timeout")
            
    def _handle_connected(self):
        """已连接状态 - 发送队列中的请求"""
        # 检查是否有待发送数据
        try:
            frame = self._send_queue.get_nowait()
            self._do_send(frame)
        except queue.Empty:
            pass
            
    def _handle_waiting(self):
        """等待响应状态"""
        try:
            # 非阻塞检查事件
            event = self._event_queue.get_nowait()
            
            if event == CommEvent.DISCONNECT:
                self.state = CommState.DISCONNECTING
                return
            elif event == CommEvent.RECONNECT:
                self.state = CommState.CONNECTING
                return
                
        except queue.Empty:
            pass
            
        # 超时检测
        if time.time() - self._last_send_time > self._timeout:
            logger.warning("Response timeout")
            self._handle_timeout()
            
    def _handle_error(self):
        """错误状态"""
        try:
            event = self._event_queue.get(timeout=1.0)
            
            if event == CommEvent.RECONNECT:
                self._do_disconnect()
                self._do_connect()
            elif event == CommEvent.DISCONNECT:
                self.state = CommState.DISCONNECTING
            elif event == CommEvent.CONNECT:
                self._do_connect()
                
        except queue.Empty:
            pass
            
    # --------------------------------------------------------
    # 连接操作
    # --------------------------------------------------------
    
    def connect(self):
        """请求连接"""
        self._event_queue.put(CommEvent.CONNECT)
        
    def disconnect(self):
        """请求断开"""
        self._event_queue.put(CommEvent.DISCONNECT)
        
    def _do_connect(self):
        """执行连接"""
        self.state = CommState.CONNECTING
        self._last_send_time = time.time()
        
        try:
            if self.protocol == "TCP":
                self._do_connect_tcp()
            else:
                self._do_connect_serial()
                
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.state = CommState.ERROR
            self._emit_callback('on_error', str(e))
            
    def _do_connect_tcp(self):
        """TCP连接"""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(self._timeout)
        self._socket.connect((self.host, self.port))
        
        logger.info(f"TCP connected: {self.host}:{self.port}")
        self.state = CommState.CONNECTED
        self._emit_callback('on_state_change', CommState.CONNECTING, CommState.CONNECTED)
        
        # 启动接收线程
        recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        recv_thread.start()
        
    def _do_connect_serial(self):
        """串口连接"""
        import serial
        self._serial = serial.Serial(
            port=self.serial_port,
            baudrate=self.baudrate,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=self._timeout
        )
        
        logger.info(f"Serial connected: {self.serial_port}@{self.baudrate}")
        self.state = CommState.CONNECTED
        
        # 启动接收线程
        recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        recv_thread.start()
        
    def _do_disconnect(self):
        """执行断开"""
        try:
            if self._socket:
                self._socket.close()
                self._socket = None
            if self._serial:
                self._serial.close()
                self._serial = None
        except Exception as e:
            logger.warning(f"Disconnect error: {e}")
            
        self.state = CommState.IDLE
        logger.info("Disconnected")
        
    # --------------------------------------------------------
    # 数据收发
    # --------------------------------------------------------
    
    def send_read_request(self, slave_id: int = 1, address: int = 0, 
                         quantity: int = 8) -> bool:
        """发送读请求"""
        frame = ModbusFrame.read_request(slave_id, address, quantity, 
                                         self._next_tid())
        return self._send_queue.put_nowait(frame)
        
    def _do_send(self, frame: ModbusFrame):
        """执行发送"""
        self.state = CommState.SENDING
        self._last_send_time = time.time()
        
        try:
            data = frame.to_bytes()
            
            if self._socket:
                self._socket.send(data)
            elif self._serial:
                self._serial.write(data)
            else:
                raise Exception("No connection")
                
            # 统计
            with self._stats_lock:
                self._stats['tx_count'] += 1
                self._stats['last_tx_time'] = time.time()
                
            logger.debug(f"TX: {data.hex().upper()}")
            self._emit_callback('on_log', f"TX: {data.hex().upper()}")
            
            self.state = CommState.WAITING
            
        except Exception as e:
            logger.error(f"Send failed: {e}")
            frame.retry_count += 1
            
            if frame.retry_count < self.MAX_RETRY:
                self._send_queue.put(frame)
                with self._stats_lock:
                    self._stats['retry_count'] += 1
            else:
                self.state = CommState.ERROR
                self._emit_callback('on_error', f"Send failed after {self.MAX_RETRY} retries")
                
    def _recv_loop(self):
        """接收循环 - 独立线程"""
        logger.info("Receive loop started")
        
        while self._running and self.state != CommState.DISCONNECTING:
            try:
                if self._socket:
                    data = self._socket.recv(1024)
                elif self._serial:
                    if self._serial.in_waiting:
                        data = self._serial.read(self._serial.in_waiting)
                    else:
                        time.sleep(0.01)
                        continue
                else:
                    break
                    
                if not data:
                    logger.warning("Connection closed by peer")
                    self._event_queue.put(CommEvent.ERROR)
                    break
                    
                # 处理接收数据
                self._process_recv(data)
                
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Recv error: {e}")
                    self._event_queue.put(CommEvent.ERROR)
                break
                
        logger.info("Receive loop ended")
        
    def _process_recv(self, data: bytes):
        """处理接收数据"""
        with self._stats_lock:
            self._stats['rx_count'] += 1
            self._stats['last_rx_time'] = time.time()
            
        logger.debug(f"RX: {data.hex().upper()}")
        self._emit_callback('on_log', f"RX: {data.hex().upper()}")
        
        # 放入接收队列
        self._recv_queue.put(data)
        
        # 状态切换
        if self.state == CommState.WAITING:
            self.state = CommState.CONNECTED
            
        # 回调
        self._emit_callback('on_data', data)
        
        # 记录历史
        self._add_history('RX', data)
        
    def _handle_timeout(self):
        """处理超时"""
        with self._stats_lock:
            self._stats['error_count'] += 1
            
        self._emit_callback('on_error', "Response timeout")
        
        # 自动重连
        self._event_queue.put(CommEvent.RECONNECT)
        
    # --------------------------------------------------------
    # 辅助方法
    # --------------------------------------------------------
    
    def _next_tid(self) -> int:
        """生成下一个Transaction ID"""
        with self._tid_lock:
            self._transaction_id = (self._transaction_id % 65535) + 1
            return self._transaction_id
            
    def _add_history(self, direction: str, data: bytes):
        """添加历史记录"""
        with self._history_lock:
            self._history.append({
                'dir': direction,
                'data': data.hex(),
                'time': time.time()
            })
            
    def _cleanup(self):
        """清理资源"""
        self._do_disconnect()
        
        # 清空队列
        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except queue.Empty:
                break
                
        # 强制垃圾回收
        gc.collect()
        
    # --------------------------------------------------------
    # 回调管理
    # --------------------------------------------------------
    
    def on_state_change(self, callback: Callable):
        """注册状态变化回调"""
        self._callbacks['on_state_change'].append(callback)
        
    def on_data(self, callback: Callable):
        """注册数据回调"""
        self._callbacks['on_data'].append(callback)
        
    def on_error(self, callback: Callable):
        """注册错误回调"""
        self._callbacks['on_error'].append(callback)
        
    def on_log(self, callback: Callable):
        """注册日志回调"""
        self._callbacks['on_log'].append(callback)
        
    def _emit_callback(self, name: str, *args, **kwargs):
        """触发回调"""
        for callback in self._callbacks.get(name, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error: {e}")


# ============================================================
# 单元测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  PortAI CommEngine Test")
    print("=" * 60)
    
    # 创建引擎
    engine = CommEngine(protocol="TCP", host="127.0.0.1", port=5000)
    
    # 注册回调
    engine.on_state_change(lambda old, new: print(f"  State: {old.name} -> {new.name}"))
    engine.on_data(lambda d: print(f"  RX: {d.hex().upper()}"))
    engine.on_error(lambda e: print(f"  ERROR: {e}"))
    engine.on_log(lambda msg: print(f"  {msg}"))
    
    # 启动引擎
    engine.start()
    
    # 连接
    print("\nConnecting...")
    engine.connect()
    time.sleep(2)
    
    # 发送请求
    print("\nSending request...")
    engine.send_read_request(slave_id=1, address=0, quantity=8)
    time.sleep(2)
    
    # 发送更多请求
    for i in range(3):
        print(f"\nRequest {i+1}...")
        engine.send_read_request()
        time.sleep(1)
    
    # 打印统计
    print("\n" + "=" * 60)
    print("  Statistics")
    print("=" * 60)
    stats = engine.stats
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # 停止
    print("\nStopping...")
    engine.disconnect()
    time.sleep(1)
    engine.stop()
    
    print("\nDone!")
