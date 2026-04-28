#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双端口通信测试 - Port 5000 <-> Port 5001
模拟PLC(5000) 和 HMI(5001) 双向通信
"""
import socket
import struct
import threading
import time
import random


class DualPortCommunicator:
    """双端口通信器"""
    
    def __init__(self):
        self.running = True
        
        # 服务器 (PLC模拟器) - 端口5000
        self.server = None
        self.server_thread = None
        
        # 客户端 (HMI模拟器) - 端口5001
        self.client = None
        self.client_thread = None
        
        # 连接
        self.plc_socket = None
        self.hmi_socket = None
        
        # 数据
        self.regs = [0] * 8
        self.tick = 0
        
        # 锁
        self.lock = threading.Lock()
        
    def start(self):
        """启动通信"""
        print("=" * 60)
        print("  双端口通信测试")
        print("  PLC Server: 127.0.0.1:5000")
        print("  HMI Client: 127.0.0.1:5001")
        print("=" * 60)
        print()
        
        # 启动PLC服务器 (端口5000)
        self.server_thread = threading.Thread(target=self.run_plc_server, daemon=True)
        self.server_thread.start()
        print("[PLC] 服务器启动中... 端口 5000")
        
        # 等待服务器启动
        time.sleep(1)
        
        # 启动HMI客户端 (端口5001)
        self.client_thread = threading.Thread(target=self.run_hmi_client, daemon=True)
        self.client_thread.start()
        print("[HMI] 客户端启动中... 端口 5001")
        
        # 保持运行
        print()
        print("=" * 60)
        print("  通信已建立! TX/RX 数据流如下:")
        print("=" * 60)
        print()
        
        try:
            while self.running:
                time.sleep(0.5)
                self.update_data()
        except KeyboardInterrupt:
            self.stop()
            
    def update_data(self):
        """更新数据"""
        with self.lock:
            self.tick += 1
            self.regs[0] = random.randint(0, 2)   # CRANE_ST
            self.regs[1] = self.tick % 100        # CONT_POS
            self.regs[2] = random.randint(0, 5)   # TRUCK_CNT
            self.regs[3] = random.randint(0, 1)   # SHIP_DET
            self.regs[4] = 0                       # ALARM
            self.regs[5] = random.randint(0, 100) # SPEED
            self.regs[6] = random.randint(20, 35)  # TEMP
            self.regs[7] = random.randint(1000, 25000)  # WEIGHT
            
    def run_plc_server(self):
        """PLC服务器 - 监听端口5000"""
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server.bind(('127.0.0.1', 5000))
            self.server.listen(1)
            print("[PLC] 服务器就绪 127.0.0.1:5000")
            
            self.server.settimeout(1.0)
            
            while self.running:
                try:
                    client, addr = self.server.accept()
                    print(f"[PLC] HMI已连接: {addr}")
                    self.plc_socket = client
                    
                    # 处理HMI请求
                    self.handle_plc_request()
                    
                except socket.timeout:
                    continue
                    
        except Exception as e:
            print(f"[PLC] 服务器错误: {e}")
        finally:
            if self.server:
                self.server.close()
                
    def handle_plc_request(self):
        """处理PLC请求"""
        buffer = bytearray()
        
        while self.running and self.plc_socket:
            try:
                data = self.plc_socket.recv(1024)
                if not data:
                    break
                    
                buffer.extend(data)
                
                # 处理请求
                while len(buffer) >= 12:
                    # 检查Modbus TCP请求
                    if buffer[7] == 0x03:  # Read Holding Registers
                        byte_count = buffer[8]
                        frame_len = 9 + byte_count
                        
                        if len(buffer) >= frame_len:
                            frame = bytes(buffer[:frame_len])
                            buffer = buffer[frame_len:]
                            
                            # 打印发送的请求
                            print(f"[TX->] {frame.hex().upper()}")
                            
                            # 发送响应
                            self.send_plc_response(frame)
                        else:
                            break
                    else:
                        buffer.pop(0)
                        
            except Exception as e:
                if self.running:
                    print(f"[PLC] 接收错误: {e}")
                break
                
        print("[PLC] HMI已断开")
        self.plc_socket = None
        
    def send_plc_response(self, request):
        """发送PLC响应"""
        if not self.plc_socket:
            return
            
        try:
            # 解析请求
            tid = struct.unpack('>H', bytes(request[0:2]))[0]
            addr = struct.unpack('>H', bytes(request[8:10]))[0]
            count = struct.unpack('>H', bytes(request[10:12]))[0]
            
            # 构建响应
            with self.lock:
                response = struct.pack('>HH', tid, 0)  # Transaction ID, Protocol
                response += struct.pack('>H', 3 + count * 2)  # Length
                response += bytes([0x01, 0x03, count * 2])  # Unit, Function, Byte Count
                
                for i in range(count):
                    val = self.regs.get(addr + i, 0) if isinstance(self.regs, dict) else self.regs[addr + i] if addr + i < len(self.regs) else 0
                    response += struct.pack('>H', val)
            
            self.plc_socket.send(response)
            print(f"[<-RX] {response.hex().upper()}")
            
        except Exception as e:
            print(f"[PLC] 响应错误: {e}")
            
    def run_hmi_client(self):
        """HMI客户端 - 连接5000并监听5001"""
        time.sleep(0.5)  # 等待服务器
        
        # 连接PLC
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            self.client.connect(('127.0.0.1', 5000))
            print("[HMI] 已连接到PLC 127.0.0.1:5000")
            self.hmi_socket = self.client
            
            # 启动5001端口监听（可选，用于接收）
            threading.Thread(target=self.listen_5001, daemon=True).start()
            
            # 发送Modbus请求
            self.hmi_poll_loop()
            
        except Exception as e:
            print(f"[HMI] 连接失败: {e}")
        finally:
            if self.client:
                self.client.close()
                self.client = None
                
    def listen_5001(self):
        """监听5001端口（可选功能）"""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            listener.bind(('127.0.0.1', 5001))
            listener.listen(1)
            print("[HMI] 监听端口 5001 (用于双向通信测试)")
            
            listener.settimeout(1.0)
            
            while self.running:
                try:
                    conn, addr = listener.accept()
                    print(f"[HMI:5001] 收到连接: {addr}")
                    
                    # 处理连接
                    data = conn.recv(1024)
                    if data:
                        print(f"[HMI:5001] 收到数据: {data.hex().upper()}")
                        conn.send(b"[ACK] Data received on port 5001")
                    
                    conn.close()
                    
                except socket.timeout:
                    continue
                    
        except Exception as e:
            print(f"[HMI:5001] 监听错误: {e}")
        finally:
            listener.close()
            
    def hmi_poll_loop(self):
        """HMI轮询循环"""
        tid = 1
        
        while self.running and self.hmi_socket:
            try:
                # 构建Modbus读请求
                tid = (tid % 65535) + 1
                
                request = struct.pack('>HH', tid, 0)  # Transaction ID, Protocol
                request += struct.pack('>H', 6)  # Length
                request += bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x08])  # Unit, Function, Address, Quantity
                
                # 发送请求
                self.hmi_socket.send(request)
                print(f"[HMI] TX: {request.hex().upper()}")
                
                # 等待响应
                response = self.hmi_socket.recv(256)
                if response:
                    print(f"[HMI] RX: {response.hex().upper()}")
                    self.parse_response(response)
                    
                # 每秒轮询一次
                time.sleep(1)
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[HMI] 轮询错误: {e}")
                break
                
        print("[HMI] 轮询结束")
        
    def parse_response(self, data):
        """解析响应"""
        if len(data) >= 9 and data[7] == 0x03:
            byte_count = data[8]
            
            with self.lock:
                for i in range(min(byte_count // 2, 8)):
                    self.regs[i] = struct.unpack('>H', data[9 + i*2:11 + i*2])[0]
                    
            # 简单显示
            vals = [str(self.regs[i]) for i in range(min(byte_count//2, 8))]
            print(f"[HMI] 数据: {', '.join(vals)}")
            
    def stop(self):
        """停止"""
        print()
        print("=" * 60)
        print("  停止通信...")
        print("=" * 60)
        
        self.running = False
        
        if self.plc_socket:
            self.plc_socket.close()
        if self.client:
            self.client.close()
        if self.server:
            self.server.close()


if __name__ == "__main__":
    comm = DualPortCommunicator()
    comm.start()
