#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PortAI 下位机模拟器 v1.0
Python PLC Simulator

模拟工业PLC控制器，支持:
1. Modbus RTU (串口)
2. Modbus TCP
3. 数据寄存器读写
4. 港口设备状态模拟

运行:
    python PLCSimulator.py
"""

import socket
import serial
import struct
import threading
import time
import random
from datetime import datetime
from collections import defaultdict

# 尝试安装依赖
try:
    import pymodbus
except ImportError:
    print("[INFO] Installing pymodbus...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'pymodbus', '-q'])
    import pymodbus


class PLCSimulator:
    """PLC模拟器"""
    
    def __init__(self):
        # 数据寄存器 (Modbus Holding Registers 0-99)
        self.holding_registers = defaultdict(int)
        
        # 输入寄存器 (Modbus Input Registers 0-99)
        self.input_registers = defaultdict(int)
        
        # 线圈状态 (Modbus Coils 0-99)
        self.coils = defaultdict(bool)
        
        # 离散输入 (Modbus Discrete Inputs 0-99)
        self.discrete_inputs = defaultdict(bool)
        
        # 港口设备状态
        self.port_devices = {
            "crane_status": 0,      # 0=停止, 1=运行, 2=故障
            "container_pos": 0,     # 位置 0-100
            "truck_count": 0,       # 卡车数量
            "ship_detected": 0,     # 船舶检测 0/1
            "alarm": 0,             # 告警标志
            "speed": 0,             # 速度
            "temperature": 25,      # 温度
            "weight": 0,            # 重量
        }
        
        # 模拟参数
        self.running = True
        self.tick = 0
        
        # 日志
        self.logs = []
        
    def log(self, msg):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {msg}"
        self.logs.append(log_msg)
        print(log_msg)
        
    def update_port_devices(self):
        """更新港口设备状态"""
        self.tick += 1
        
        # 模拟数据变化
        self.port_devices["container_pos"] = (self.port_devices["container_pos"] + 1) % 100
        self.port_devices["truck_count"] = random.randint(0, 5)
        self.port_devices["temperature"] = 20 + random.randint(0, 15)
        self.port_devices["weight"] = random.randint(1000, 25000)
        
        # 随机事件
        if self.tick % 20 == 0:
            self.port_devices["ship_detected"] = random.randint(0, 1)
            
        if random.random() < 0.02:
            self.port_devices["alarm"] = 1 if self.port_devices["alarm"] == 0 else 0
            
        if self.tick % 10 == 0:
            self.port_devices["speed"] = random.randint(0, 100)
            self.port_devices["crane_status"] = random.randint(0, 2)
        
        # 同步到Modbus寄存器
        self.holding_registers[0] = self.port_devices["crane_status"]
        self.holding_registers[1] = self.port_devices["container_pos"]
        self.holding_registers[2] = self.port_devices["truck_count"]
        self.holding_registers[3] = self.port_devices["ship_detected"]
        self.holding_registers[4] = self.port_devices["alarm"]
        self.holding_registers[5] = self.port_devices["speed"]
        self.holding_registers[6] = self.port_devices["temperature"]
        self.holding_registers[7] = self.port_devices["weight"]
        
        # 离散输入 (DI)
        self.discrete_inputs[0] = self.port_devices["crane_status"] == 1
        self.discrete_inputs[1] = self.port_devices["alarm"] == 1
        self.discrete_inputs[2] = self.port_devices["ship_detected"] == 1
        
    def calc_crc16(self, data):
        """计算Modbus CRC16"""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc
        
    def handle_modbus_rtu(self, data):
        """处理Modbus RTU请求"""
        if len(data) < 5:
            return None
            
        # 解析
        slave_id = data[0]
        function_code = data[1]
        
        if function_code == 0x03:  # 读保持寄存器
            addr = struct.unpack('>H', bytes(data[2:4]))[0]
            count = struct.unpack('>H', bytes(data[4:6]))[0]
            
            # 计算CRC
            crc = struct.pack('<H', self.calc_crc16(data[:6]))
            
            # 响应
            response = bytes([slave_id, function_code, count * 2])
            for i in range(count):
                val = self.holding_registers.get(addr + i, 0)
                response += struct.pack('>H', val)
            response += struct.pack('<H', self.calc_crc16(response))
            
            return response
            
        elif function_code == 0x04:  # 读输入寄存器
            addr = struct.unpack('>H', bytes(data[2:4]))[0]
            count = struct.unpack('>H', bytes(data[4:6]))[0]
            
            response = bytes([slave_id, function_code, count * 2])
            for i in range(count):
                val = self.input_registers.get(addr + i, 0)
                response += struct.pack('>H', val)
            response += struct.pack('<H', self.calc_crc16(response))
            
            return response
            
        elif function_code == 0x01:  # 读线圈
            addr = struct.unpack('>H', bytes(data[2:4]))[0]
            count = struct.unpack('>H', bytes(data[4:6]))[0]
            
            response = bytes([slave_id, function_code, (count + 7) // 8])
            byte_val = 0
            for i in range(count):
                if self.coils.get(addr + i, False):
                    byte_val |= (1 << (i % 8))
                if (i % 8 == 7) or (i == count - 1):
                    response += bytes([byte_val])
                    byte_val = 0
            response += struct.pack('<H', self.calc_crc16(response))
            
            return response
            
        elif function_code == 0x05:  # 写单个线圈
            addr = struct.unpack('>H', bytes(data[2:4]))[0]
            value = struct.unpack('>H', bytes(data[4:6]))[0]
            self.coils[addr] = (value == 0xFF00)
            
            # 响应回显
            return data[:6] + struct.pack('<H', self.calc_crc16(data[:6]))
            
        return None
        
    def handle_modbus_tcp(self, data):
        """处理Modbus TCP请求"""
        if len(data) < 12:
            return None
            
        # 解析
        transaction_id = struct.unpack('>H', bytes(data[0:2]))[0]
        protocol_id = struct.unpack('>H', bytes(data[2:4]))[0]
        length = struct.unpack('>H', bytes(data[4:6]))[0]
        slave_id = data[6]
        function_code = data[7]
        
        if function_code == 0x03:  # 读保持寄存器
            addr = struct.unpack('>H', bytes(data[8:10]))[0]
            count = struct.unpack('>H', bytes(data[10:12]))[0]
            
            # Modbus TCP 响应 (无CRC!)
            response = struct.pack('>HH', transaction_id, protocol_id)
            response += struct.pack('>H', 3 + count * 2)  # Length: 3 + byte_count
            response += bytes([slave_id, function_code, count * 2])  # Function + byte_count
            
            for i in range(count):
                val = self.holding_registers.get(addr + i, 0)
                response += struct.pack('>H', val)
            
            return response
            
        return None
        
    def start_serial_server(self, port='COM3', baudrate=9600):
        """启动串口服务器"""
        try:
            ser = serial.Serial(port, baudrate, timeout=0.1)
            self.log(f"[RTU] 串口服务器启动 {port} @ {baudrate}")
            
            buffer = bytearray()
            
            while self.running:
                try:
                    if ser.in_waiting:
                        buffer.extend(ser.read(ser.in_waiting))
                        
                        # 处理完整帧 (简单方法: 等待一定时间或足够数据)
                        if len(buffer) >= 8:
                            # 尝试解析
                            frame = bytes(buffer[:min(260, len(buffer))])
                            response = self.handle_modbus_rtu(frame)
                            if response:
                                ser.write(response)
                                self.log(f"[RTU] 响应: {response.hex().upper()}")
                                buffer = buffer[len(frame):]
                            else:
                                buffer = buffer[1:]  # 移出一个字节继续尝试
                                
                    self.update_port_devices()
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.log(f"[RTU] Error: {e}")
                    
        except serial.SerialException as e:
            self.log(f"[RTU] 串口打开失败: {e}")
            self.log("[RTU] 提示: 请检查串口是否被占用或指定正确的串口号")
        finally:
            if 'ser' in locals():
                ser.close()
                
    def start_tcp_server(self, host='127.0.0.1', port=5000):
        """启动TCP服务器"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(1.0)  # 添加超时让循环能检查running状态
        
        try:
            server.bind((host, port))
            server.listen(5)
            self.log(f"[TCP] 服务器启动 {host}:{port}")
            
            while self.running:
                try:
                    client, addr = server.accept()
                    self.log(f"[TCP] 客户端连接: {addr}")
                    client.settimeout(5.0)
                    
                    while self.running:
                        try:
                            data = client.recv(1024)
                            if not data:
                                break
                            
                            self.log(f"[TCP] 收到: {data.hex().upper()}")
                            
                            # 更新寄存器数据
                            self.update_port_devices()
                            
                            # 处理Modbus请求
                            response = self.handle_modbus_tcp(bytes(data))
                            if response:
                                client.send(response)
                                self.log(f"[TCP] 响应: {response.hex().upper()}")
                                
                        except socket.timeout:
                            # 超时后继续循环检查running状态
                            self.update_port_devices()
                            continue
                            
                    client.close()
                    self.log(f"[TCP] 客户端断开: {addr}")
                    
                except socket.timeout:
                    # 接受超时，继续检查running
                    self.update_port_devices()
                    continue
                except Exception as e:
                    if self.running:
                        self.log(f"[TCP] Error: {e}")
                    
        except OSError as e:
            self.log(f"[TCP] 端口绑定失败: {e}")
            self.log("[TCP] 提示: 端口可能被占用，请使用其他端口")
        finally:
            server.close()
            
    def show_status(self):
        """显示状态"""
        print("\n" + "="*60)
        print("PortAI PLC Simulator Status")
        print("="*60)
        print(f"Tick: {self.tick}")
        print("\nPort Devices:")
        for key, val in self.port_devices.items():
            print(f"  {key}: {val}")
        print("\nHolding Registers (0-7):")
        for i in range(8):
            print(f"  [{i}] = {self.holding_registers[i]}")
        print("="*60)


def main():
    print("\n" + "="*60)
    print("    PortAI 下位机模拟器 v1.0")
    print("    PLC Simulator for Industrial Communication")
    print("="*60)
    print()
    
    plc = PLCSimulator()
    
    # 选择通信方式
    print("选择通信方式 / Select Communication Method:")
    print("  1. TCP Server (Modbus TCP) - 推荐")
    print("  2. Serial Port (Modbus RTU)")
    print("  3. Both / 全部")
    print()
    
    try:
        choice = input("请选择 (1-3): ").strip()
    except:
        choice = "1"
        
    if choice == "1":
        # TCP服务器
        port = int(input("端口 (默认 5000): ").strip() or "5000")
        tcp_thread = threading.Thread(target=plc.start_tcp_server, args=('127.0.0.1', port), daemon=True)
        tcp_thread.start()
        
        # 状态显示线程
        def show_loop():
            while plc.running:
                time.sleep(3)
                plc.show_status()
        status_thread = threading.Thread(target=show_loop, daemon=True)
        status_thread.start()
        
        input("\n按 Enter 退出...\n")
        
    elif choice == "2":
        # 串口服务器
        port = input("串口 (默认 COM3): ").strip() or "COM3"
        baud = int(input("波特率 (默认 9600): ").strip() or "9600")
        
        serial_thread = threading.Thread(target=plc.start_serial_server, args=(port, baud), daemon=True)
        serial_thread.start()
        
        input("\n按 Enter 退出...\n")
        
    else:
        # 全部
        port = int(input("TCP端口 (默认 5000): ").strip() or "5000")
        serial_port = input("串口 (默认 COM3): ").strip() or "COM3"
        
        tcp_thread = threading.Thread(target=plc.start_tcp_server, args=('127.0.0.1', port), daemon=True)
        serial_thread = threading.Thread(target=plc.start_serial_server, args=(serial_port, 9600), daemon=True)
        
        tcp_thread.start()
        serial_thread.start()
        
        input("\n按 Enter 退出...\n")
        
    plc.running = False
    print("PLC Simulator stopped.")


if __name__ == "__main__":
    main()
