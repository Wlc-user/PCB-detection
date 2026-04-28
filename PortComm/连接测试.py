#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PortAI 连接测试工具 v1.0
自动测试Modbus TCP通信
"""
import socket
import struct
import time
import sys
import os
import subprocess
import threading
import signal

# 全局变量
plc_process = None
running = True


def signal_handler(sig, frame):
    global running, plc_process
    print("\nStopping PLC simulator...")
    running = False
    if plc_process:
        plc_process.terminate()
    sys.exit(0)


def start_plc_simulator():
    """启动PLC模拟器"""
    global plc_process
    
    print("=" * 60)
    print("  PortAI Connection Test Tool")
    print("=" * 60)
    print()
    
    print("[1/5] Starting PLC simulator...")
    
    try:
        plc_process = subprocess.Popen(
            [sys.executable, "start_plc.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        time.sleep(2)
        
        if plc_process.poll() is not None:
            print("[ERROR] PLC simulator failed to start!")
            return False
            
        print("       PLC simulator started (127.0.0.1:5000)")
        return True
        
    except Exception as e:
        print(f"[ERROR] Start failed: {e}")
        return False


def test_connection():
    """测试连接"""
    print()
    print("[2/5] Testing TCP connection...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    
    try:
        sock.connect(("127.0.0.1", 5000))
        print("       [OK] TCP connected!")
        return sock
    except Exception as e:
        print(f"       [FAIL] TCP failed: {e}")
        return None


def send_and_receive(sock):
    """发送请求并接收响应"""
    print()
    print("[3/5] Sending Modbus request...")
    
    # Modbus TCP request: Read holding registers 0-7
    request = bytes([
        0x00, 0x01,  # Transaction ID
        0x00, 0x00,  # Protocol ID
        0x00, 0x06,  # Length
        0x01,        # Unit ID
        0x03,        # Function: Read Holding Registers
        0x00, 0x00,  # Start Address: 0
        0x00, 0x08   # Quantity: 8 registers
    ])
    
    print(f"       TX: {request.hex().upper()}")
    
    try:
        sock.send(request)
        print("       [OK] Send success!")
    except Exception as e:
        print(f"       [FAIL] Send failed: {e}")
        return None
    
    print()
    print("[4/5] Waiting for PLC response...")
    
    try:
        response = sock.recv(256)
        print(f"       RX: {response.hex().upper()}")
        print("       [OK] Receive success!")
        return response
    except socket.timeout:
        print("       [FAIL] Receive timeout!")
        return None
    except Exception as e:
        print(f"       [FAIL] Receive failed: {e}")
        return None


def parse_response(data):
    """解析响应"""
    print()
    print("[5/5] Parsing data...")
    print()
    print("=" * 60)
    print("  Register Data")
    print("=" * 60)
    
    if data and len(data) >= 9 and data[7] == 0x03:
        byte_count = data[8]
        
        registers = [
            ("0", "CRANE_ST", "Crane Status"),
            ("1", "CONT_POS", "Container Pos"),
            ("2", "TRUCK_CNT", "Truck Count"),
            ("3", "SHIP_DET", "Ship Detect"),
            ("4", "ALARM", "Alarm Flag"),
            ("5", "SPEED", "Speed"),
            ("6", "TEMP", "Temperature"),
            ("7", "WEIGHT", "Weight")
        ]
        
        print(f"{'Addr':<6} {'Name':<12} {'Value':<8} {'Description'}")
        print("-" * 60)
        
        for i, (addr, name, desc) in enumerate(registers[:byte_count//2]):
            if 9 + i * 2 + 2 <= len(data):
                value = struct.unpack('>H', data[9 + i*2 : 11 + i*2])[0]
                print(f"R{addr:<5} {name:<12} {value:<8} {desc}")
        
        print()
        print("=" * 60)
        print("  [OK] Test completed! Communication is working!")
        print("=" * 60)
        return True
    else:
        print("       [FAIL] Invalid response format!")
        return False


def main():
    global plc_process, running
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # 启动PLC
    if not start_plc_simulator():
        return
    
    # 连接测试
    sock = test_connection()
    if not sock:
        if plc_process:
            plc_process.terminate()
        return
    
    # 发送接收
    response = send_and_receive(sock)
    if response:
        parse_response(response)
    
    # 关闭
    sock.close()
    
    print()
    print("Press Ctrl+C to stop PLC simulator...")
    
    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if plc_process:
            plc_process.terminate()
            print("PLC simulator stopped")


if __name__ == "__main__":
    main()
