#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PortAI 工业通信上位机 - 生产版 v4.0
支持:
- Modbus TCP / RTU 真实PLC连接
- TX/RX 发送接收指示
- 数据监控
- 报警记录

运行: python production_hmi.py
"""
import socket
import struct
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import csv
import time
from datetime import datetime
from collections import deque

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# 配置
CONFIG = {
    "tcp": {"enabled": True, "host": "127.0.0.1", "port": 5001, "timeout": 5},
    "serial": {"enabled": False, "port": "COM3", "baudrate": 9600, "timeout": 1},
    "modbus": {"slave_id": 1, "start_address": 0, "register_count": 8},
    "registers": {
        "0": {"name": "CRANE_ST", "desc": "起重机状态", "unit": ""},
        "1": {"name": "CONT_POS", "desc": "集装箱位置", "unit": ""},
        "2": {"name": "TRUCK_CNT", "desc": "卡车数量", "unit": "辆"},
        "3": {"name": "SHIP_DET", "desc": "船舶检测", "unit": ""},
        "4": {"name": "ALARM", "desc": "告警标志", "unit": ""},
        "5": {"name": "SPEED", "desc": "速度", "unit": "km/h"},
        "6": {"name": "TEMP", "desc": "温度", "unit": "°C"},
        "7": {"name": "WEIGHT", "desc": "重量", "unit": "kg"}
    },
    "alarms": {"temperature_high": 35, "weight_max": 50000, "speed_max": 120},
    "poll_interval": 500
}


class PortHMI:
    """PortAI 上位机主类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PortAI 工业监控系统 v4.0")
        self.root.geometry("1000x700")
        self.root.configure(bg='#1a1a2e')
        
        # 通信
        self.client = None
        self.serial_port = None
        self.connected = False
        self.running = True
        self.transaction_id = 1
        
        # 数据
        self.reg_values = [0] * 8
        self.reg_history = deque(maxlen=100)
        self.alarm_count = 0
        self.start_time = None
        
        # TX/RX 统计
        self.tx_count = 0
        self.rx_count = 0
        
        # TX/RX 指示灯
        self.tx_indicator = None
        self.rx_indicator = None
        
        # 配置
        self.tcp_enabled = tk.BooleanVar(value=CONFIG["tcp"]["enabled"])
        self.serial_enabled = tk.BooleanVar(value=CONFIG["serial"]["enabled"])
        
        self.setup_ui()
        
        # 定时器
        self.poll_timer = None
        
    def setup_ui(self):
        """设置UI"""
        
        # === 标题栏 ===
        title_frame = tk.Frame(self.root, bg='#16213e', height=50)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="PortAI 工业监控系统",
                font=("Microsoft YaHei", 16, "bold"),
                fg='#00ff41', bg='#16213e').pack(side='left', padx=20)
        
        self.status_indicator = tk.Label(title_frame, text="●",
                                        font=("Arial", 20),
                                        fg='gray', bg='#16213e')
        self.status_indicator.pack(side='left', padx=5)
        
        self.status_text = tk.Label(title_frame, text="离线",
                                   font=("Microsoft YaHei", 10),
                                   fg='gray', bg='#16213e')
        self.status_text.pack(side='left')
        
        # TX/RX 指示灯
        tk.Label(title_frame, text="TX:", bg='#16213e', fg='white', font=("Arial", 10, "bold")).pack(side='right', padx=(20, 5))
        self.tx_indicator = tk.Label(title_frame, text="●", font=("Arial", 16), fg='gray', bg='#16213e')
        self.tx_indicator.pack(side='right')
        
        tk.Label(title_frame, text="RX:", bg='#16213e', fg='white', font=("Arial", 10, "bold")).pack(side='right', padx=(10, 5))
        self.rx_indicator = tk.Label(title_frame, text="●", font=("Arial", 16), fg='gray', bg='#16213e')
        self.rx_indicator.pack(side='right')
        
        # === 连接设置区 ===
        conn_frame = tk.LabelFrame(self.root, text="连接设置",
                                  bg='#16213e', fg='white',
                                  font=("Microsoft YaHei", 10), padx=10, pady=10)
        conn_frame.pack(fill='x', padx=10, pady=5)
        
        # TCP设置
        tcp_frame = tk.Frame(conn_frame, bg='#16213e')
        tcp_frame.pack(fill='x')
        
        tk.Checkbutton(tcp_frame, text="TCP", variable=self.tcp_enabled,
                      bg='#16213e', fg='white', selectcolor='#16213e',
                      activebackground='#16213e').pack(side='left')
        
        tk.Label(tcp_frame, text="IP:", bg='#16213e', fg='white').pack(side='left', padx=(10, 5))
        self.host_entry = tk.Entry(tcp_frame, width=15, bg='#0a0a15', fg='#00ff41')
        self.host_entry.insert(0, CONFIG["tcp"]["host"])
        self.host_entry.pack(side='left')
        
        tk.Label(tcp_frame, text="Port:", bg='#16213e', fg='white').pack(side='left', padx=(10, 5))
        self.tcp_port_entry = tk.Entry(tcp_frame, width=8, bg='#0a0a15', fg='#00ff41')
        self.tcp_port_entry.insert(0, str(CONFIG["tcp"]["port"]))
        self.tcp_port_entry.pack(side='left')
        
        # 串口设置
        serial_frame = tk.Frame(conn_frame, bg='#16213e')
        serial_frame.pack(fill='x', pady=(10, 0))
        
        tk.Checkbutton(serial_frame, text="Serial", variable=self.serial_enabled,
                      bg='#16213e', fg='white', selectcolor='#16213e',
                      activebackground='#16213e').pack(side='left')
        
        tk.Label(serial_frame, text="COM:", bg='#16213e', fg='white').pack(side='left', padx=(10, 5))
        self.serial_entry = tk.Entry(serial_frame, width=10, bg='#0a0a15', fg='#00ff41')
        self.serial_entry.insert(0, CONFIG["serial"]["port"])
        self.serial_entry.pack(side='left')
        
        tk.Label(serial_frame, text="Baud:", bg='#16213e', fg='white').pack(side='left', padx=(10, 5))
        self.baud_combo = ttk.Combobox(serial_frame, width=8, values=['9600', '19200', '38400', '115200'])
        self.baud_combo.current(0)
        self.baud_combo.pack(side='left')
        
        # 连接按钮
        self.connect_btn = tk.Button(conn_frame, text="▶ 连接",
                                     command=self.toggle_connection,
                                     bg='#00aa55', fg='white', font=("Microsoft YaHei", 11, "bold"),
                                     width=15, height=2)
        self.connect_btn.pack(pady=10)
        
        # === 数据面板 ===
        data_frame = tk.LabelFrame(self.root, text="数据寄存器 (Modbus 0-7)",
                                  bg='#16213e', fg='white',
                                  font=("Microsoft YaHei", 10), padx=10, pady=10)
        data_frame.pack(fill='x', padx=10, pady=5)
        
        self.data_labels = []
        self.data_frames = []
        
        for i in range(8):
            frame = tk.Frame(data_frame, bg='#0a0a15', relief='sunken', bd=1)
            frame.pack(side='left', padx=3, pady=5, fill='both', expand=True)
            self.data_frames.append(frame)
            
            name = CONFIG["registers"][str(i)]["name"]
            desc = CONFIG["registers"][str(i)]["desc"]
            unit = CONFIG["registers"][str(i)]["unit"]
            
            tk.Label(frame, text=f"{i}: {name}",
                    bg='#0a0a15', fg='#888888', font=("Consolas", 8)).pack(pady=(3, 0))
            
            value_label = tk.Label(frame, text="--",
                                  bg='#0a0a15', fg='#00ff41',
                                  font=("Consolas", 16, "bold"))
            value_label.pack(pady=3)
            self.data_labels.append(value_label)
            
            tk.Label(frame, text=desc,
                    bg='#0a0a15', fg='#666666', font=("Microsoft YaHei", 7)).pack()
            
            if unit:
                tk.Label(frame, text=unit,
                        bg='#0a0a15', fg='#666666', font=("Microsoft YaHei", 7)).pack()
        
        # === 日志面板 ===
        log_frame = tk.LabelFrame(self.root, text="通信日志 (TX/RX)",
                                bg='#16213e', fg='white',
                                font=("Microsoft YaHei", 10), padx=10, pady=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 日志控制
        ctrl_frame = tk.Frame(log_frame, bg='#16213e')
        ctrl_frame.pack(fill='x')
        
        tk.Label(ctrl_frame, text="TX:", bg='#16213e', fg='#00aaff', font=("Consolas", 10, "bold")).pack(side='left')
        self.tx_label = tk.Label(ctrl_frame, text="0", bg='#16213e', fg='#00aaff', font=("Consolas", 10))
        self.tx_label.pack(side='left', padx=(0, 20))
        
        tk.Label(ctrl_frame, text="RX:", bg='#16213e', fg='#00ff88', font=("Consolas", 10, "bold")).pack(side='left')
        self.rx_label = tk.Label(ctrl_frame, text="0", bg='#16213e', fg='#00ff88', font=("Consolas", 10))
        self.rx_label.pack(side='left')
        
        tk.Button(ctrl_frame, text="清空日志", command=lambda: self.log_text.delete(1.0, 'end'),
                 bg='#3c3c3c', fg='white', width=10).pack(side='right')
        tk.Button(ctrl_frame, text="导出CSV", command=self.export_log,
                 bg='#3c3c3c', fg='white', width=10).pack(side='right', padx=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame,
                                                  bg='#050f05', fg='#00ff41',
                                                  font=("Consolas", 9),
                                                  wrap='word')
        self.log_text.pack(fill='both', expand=True, pady=5)
        
        # 底部状态栏
        bottom_frame = tk.Frame(self.root, bg='#0a0a15', height=25)
        bottom_frame.pack(fill='x', side='bottom')
        bottom_frame.pack_propagate(False)
        
        self.bottom_label = tk.Label(bottom_frame, text="就绪 | Modbus RTU/TCP",
                                    bg='#0a0a15', fg='gray',
                                    font=("Consolas", 8), anchor='w')
        self.bottom_label.pack(side='left', padx=10)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def flash_tx(self):
        """TX发送指示"""
        self.tx_indicator.config(fg='#00aaff')
        self.root.after(100, lambda: self.tx_indicator.config(fg='#004488'))
        self.root.after(200, lambda: self.tx_indicator.config(fg='gray'))
        
    def flash_rx(self):
        """RX接收指示"""
        self.rx_indicator.config(fg='#00ff88')
        self.root.after(100, lambda: self.rx_indicator.config(fg='#008844'))
        self.root.after(200, lambda: self.rx_indicator.config(fg='gray'))
        
    def log(self, msg, level="INFO"):
        """写日志"""
        colors = {"INFO": "#00ff41", "WARN": "#ffaa00", "ERROR": "#ff4444", 
                 "TX": "#00aaff", "RX": "#00ff88", "DATA": "#00bcd4"}
        color = colors.get(level, "#00ff41")
        
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_text.tag_config(level, foreground=color)
        self.log_text.insert('end', f"[{timestamp}] {msg}\n", level)
        self.log_text.see('end')
        
    def toggle_connection(self):
        """切换连接"""
        if self.connected:
            self.disconnect()
        else:
            self.connect()
            
    def connect(self):
        """连接PLC"""
        self.log("="*50)
        self.log("正在连接PLC...", "INFO")
        
        try:
            if self.tcp_enabled.get():
                self.connect_tcp()
            elif self.serial_enabled.get() and SERIAL_AVAILABLE:
                self.connect_serial()
            else:
                self.log("请选择连接方式 (TCP 或 Serial)", "WARN")
                messagebox.showwarning("提示", "请选择连接方式")
                return
        except Exception as e:
            self.log(f"连接失败: {e}", "ERROR")
            messagebox.showerror("连接错误", f"无法连接PLC:\n{e}")
            
    def connect_tcp(self):
        """TCP连接"""
        host = self.host_entry.get().strip()
        port = int(self.tcp_port_entry.get().strip())
        
        self.log(f"[TCP] 连接 {host}:{port}...", "INFO")
        
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.settimeout(CONFIG["tcp"]["timeout"])
        self.client.connect((host, port))
        
        self.connected = True
        self.start_time = time.time()
        
        self.connect_btn.config(text="■ 断开", bg='#aa0000')
        self.status_indicator.config(fg='green')
        self.status_text.config(text="TCP已连接", fg='green')
        
        self.log(f"[TCP] 连接成功! {host}:{port}", "INFO")
        
        recv_thread = threading.Thread(target=self.receive_loop, daemon=True)
        recv_thread.start()
        
        self.start_polling()
        
    def connect_serial(self):
        """串口连接"""
        if not SERIAL_AVAILABLE:
            self.log("[Serial] pyserial库未安装", "ERROR")
            return
            
        port = self.serial_entry.get().strip()
        baud = int(self.baud_combo.get())
        
        self.log(f"[Serial] 连接 {port}@{baud}...", "INFO")
        
        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baud,
                bytesize=8,
                parity='N',
                stopbits=1,
                timeout=CONFIG["serial"]["timeout"]
            )
        except serial.SerialException as e:
            self.log(f"[Serial] 打开串口失败: {e}", "ERROR")
            messagebox.showerror("串口错误", f"无法打开串口 {port}\n{e}")
            return
        
        self.connected = True
        self.start_time = time.time()
        
        self.connect_btn.config(text="■ 断开", bg='#aa0000')
        self.status_indicator.config(fg='green')
        self.status_text.config(text="Serial已连接", fg='green')
        
        self.log(f"[Serial] 连接成功! {port}", "INFO")
        
        recv_thread = threading.Thread(target=self.serial_receive_loop, daemon=True)
        recv_thread.start()
        
        self.start_polling()
        
    def disconnect(self):
        """断开连接"""
        self.connected = False
        
        if self.poll_timer:
            self.root.after_cancel(self.poll_timer)
            
        try:
            if self.client:
                self.client.close()
        except:
            pass
            
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
        except:
            pass
        
        self.connect_btn.config(text="▶ 连接", bg='#00aa55')
        self.status_indicator.config(fg='gray')
        self.status_text.config(text="离线", fg='gray')
        
        self.log("[断开] 连接已断开", "INFO")
        
    def receive_loop(self):
        """TCP接收循环"""
        buffer = bytearray()
        
        while self.connected:
            try:
                data = self.client.recv(256)
                if not data:
                    break
                    
                self.rx_count += 1
                self.root.after(0, lambda d=data: self.flash_rx())
                self.root.after(0, lambda b=self.rx_count: self.update_rx_count(b))
                
                buffer.extend(data)
                self.log(f"RX: {data.hex().upper()}", "RX")
                
                # 处理Modbus TCP响应
                while len(buffer) >= 9:
                    if buffer[7] == 0x03:
                        byte_count = buffer[8]
                        frame_len = 9 + byte_count
                        
                        if len(buffer) >= frame_len:
                            frame = bytes(buffer[:frame_len])
                            buffer = buffer[frame_len:]
                            self.root.after(0, lambda f=frame: self.process_response(f))
                        else:
                            break
                    else:
                        buffer.pop(0)
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.connected:
                    self.root.after(0, lambda e=e: self.log(f"[错误] 接收: {e}", "ERROR"))
                break
                
        if self.connected:
            self.root.after(0, self.disconnect)
            
    def serial_receive_loop(self):
        """串口接收循环"""
        buffer = bytearray()
        
        while self.connected and self.serial_port:
            try:
                if self.serial_port.in_waiting:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    
                    self.rx_count += 1
                    self.root.after(0, lambda: self.flash_rx())
                    self.root.after(0, lambda b=self.rx_count: self.update_rx_count(b))
                    
                    buffer.extend(data)
                    self.log(f"RX: {data.hex().upper()}", "RX")
                    
                    while len(buffer) >= 5:
                        byte_count = buffer[2]
                        frame_len = 5 + byte_count
                        
                        if len(buffer) >= frame_len:
                            frame = bytes(buffer[:frame_len])
                            buffer = buffer[frame_len:]
                            self.root.after(0, lambda f=frame: self.process_rtu_response(f))
                        else:
                            break
                            
            except Exception as e:
                if self.connected:
                    self.root.after(0, lambda e=e: self.log(f"[错误] Serial: {e}", "ERROR"))
                break
                
        if self.connected:
            self.root.after(0, self.disconnect)
            
    def process_response(self, data):
        """处理Modbus TCP响应"""
        if len(data) >= 9 and data[7] == 0x03:
            byte_count = data[8]
            
            for i in range(min(byte_count // 2, 8)):
                value = struct.unpack('>H', data[9 + i*2 : 11 + i*2])[0]
                self.reg_values[i] = value
                self.data_labels[i].config(text=str(value))
                
                if i == 6 and value > CONFIG["alarms"]["temperature_high"]:
                    self.log(f"[告警] 温度过高: {value}°C", "WARN")
                    self.data_labels[i].config(fg='red')
                elif i == 4 and value > 0:
                    self.log("[告警] 检测到告警", "WARN")
                    self.data_labels[i].config(fg='red')
                else:
                    self.data_labels[i].config(fg='#00ff41')
                    
            self.update_stats()
            
    def process_rtu_response(self, data):
        """处理Modbus RTU响应"""
        if len(data) >= 5 and data[1] == 0x03:
            byte_count = data[2]
            
            for i in range(min(byte_count // 2, 8)):
                value = struct.unpack('>H', data[3 + i*2 : 5 + i*2])[0]
                self.reg_values[i] = value
                self.data_labels[i].config(text=str(value))
                
            self.update_stats()
            
    def update_tx_count(self, count):
        self.tx_label.config(text=str(count))
        
    def update_rx_count(self, count):
        self.rx_label.config(text=str(count))
            
    def send_read_request(self):
        """发送读取请求"""
        if not self.connected:
            return
            
        self.tx_count += 1
        self.root.after(0, lambda c=self.tx_count: self.update_tx_count(c))
        self.root.after(0, self.flash_tx)
        
        if self.tcp_enabled.get():
            self.send_tcp_request()
        else:
            self.send_rtu_request()
            
    def send_tcp_request(self):
        """发送TCP请求"""
        self.transaction_id = (self.transaction_id % 65535) + 1
        
        frame = struct.pack('>HH', self.transaction_id, 0)  # Transaction ID, Protocol
        frame += struct.pack('>H', 6)  # Length
        frame += bytes([CONFIG["modbus"]["slave_id"]])  # Unit ID
        frame += bytes([0x03])  # Function: Read Holding Registers
        frame += struct.pack('>HH', CONFIG["modbus"]["start_address"], CONFIG["modbus"]["register_count"])
        
        try:
            self.client.send(frame)
            self.log(f"TX: {frame.hex().upper()}", "TX")
        except Exception as e:
            self.log(f"[错误] 发送失败: {e}", "ERROR")
            
    def send_rtu_request(self):
        """发送RTU请求"""
        if not self.serial_port:
            return
            
        slave_id = CONFIG["modbus"]["slave_id"]
        start_addr = CONFIG["modbus"]["start_address"]
        count = CONFIG["modbus"]["register_count"]
        
        frame = bytes([slave_id, 0x03])
        frame += struct.pack('>HH', start_addr, count)
        
        crc = self.calc_crc16(frame)
        frame += struct.pack('<H', crc)
        
        try:
            self.serial_port.write(frame)
            self.log(f"TX: {frame.hex().upper()}", "TX")
        except Exception as e:
            self.log(f"[错误] 发送失败: {e}", "ERROR")
            
    def calc_crc16(self, data):
        """计算CRC16"""
        crc = 0xFFFF
        for b in data:
            crc ^= b
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc
            
    def start_polling(self):
        """启动轮询"""
        self.send_read_request()
        self.poll_timer = self.root.after(CONFIG["poll_interval"], self.start_polling)
        
    def update_stats(self):
        """更新统计"""
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            uptime = f"{elapsed//60:02d}:{elapsed%60:02d}"
            conn_type = "TCP" if self.tcp_enabled.get() else "RTU"
            self.bottom_label.config(
                text=f"{conn_type} | 运行: {uptime} | TX: {self.tx_count} | RX: {self.rx_count}"
            )
            
    def export_log(self):
        """导出日志"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("文本文件", "*.txt")]
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.log_text.get(1.0, 'end'))
            self.log(f"日志已导出: {filename}", "INFO")
            
    def on_closing(self):
        """关闭窗口"""
        self.running = False
        if self.connected:
            self.disconnect()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PortHMI(root)
    root.mainloop()
