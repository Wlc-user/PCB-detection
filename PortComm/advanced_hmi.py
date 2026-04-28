#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PortAI 高级HMI - 使用状态机通信引擎
支持:
- 线程安全通信
- 状态机驱动
- 自动重连
- TX/RX指示
- 内存优化

运行: python advanced_hmi.py
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
from datetime import datetime
from collections import deque

from comm_engine import CommEngine, CommState


class AdvancedHMI:
    """高级HMI - 使用通信引擎"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PortAI 高级HMI - 状态机版")
        self.root.geometry("1000x700")
        self.root.configure(bg='#1a1a2e')
        
        # 通信引擎
        self.engine: CommEngine = None
        self.engine_thread: threading.Thread = None
        
        # 数据
        self.reg_values = [0] * 8
        self.reg_history = deque(maxlen=100)
        self.tx_count = 0
        self.rx_count = 0
        self.start_time = None
        
        # 状态显示
        self.state_labels = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI"""
        
        # === 标题栏 ===
        title_frame = tk.Frame(self.root, bg='#16213e', height=50)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="PortAI 高级HMI",
                font=("Microsoft YaHei", 16, "bold"),
                fg='#00ff41', bg='#16213e').pack(side='left', padx=20)
        
        # 状态指示
        self.status_indicator = tk.Label(title_frame, text="●",
                                       font=("Arial", 20), fg='gray', bg='#16213e')
        self.status_indicator.pack(side='left', padx=5)
        
        self.status_text = tk.Label(title_frame, text="离线",
                                   font=("Microsoft YaHei", 10), fg='gray', bg='#16213e')
        self.status_text.pack(side='left')
        
        # TX/RX
        tk.Label(title_frame, text="TX:", bg='#16213e', fg='white', 
                font=("Arial", 10, "bold")).pack(side='right', padx=(20, 5))
        self.tx_indicator = tk.Label(title_frame, text="●", font=("Arial", 16), 
                                    fg='gray', bg='#16213e')
        self.tx_indicator.pack(side='right')
        
        tk.Label(title_frame, text="RX:", bg='#16213e', fg='white',
                font=("Arial", 10, "bold")).pack(side='right', padx=(10, 5))
        self.rx_indicator = tk.Label(title_frame, text="●", font=("Arial", 16),
                                    fg='gray', bg='#16213e')
        self.rx_indicator.pack(side='right')
        
        # === 连接设置 ===
        conn_frame = tk.LabelFrame(self.root, text="连接设置",
                                  bg='#16213e', fg='white', padx=10, pady=10)
        conn_frame.pack(fill='x', padx=10, pady=5)
        
        # TCP设置
        tcp_frame = tk.Frame(conn_frame, bg='#16213e')
        tcp_frame.pack(fill='x')
        
        tk.Label(tcp_frame, text="IP:", bg='#16213e', fg='white').pack(side='left')
        self.host_entry = tk.Entry(tcp_frame, width=15, bg='#0a0a15', fg='#00ff41')
        self.host_entry.insert(0, "127.0.0.1")
        self.host_entry.pack(side='left', padx=5)
        
        tk.Label(tcp_frame, text="Port:", bg='#16213e', fg='white').pack(side='left', padx=(10, 5))
        self.port_entry = tk.Entry(tcp_frame, width=8, bg='#0a0a15', fg='#00ff41')
        self.port_entry.insert(0, "5000")
        self.port_entry.pack(side='left')
        
        # 连接按钮
        self.connect_btn = tk.Button(conn_frame, text="连接",
                                     command=self.toggle_connection,
                                     bg='#00aa55', fg='white', 
                                     font=("Microsoft YaHei", 11, "bold"),
                                     width=15, height=2)
        self.connect_btn.pack(pady=10)
        
        # === 状态显示 ===
        state_frame = tk.LabelFrame(self.root, text="通信状态机",
                                  bg='#16213e', fg='white', padx=10, pady=10)
        state_frame.pack(fill='x', padx=10, pady=5)
        
        states = ['IDLE', 'CONNECTING', 'CONNECTED', 'SENDING', 
                  'WAITING', 'RECEIVING', 'ERROR']
        
        for state in states:
            frame = tk.Frame(state_frame, bg='#0a0a15')
            frame.pack(side='left', padx=3, pady=3)
            
            label = tk.Label(frame, text=state, bg='#0a0a15', fg='#444444',
                           font=("Consolas", 8), width=12)
            label.pack()
            self.state_labels[state] = label
            
            indicator = tk.Label(frame, text="○", bg='#0a0a15', fg='#333333',
                               font=("Arial", 12))
            indicator.pack()
            self.state_labels[f"{state}_dot"] = indicator
            
        # === 数据面板 ===
        data_frame = tk.LabelFrame(self.root, text="数据寄存器",
                                  bg='#16213e', fg='white', padx=10, pady=10)
        data_frame.pack(fill='x', padx=10, pady=5)
        
        self.data_labels = []
        
        names = ["CRANE_ST", "CONT_POS", "TRUCK_CNT", "SHIP_DET",
                "ALARM", "SPEED", "TEMP", "WEIGHT"]
        descs = ["起重机状态", "集装箱位置", "卡车数量", "船舶检测",
                "告警", "速度", "温度", "重量"]
        units = ["", "", "辆", "", "", "km/h", "°C", "kg"]
        
        for i in range(8):
            frame = tk.Frame(data_frame, bg='#0a0a15', relief='sunken', bd=1)
            frame.pack(side='left', padx=3, pady=5, fill='both', expand=True)
            
            tk.Label(frame, text=f"{i}: {names[i]}", bg='#0a0a15', fg='#888888',
                    font=("Consolas", 8)).pack()
            
            val_label = tk.Label(frame, text="--", bg='#0a0a15', fg='#00ff41',
                                font=("Consolas", 14, "bold"))
            val_label.pack(pady=2)
            self.data_labels.append(val_label)
            
            tk.Label(frame, text=descs[i], bg='#0a0a15', fg='#666666',
                    font=("Microsoft YaHei", 7)).pack()
            
            if units[i]:
                tk.Label(frame, text=units[i], bg='#0a0a15', fg='#666666',
                        font=("Microsoft YaHei", 7)).pack()
        
        # === 日志面板 ===
        log_frame = tk.LabelFrame(self.root, text="通信日志",
                                bg='#16213e', fg='white', padx=10, pady=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 统计
        ctrl_frame = tk.Frame(log_frame, bg='#16213e')
        ctrl_frame.pack(fill='x')
        
        tk.Label(ctrl_frame, text="TX:", bg='#16213e', fg='#00aaff',
                font=("Consolas", 10)).pack(side='left')
        self.tx_label = tk.Label(ctrl_frame, text="0", bg='#16213e', fg='#00aaff',
                                font=("Consolas", 10))
        self.tx_label.pack(side='left', padx=(0, 20))
        
        tk.Label(ctrl_frame, text="RX:", bg='#16213e', fg='#00ff88',
                font=("Consolas", 10)).pack(side='left')
        self.rx_label = tk.Label(ctrl_frame, text="0", bg='#16213e', fg='#00ff88',
                                font=("Consolas", 10))
        self.rx_label.pack(side='left')
        
        tk.Button(ctrl_frame, text="清空", command=lambda: self.log_text.delete(1.0, 'end'),
                 bg='#3c3c3c', fg='white', width=8).pack(side='right')
        
        self.log_text = scrolledtext.ScrolledText(log_frame, bg='#050f05', fg='#00ff41',
                                                  font=("Consolas", 9))
        self.log_text.pack(fill='both', expand=True, pady=5)
        
        # === 底部状态栏 ===
        bottom_frame = tk.Frame(self.root, bg='#0a0a15', height=25)
        bottom_frame.pack(fill='x', side='bottom')
        bottom_frame.pack_propagate(False)
        
        self.bottom_label = tk.Label(bottom_frame, text="就绪 | 线程安全状态机",
                                    bg='#0a0a15', fg='gray',
                                    font=("Consolas", 8), anchor='w')
        self.bottom_label.pack(side='left', padx=10)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def toggle_connection(self):
        """切换连接"""
        if self.engine and self.engine.state not in [CommState.IDLE, CommState.ERROR]:
            self.disconnect()
        else:
            self.connect()
            
    def connect(self):
        """连接"""
        host = self.host_entry.get().strip()
        port = int(self.port_entry.get().strip())
        
        self.log(f"连接 {host}:{port}...")
        
        # 创建通信引擎
        self.engine = CommEngine(protocol="TCP", host=host, port=port)
        
        # 注册回调
        self.engine.on_state_change(self.on_state_change)
        self.engine.on_data(self.on_data)
        self.engine.on_error(self.on_error)
        self.engine.on_log(self.on_log)
        
        # 启动引擎线程
        self.engine.start()
        
        # 连接
        self.engine.connect()
        
        self.start_time = time.time()
        self.connect_btn.config(text="断开", bg='#aa0000')
        
    def disconnect(self):
        """断开"""
        if self.engine:
            self.engine.disconnect()
            time.sleep(0.5)
            self.engine.stop()
            self.engine = None
            
        self.connect_btn.config(text="连接", bg='#00aa55')
        self.status_indicator.config(fg='gray')
        self.status_text.config(text="离线", fg='gray')
        self.log("已断开")
        
    def on_state_change(self, old: CommState, new: CommState):
        """状态变化回调"""
        self.root.after(0, lambda: self._update_state_ui(old, new))
        
    def _update_state_ui(self, old: CommState, new: CommState):
        """更新状态UI"""
        # 更新状态指示
        if new == CommState.CONNECTED:
            self.status_indicator.config(fg='green')
            self.status_text.config(text=f"已连接", fg='green')
            self.connect_btn.config(text="断开", bg='#aa0000')
        elif new == CommState.ERROR:
            self.status_indicator.config(fg='red')
            self.status_text.config(text="错误", fg='red')
        elif new == CommState.CONNECTING:
            self.status_indicator.config(fg='yellow')
            self.status_text.config(text="连接中...", fg='yellow')
            
        # 更新状态机显示
        for state in self.state_labels:
            if not state.endswith('_dot'):
                self.state_labels[state].config(fg='#444444' if new.name != state else '#00ff41')
                self.state_labels[f"{state}_dot"].config(text="○", fg='#333333')
                
        if new.name in self.state_labels:
            self.state_labels[new.name].config(fg='#00ff41', bg='#0a2a0a')
            self.state_labels[f"{new.name}_dot"].config(text="●", fg='#00ff41')
            
        # 自动发送请求
        if new == CommState.CONNECTED:
            self.engine.send_read_request()
            
    def on_data(self, data: bytes):
        """数据回调"""
        self.root.after(0, lambda: self._process_data(data))
        
    def _process_data(self, data: bytes):
        """处理数据"""
        self.rx_count += 1
        self.rx_label.config(text=str(self.rx_count))
        
        # 闪烁RX
        self.rx_indicator.config(fg='#00ff88')
        self.root.after(100, lambda: self.rx_indicator.config(fg='gray'))
        
        # 解析Modbus
        if len(data) >= 9 and data[7] == 0x03:
            byte_count = data[8]
            
            for i in range(min(byte_count // 2, 8)):
                value = int.from_bytes(data[9+i*2:11+i*2], 'big')
                self.reg_values[i] = value
                self.data_labels[i].config(text=str(value))
                
                # 告警颜色
                if i == 6 and value > 35:
                    self.data_labels[i].config(fg='red')
                else:
                    self.data_labels[i].config(fg='#00ff41')
                    
            # 继续轮询
            if self.engine and self.engine.state == CommState.CONNECTED:
                time.sleep(0.5)
                self.engine.send_read_request()
                
        self.update_stats()
        
    def on_error(self, msg: str):
        """错误回调"""
        self.root.after(0, lambda: self.log(f"[错误] {msg}", "ERROR"))
        
    def on_log(self, msg: str):
        """日志回调"""
        if "TX:" in msg:
            self.root.after(0, lambda: self.log(msg, "TX"))
            self.root.after(0, self._flash_tx)
        elif "RX:" in msg:
            self.root.after(0, lambda: self.log(msg, "RX"))
        else:
            self.root.after(0, lambda: self.log(msg))
            
    def _flash_tx(self):
        """TX闪烁"""
        self.tx_count += 1
        self.tx_label.config(text=str(self.tx_count))
        self.tx_indicator.config(fg='#00aaff')
        self.root.after(100, lambda: self.tx_indicator.config(fg='gray'))
        
    def log(self, msg: str, level: str = "INFO"):
        """写日志"""
        colors = {"INFO": "#00ff41", "TX": "#00aaff", "RX": "#00ff88", "ERROR": "#ff4444"}
        color = colors.get(level, "#00ff41")
        
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_text.tag_config(level, foreground=color)
        self.log_text.insert('end', f"[{ts}] {msg}\n", level)
        self.log_text.see('end')
        
    def update_stats(self):
        """更新统计"""
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            uptime = f"{elapsed//60:02d}:{elapsed%60:02d}"
            self.bottom_label.config(
                text=f"状态机驱动 | 运行: {uptime} | TX: {self.tx_count} | RX: {self.rx_count}"
            )
            
    def on_closing(self):
        """关闭窗口"""
        if self.engine:
            self.engine.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedHMI(root)
    root.mainloop()
