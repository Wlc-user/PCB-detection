"""
PortAI 工业通信上位机 v2.0
Python + Tkinter 版本
支持 Modbus TCP 通信
"""
import socket
import struct
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from datetime import datetime

class PLCCommGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PortAI 工业通信上位机 v2.0")
        self.root.geometry("1000x700")
        self.root.configure(bg='#1a1a2e')
        
        self.client = None
        self.stream = None
        self.connected = False
        self.running = True
        
        self.reg_names = ['CRANE_ST', 'CONT_POS', 'TRUCK_CNT', 'SHIP_DET', 
                         'ALARM', 'SPEED', 'TEMP', 'WEIGHT']
        self.reg_units = ['', '', '辆', '', '', 'km/h', '°C', 'kg']
        self.reg_values = [0] * 8
        
        self.setup_ui()
        
        # 自动重连定时器
        self.poll_timer = None
        
    def setup_ui(self):
        # 标题栏
        title_frame = tk.Frame(self.root, bg='#16213e', height=50)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="PortAI 工业通信系统 - 上位机",
                font=("Microsoft YaHei", 16, "bold"),
                fg='#00ff41', bg='#16213e').pack(pady=12)
        
        # 左侧控制面板
        left_frame = tk.Frame(self.root, bg='#16213e', width=300)
        left_frame.pack(side='left', fill='y', padx=5, pady=5)
        left_frame.pack_propagate(False)
        
        # 连接配置
        conn_frame = tk.LabelFrame(left_frame, text="连接配置", 
                                   bg='#16213e', fg='white', font=("Microsoft YaHei", 10))
        conn_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(conn_frame, text="协议:", bg='#16213e', fg='white').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.protocol_var = tk.StringVar(value="Modbus TCP")
        ttk.Combobox(conn_frame, textvariable=self.protocol_var, 
                    values=["Modbus TCP", "自定义协议"],
                    state='readonly', width=18).grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(conn_frame, text="IP地址:", bg='#16213e', fg='white').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.host_entry = tk.Entry(conn_frame, width=20)
        self.host_entry.insert(0, "127.0.0.1")
        self.host_entry.grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(conn_frame, text="端口:", bg='#16213e', fg='white').grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.port_entry = tk.Entry(conn_frame, width=20)
        self.port_entry.insert(0, "5000")
        self.port_entry.grid(row=2, column=1, padx=5, pady=5)
        
        self.connect_btn = tk.Button(conn_frame, text="连接 Connect",
                                      bg='#00aa55', fg='white', font=("Microsoft YaHei", 11),
                                      command=self.toggle_connection, width=20)
        self.connect_btn.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.status_label = tk.Label(conn_frame, text="[待机] 未连接",
                                       bg='#0a0a15', fg='gray', font=("Consolas", 9),
                                       width=25, height=2)
        self.status_label.grid(row=4, column=0, columnspan=2, pady=5)
        
        # 数据寄存器面板
        data_frame = tk.LabelFrame(left_frame, text="PLC 数据寄存器",
                                    bg='#16213e', fg='#00ff41', font=("Microsoft YaHei", 10))
        data_frame.pack(fill='x', padx=10, pady=10)
        
        self.data_labels = []
        for i in range(8):
            row_frame = tk.Frame(data_frame, bg='#0f0f1e')
            row_frame.pack(fill='x', padx=5, pady=2)
            
            name_label = tk.Label(row_frame, text=f"{self.reg_names[i]} [{i}]",
                                   bg='#0f0f1e', fg='#00bcd4', font=("Consolas", 8),
                                   width=12, anchor='w')
            name_label.pack(side='left')
            
            value_label = tk.Label(row_frame, text="--",
                                   bg='#0f0f1e', fg='white', font=("Consolas", 10, "bold"),
                                   width=8)
            value_label.pack(side='left')
            
            unit_label = tk.Label(row_frame, text=self.reg_units[i],
                                   bg='#0f0f1e', fg='gray', font=("Consolas", 8),
                                   width=6)
            unit_label.pack(side='left')
            
            self.data_labels.append(value_label)
        
        # 发送命令面板
        send_frame = tk.LabelFrame(left_frame, text="发送命令",
                                    bg='#16213e', fg='white', font=("Microsoft YaHei", 10))
        send_frame.pack(fill='x', padx=10, pady=10)
        
        self.send_entry = tk.Entry(send_frame, font=("Consolas", 10))
        self.send_entry.pack(fill='x', padx=5, pady=5)
        
        tk.Button(send_frame, text="发送 Send", bg='#0f3460', fg='white',
                  command=self.send_command).pack(pady=5)
        
        # 右侧日志面板
        right_frame = tk.Frame(self.root, bg='#0a0a15')
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        log_title_frame = tk.Frame(right_frame, bg='#0a0a15')
        log_title_frame.pack(fill='x')
        
        tk.Label(log_title_frame, text="通信日志 (Communication Log)",
                bg='#0a0a15', fg='white', font=("Microsoft YaHei", 11)).pack(side='left')
        
        tk.Button(log_title_frame, text="清空", bg='#3c3c3c', fg='white',
                 command=lambda: self.log_text.delete(1.0, 'end'),
                 width=8).pack(side='right')
        
        self.log_text = scrolledtext.ScrolledText(right_frame,
                                                    bg='#050f05', fg='#00ff41',
                                                    font=("Consolas", 9),
                                                    wrap='word')
        self.log_text.pack(fill='both', expand=True, pady=5)
        
        # 窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert('end', f"[{timestamp}] {msg}\n")
        self.log_text.see('end')
        
    def toggle_connection(self):
        if self.connected:
            self.disconnect()
        else:
            self.connect()
            
    def connect(self):
        host = self.host_entry.get().strip()
        port = int(self.port_entry.get().strip())
        
        self.log(f"[INFO] 正在连接 {host}:{port}...")
        
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.settimeout(5)
            self.client.connect((host, port))
            
            self.connected = True
            self.connect_btn.config(text="断开 Disconnect", bg='#aa0000')
            self.status_label.config(text="[已连接] Connected", fg='#00ff41')
            
            self.log("[OK] 连接成功!")
            
            # 启动接收线程
            recv_thread = threading.Thread(target=self.receive_loop, daemon=True)
            recv_thread.start()
            
            # 启动轮询
            self.start_polling()
            
        except Exception as e:
            self.log(f"[ERROR] 连接失败: {e}")
            messagebox.showerror("连接错误", f"无法连接到PLC:\n{e}")
            
    def disconnect(self):
        self.connected = False
        self.running = False
        
        if self.poll_timer:
            self.root.after_cancel(self.poll_timer)
            
        try:
            if self.client:
                self.client.close()
        except:
            pass
            
        self.connect_btn.config(text="连接 Connect", bg='#00aa55')
        self.status_label.config(text="[断开] Disconnected", fg='red')
        self.log("[INFO] 已断开连接")
        
    def receive_loop(self):
        buffer = bytearray()
        
        while self.connected and self.running:
            try:
                data = self.client.recv(256)
                if not data:
                    break
                    
                buffer.extend(data)
                
                # 处理完整帧
                while len(buffer) >= 9:
                    if buffer[7] == 0x03:  # Read Holding Registers
                        byte_count = buffer[8]
                        frame_len = 9 + byte_count
                        
                        if len(buffer) >= frame_len:
                            frame = bytes(buffer[:frame_len])
                            buffer = buffer[frame_len:]
                            
                            self.root.after(0, lambda f=frame: self.parse_response(f))
                        else:
                            break
                    else:
                        buffer.pop(0)
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.connected:
                    self.root.after(0, lambda e=e: self.log(f"[ERROR] {e}"))
                break
                
        if self.connected:
            self.root.after(0, self.disconnect)
            
    def parse_response(self, data):
        self.log(f"[RECV] {data.hex().upper()}")
        
        if len(data) >= 9 and data[7] == 0x03:
            byte_count = data[8]
            
            for i in range(min(byte_count // 2, 8)):
                value = struct.unpack('>H', data[9 + i*2 : 11 + i*2])[0]
                self.reg_values[i] = value
                self.data_labels[i].config(text=str(value))
                
                if value > 0:
                    self.data_labels[i].config(fg='#00ff41')
                else:
                    self.data_labels[i].config(fg='gray')
                    
            # 更新告警状态
            if self.reg_values[4] > 0:  # ALARM
                self.status_label.config(text="[告警] ALARM!", fg='red')
                
    def send_read_request(self):
        if not self.connected:
            return
            
        # Modbus TCP Read Holding Registers
        frame = bytes([
            0x00, 0x01,  # Transaction ID
            0x00, 0x00,  # Protocol ID
            0x00, 0x06,  # Length
            0x01,        # Unit ID
            0x03,        # Function: Read Holding Registers
            0x00, 0x00,  # Start Address
            0x00, 0x08   # Quantity: 8 registers
        ])
        
        try:
            self.client.send(frame)
            self.log("[SEND] Read 8 Holding Registers")
        except Exception as e:
            self.log(f"[ERROR] {e}")
            
    def start_polling(self):
        if self.connected:
            self.send_read_request()
            self.poll_timer = self.root.after(500, self.start_polling)
            
    def send_command(self):
        if not self.connected:
            return
            
        cmd = self.send_entry.get().strip()
        if not cmd:
            return
            
        try:
            self.client.send((cmd + "\n").encode())
            self.log(f"[SEND] {cmd}")
            self.send_entry.delete(0, 'end')
        except Exception as e:
            self.log(f"[ERROR] {e}")
            
    def on_closing(self):
        self.running = False
        self.disconnect()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PLCCommGUI(root)
    root.mainloop()
