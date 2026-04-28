/*
 * PortAI 工业通信上位机 v2.0
 * 简化版 - 仅TCP通信
 */
using System;
using System.Drawing;
using System.Net.Sockets;
using System.Text;
using System.Windows.Forms;

class PortCommApp : Form
{
    private ComboBox cmbProtocol;
    private TextBox txtHost, txtPort;
    private TextBox txtLog;
    private TextBox txtSend;
    private Button btnConnect, btnSend;
    private Label lblStatus;
    private Label[] dataLabels;
    
    private TcpClient client;
    private NetworkStream stream;
    private bool connected;
    private short[] regValues = new short[8];
    
    public PortCommApp()
    {
        InitializeComponent();
        this.Text = "PortAI 工业通信上位机 v2.0";
        this.StartPosition = FormStartPosition.CenterScreen;
    }
    
    void InitializeComponent()
    {
        this.Size = new Size(900, 600);
        this.BackColor = Color.FromArgb(26, 26, 46);
        
        // 标题
        Label title = new Label()
        {
            Text = "PortAI 工业通信系统 - 上位机",
            Font = new Font("Microsoft YaHei", 16, FontStyle.Bold),
            ForeColor = Color.Lime,
            BackColor = Color.FromArgb(22, 33, 62),
            Location = new Point(0, 0),
            Size = new Size(900, 50),
            TextAlign = ContentAlignment.MiddleCenter
        };
        
        // 左侧配置面板
        Panel leftPanel = new Panel()
        {
            Location = new Point(10, 60),
            Size = new Size(280, 490),
            BackColor = Color.FromArgb(22, 33, 62)
        };
        
        Label protoLbl = new Label() { Text = "协议:", ForeColor = Color.White, Location = new Point(10, 15), Font = new Font("Microsoft YaHei", 10) };
        cmbProtocol = new ComboBox()
        {
            Location = new Point(70, 12),
            Width = 190,
            DropDownStyle = ComboBoxStyle.DropDownList,
            Font = new Font("Microsoft YaHei", 10)
        };
        cmbProtocol.Items.AddRange(new object[] { "Modbus TCP", "自定义协议" });
        cmbProtocol.SelectedIndex = 0;
        
        Label hostLbl = new Label() { Text = "IP地址:", ForeColor = Color.White, Location = new Point(10, 50), Font = new Font("Microsoft YaHei", 10) };
        txtHost = new TextBox() { Location = new Point(70, 47), Width = 190, Text = "127.0.0.1", Font = new Font("Microsoft YaHei", 10) };
        
        Label portLbl = new Label() { Text = "端口:", ForeColor = Color.White, Location = new Point(10, 85), Font = new Font("Microsoft YaHei", 10) };
        txtPort = new TextBox() { Location = new Point(70, 82), Width = 190, Text = "5000", Font = new Font("Microsoft YaHei", 10) };
        
        btnConnect = new Button()
        {
            Text = "连接 Connect",
            Location = new Point(10, 125),
            Size = new Size(250, 40),
            BackColor = Color.FromArgb(0, 170, 85),
            ForeColor = Color.White,
            FlatStyle = FlatStyle.Flat,
            Font = new Font("Microsoft YaHei", 11)
        };
        btnConnect.Click += BtnConnect_Click;
        
        lblStatus = new Label()
        {
            Text = "[待机] 未连接",
            Location = new Point(10, 175),
            Size = new Size(250, 30),
            ForeColor = Color.Gray,
            BackColor = Color.FromArgb(10, 10, 21),
            TextAlign = ContentAlignment.MiddleCenter,
            Font = new Font("Microsoft YaHei", 10)
        };
        
        Label sep = new Label() { Text = "─────────────────────", ForeColor = Color.Gray, Location = new Point(10, 215) };
        
        Label sendLbl = new Label() { Text = "发送命令:", ForeColor = Color.White, Location = new Point(10, 245), Font = new Font("Microsoft YaHei", 10) };
        txtSend = new TextBox() { Location = new Point(10, 275), Size = new Size(250, 30), Font = new Font("Consolas", 10) };
        
        btnSend = new Button()
        {
            Text = "发送 Send",
            Location = new Point(10, 315),
            Size = new Size(250, 35),
            BackColor = Color.FromArgb(15, 52, 96),
            ForeColor = Color.White,
            FlatStyle = FlatStyle.Flat,
            Font = new Font("Microsoft YaHei", 10),
            Enabled = false
        };
        btnSend.Click += BtnSend_Click;
        
        // 数据面板
        Label dataLbl = new Label() { Text = "PLC 数据寄存器", ForeColor = Color.Lime, Location = new Point(10, 365), Font = new Font("Microsoft YaHei", 11, FontStyle.Bold) };
        
        string[] names = { "CRANE_ST [0]", "CONT_POS [1]", "TRUCK_CNT [2]", "SHIP_DET [3]", "ALARM [4]", "SPEED [5]", "TEMP [6]", "WEIGHT [7]" };
        string[] units = { "", "", "辆", "", "", "km/h", "°C", "kg" };
        
        dataLabels = new Label[8];
        for (int i = 0; i < 8; i++)
        {
            Panel row = new Panel() 
            { 
                Location = new Point(0, 395 + i * 22), 
                Size = new Size(280, 22), 
                BackColor = Color.FromArgb(15, 15, 30) 
            };
            
            Label n = new Label() 
            { 
                Text = names[i], 
                Location = new Point(10, 3), 
                Size = new Size(110, 16), 
                ForeColor = Color.Cyan,
                Font = new Font("Consolas", 8)
            };
            
            dataLabels[i] = new Label() 
            { 
                Text = "--", 
                Location = new Point(125, 3), 
                Size = new Size(80, 16), 
                ForeColor = Color.White, 
                Font = new Font("Consolas", 10, FontStyle.Bold) 
            };
            
            Label u = new Label()
            {
                Text = units[i],
                Location = new Point(210, 3),
                Size = new Size(60, 16),
                ForeColor = Color.Gray,
                Font = new Font("Consolas", 8)
            };
            
            row.Controls.AddRange(new Control[] { n, dataLabels[i], u });
            leftPanel.Controls.Add(row);
        }
        
        leftPanel.Controls.AddRange(new Control[] { 
            protoLbl, cmbProtocol, hostLbl, txtHost, portLbl, txtPort, 
            btnConnect, lblStatus, sep, sendLbl, txtSend, btnSend, dataDbl 
        });
        
        // 右侧日志
        Panel rightPanel = new Panel()
        {
            Location = new Point(300, 60),
            Size = new Size(585, 490),
            BackColor = Color.FromArgb(10, 10, 21)
        };
        
        Label logTitle = new Label() 
        { 
            Text = "通信日志 (Communication Log)", 
            ForeColor = Color.White, 
            Location = new Point(10, 10), 
            Font = new Font("Microsoft YaHei", 11, FontStyle.Bold) 
        };
        
        Button btnClear = new Button()
        {
            Text = "清空",
            Location = new Point(480, 8),
            Size = new Size(90, 28),
            BackColor = Color.FromArgb(60, 60, 60),
            ForeColor = Color.White,
            FlatStyle = FlatStyle.Flat
        };
        btnClear.Click += (s, e) => txtLog.Clear();
        
        txtLog = new TextBox()
        {
            Location = new Point(10, 40),
            Size = new Size(565, 440),
            Multiline = true,
            ReadOnly = true,
            BackColor = Color.FromArgb(5, 15, 5),
            ForeColor = Color.Lime,
            Font = new Font("Consolas", 9),
            ScrollBars = ScrollBars.Vertical
        };
        
        rightPanel.Controls.AddRange(new Control[] { logTitle, btnClear, txtLog });
        
        this.Controls.AddRange(new Control[] { title, leftPanel, rightPanel });
    }
    
    void BtnConnect_Click(object sender, EventArgs e)
    {
        if (connected) Disconnect();
        else Connect();
    }
    
    void Connect()
    {
        try
        {
            string host = txtHost.Text;
            int port = int.Parse(txtPort.Text);
            
            Log($"[INFO] 正在连接 {host}:{port}...");
            
            client = new TcpClient();
            client.Connect(host, port);
            stream = client.GetStream();
            connected = true;
            
            btnConnect.Text = "断开 Disconnect";
            btnConnect.BackColor = Color.FromArgb(170, 0, 0);
            btnSend.Enabled = true;
            lblStatus.Text = "[已连接] Connected";
            lblStatus.ForeColor = Color.Lime;
            
            Log("[OK] 连接成功!");
            
            System.Threading.Thread recvThread = new System.Threading.Thread(ReceiveLoop);
            recvThread.IsBackground = true;
            recvThread.Start();
            
            SendReadRequest();
            
            System.Threading.Timer timer = new System.Threading.Timer(_ => {
                if (connected) SendReadRequest();
            }, null, 1000, 500);
        }
        catch (Exception ex)
        {
            Log($"[ERROR] 连接失败: {ex.Message}");
        }
    }
    
    void Disconnect()
    {
        connected = false;
        try { stream?.Close(); } catch { }
        try { client?.Close(); } catch { }
        
        btnConnect.Text = "连接 Connect";
        btnConnect.BackColor = Color.FromArgb(0, 170, 85);
        btnSend.Enabled = false;
        lblStatus.Text = "[断开] Disconnected";
        lblStatus.ForeColor = Color.Red;
        
        Log("[INFO] 已断开连接");
    }
    
    void ReceiveLoop()
    {
        byte[] buffer = new byte[256];
        
        while (connected)
        {
            try
            {
                if (stream.DataAvailable)
                {
                    int len = stream.Read(buffer, 0, buffer.Length);
                    if (len > 0)
                    {
                        string hex = BitConverter.ToString(buffer, 0, len);
                        this.Invoke(new Action(() =>
                        {
                            Log($"[RECV] {hex}");
                            ParseData(buffer, len);
                        }));
                    }
                }
                System.Threading.Thread.Sleep(50);
            }
            catch { break; }
        }
    }
    
    void SendReadRequest()
    {
        if (!connected) return;
        
        // Modbus TCP Read Holding Registers
        byte[] frame = new byte[12];
        frame[0] = 0x00; frame[1] = 0x01;  // Transaction ID
        frame[2] = 0x00; frame[3] = 0x00;  // Protocol ID
        frame[4] = 0x00; frame[5] = 0x06;  // Length
        frame[6] = 0x01;                   // Unit ID
        frame[7] = 0x03;                   // Function: Read Holding Registers
        frame[8] = 0x00; frame[9] = 0x00;  // Start Address
        frame[10] = 0x00; frame[11] = 0x08; // Quantity: 8 registers
        
        try
        {
            stream.Write(frame, 0, frame.Length);
            Log("[SEND] Read 8 Holding Registers (01 03 00 00 00 08)");
        }
        catch (Exception ex)
        {
            this.Invoke(new Action(() => Log($"[ERROR] {ex.Message}")));
        }
    }
    
    void ParseData(byte[] data, int len)
    {
        // 简单解析: 假设返回格式是文本 "REG:val1,val2,..."
        string text = Encoding.ASCII.GetString(data, 0, len).Trim();
        
        if (text.StartsWith("REG:"))
        {
            string[] parts = text.Substring(4).Split(',');
            for (int i = 0; i < Math.Min(parts.Length, 8); i++)
            {
                if (short.TryParse(parts[i], out short val))
                {
                    regValues[i] = val;
                    dataLabels[i].Text = val.ToString();
                    dataLabels[i].ForeColor = val > 0 ? Color.Lime : Color.Gray;
                }
            }
            return;
        }
        
        // Modbus TCP 响应解析
        if (len >= 9 && data[7] == 0x03)
        {
            int byteCount = data[8];
            for (int i = 0; i < byteCount / 2 && i < 8; i++)
            {
                short value = (short)((data[9 + i * 2] << 8) | data[10 + i * 2]);
                regValues[i] = value;
                dataLabels[i].Text = value.ToString();
                dataLabels[i].ForeColor = value > 0 ? Color.Lime : Color.Gray;
            }
        }
    }
    
    void BtnSend_Click(object sender, EventArgs e)
    {
        if (!connected) return;
        
        string cmd = txtSend.Text.Trim();
        if (string.IsNullOrEmpty(cmd)) return;
        
        try
        {
            byte[] data = Encoding.ASCII.GetBytes(cmd + "\n");
            stream.Write(data, 0, data.Length);
            Log($"[SEND] {cmd}");
            txtSend.Clear();
        }
        catch (Exception ex)
        {
            Log($"[ERROR] {ex.Message}");
        }
    }
    
    void Log(string msg)
    {
        txtLog.AppendText($"[{DateTime.Now:HH:mm:ss}] {msg}\n");
        txtLog.SelectionStart = txtLog.Text.Length;
        txtLog.ScrollToCaret();
    }
    
    protected override void OnFormClosing(FormClosingEventArgs e)
    {
        Disconnect();
        base.OnFormClosing(e);
    }
    
    [System.STAThread]
    static void Main()
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.Run(new PortCommApp());
    }
}
