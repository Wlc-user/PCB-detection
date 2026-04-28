using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Net.WebSockets;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Windows;
using System.Windows.Media.Imaging;
using Microsoft.Win32;

namespace VisionClient.ViewModels;

/// <summary>
/// 主视图模型
/// </summary>
public class MainViewModel : INotifyPropertyChanged, IDisposable
{
    #region 属性
    
    private bool _isConnected;
    public bool IsConnected
    {
        get => _isConnected;
        set { _isConnected = value; OnPropertyChanged(); UpdateConnectButtonText(); }
    }
    
    private bool _isConnecting;
    public bool IsConnecting
    {
        get => _isConnecting;
        set { _isConnecting = value; OnPropertyChanged(); UpdateConnectButtonText(); }
    }
    
    private string _connectionStatus = "未连接";
    public string ConnectionStatus
    {
        get => _connectionStatus;
        set { _connectionStatus = value; OnPropertyChanged(); }
    }
    
    private string _connectButtonText = "连接";
    public string ConnectButtonText
    {
        get => _connectButtonText;
        set { _connectButtonText = value; OnPropertyChanged(); }
    }
    
    private string _fps = "0";
    public string Fps
    {
        get => _fps;
        set { _fps = value; OnPropertyChanged(); }
    }
    
    private string _latency = "0";
    public string Latency
    {
        get => _latency;
        set { _latency = value; OnPropertyChanged(); }
    }
    
    private string _detectionCount = "0";
    public string DetectionCount
    {
        get => _detectionCount;
        set { _detectionCount = value; OnPropertyChanged(); }
    }
    
    private float _confidenceThreshold = 0.25f;
    public float ConfidenceThreshold
    {
        get => _confidenceThreshold;
        set { _confidenceThreshold = value; }
    }
    
    private float _iouThreshold = 0.45f;
    public float IouThreshold
    {
        get => _iouThreshold;
        set { _iouThreshold = value; }
    }
    
    private BitmapSource? _currentImage;
    public BitmapSource? CurrentImage
    {
        get => _currentImage;
        set { _currentImage = value; OnPropertyChanged(); }
    }
    
    public event Action<BitmapSource>? ImageUpdated;
    public event Action<string>? StatusUpdated;
    public event Action<List<DetectionResult>>? DetectionUpdated;
    
    #endregion
    
    #region 命令
    
    public RelayCommand ConnectCommand { get; }
    public RelayCommand OpenImageCommand { get; }
    public RelayCommand OpenVideoCommand { get; }
    public RelayCommand StartCameraCommand { get; }
    public RelayCommand SnapshotCommand { get; }
    
    #endregion
    
    #region 私有字段
    
    private ClientWebSocket? _wsClient;
    private CancellationTokenSource? _cts;
    private readonly string _wsServer = "ws://localhost:8765";
    private readonly Stopwatch _stopwatch = new();
    private int _frameCount;
    private string _protocol = "websocket";
    
    #endregion
    
    public MainViewModel()
    {
        ConnectCommand = new RelayCommand(ExecuteConnect, () => !IsConnected && !IsConnecting);
        OpenImageCommand = new RelayCommand(ExecuteOpenImage);
        OpenVideoCommand = new RelayCommand(ExecuteOpenVideo);
        StartCameraCommand = new RelayCommand(ExecuteStartCamera);
        SnapshotCommand = new RelayCommand(ExecuteSnapshot);
        
        Log("VisionClient ViewModel 初始化完成");
    }
    
    private void UpdateConnectButtonText()
    {
        if (IsConnecting)
            ConnectButtonText = "连接中...";
        else if (IsConnected)
            ConnectButtonText = "断开";
        else
            ConnectButtonText = "连接";
    }
    
    private void Log(string message)
    {
        System.Diagnostics.Debug.WriteLine($"[ViewModel] {message}");
    }
    
    #region 命令实现
    
    private async void ExecuteConnect()
    {
        if (IsConnected)
        {
            await DisconnectAsync();
        }
        else
        {
            await ConnectAsync();
        }
    }
    
    private async Task ConnectAsync()
    {
        try
        {
            IsConnecting = true;
            StatusUpdated?.Invoke("正在连接服务器...");
            Log($"尝试连接 {_protocol} 服务器...");
            
            await ConnectWebSocketAsync();
            
            IsConnected = true;
            ConnectionStatus = "已连接";
            StatusUpdated?.Invoke("WebSocket 连接成功");
            Log("WebSocket 连接成功");
        }
        catch (Exception ex)
        {
            Log($"连接失败: {ex.Message}");
            StatusUpdated?.Invoke($"连接失败: {ex.Message}");
            MessageBox.Show($"连接失败: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            IsConnecting = false;
        }
    }
    
    private async Task DisconnectAsync()
    {
        try
        {
            Log("正在断开连接...");
            
            if (_wsClient != null)
            {
                if (_wsClient.State == WebSocketState.Open)
                    await _wsClient.CloseAsync(WebSocketCloseStatus.NormalClosure, "Client closing", CancellationToken.None);
                _wsClient.Dispose();
                _wsClient = null;
            }
            
            _cts?.Cancel();
            _cts?.Dispose();
            _cts = null;
            
            IsConnected = false;
            ConnectionStatus = "未连接";
            StatusUpdated?.Invoke("已断开连接");
            Log("断开连接成功");
        }
        catch (Exception ex)
        {
            Log($"断开连接时出错: {ex.Message}");
        }
    }
    
    private async Task ConnectWebSocketAsync()
    {
        Log($"初始化 WebSocket: {_wsServer}");
        
        _wsClient = new ClientWebSocket();
        _cts = new CancellationTokenSource();
        
        await _wsClient.ConnectAsync(new Uri(_wsServer), _cts.Token);
        
        Log("WebSocket 已连接，开始监听消息...");
        
        // 启动接收消息的任务
        _ = Task.Run(ReceiveWebSocketMessagesAsync);
    }
    
    private async Task ReceiveWebSocketMessagesAsync()
    {
        var buffer = new byte[1024 * 1024]; // 1MB buffer
        
        try
        {
            while (_wsClient?.State == WebSocketState.Open && _cts?.IsCancellationRequested == false)
            {
                var result = await _wsClient.ReceiveAsync(new ArraySegment<byte>(buffer), _cts.Token);
                
                if (result.MessageType == WebSocketMessageType.Close)
                {
                    Log("服务器关闭连接");
                    break;
                }
                
                if (result.MessageType == WebSocketMessageType.Binary)
                {
                    var data = buffer.Take(result.Count).ToArray();
                    ProcessResultMessage(data);
                }
                else if (result.MessageType == WebSocketMessageType.Text)
                {
                    var text = Encoding.UTF8.GetString(buffer, 0, result.Count);
                    Log($"收到文本消息: {text}");
                }
            }
        }
        catch (OperationCanceledException)
        {
            Log("WebSocket 接收任务取消");
        }
        catch (Exception ex)
        {
            Log($"WebSocket 接收错误: {ex.Message}");
        }
    }
    
    private void ProcessResultMessage(byte[] data)
    {
        try
        {
            // 解析 JSON 结果
            var json = Encoding.UTF8.GetString(data);
            var result = JsonSerializer.Deserialize<DetectResult>(json);
            
            if (result != null)
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    DetectionCount = result.Detections?.Count.ToString() ?? "0";
                    
                    var detections = result.Detections?.Select(d => new DetectionResult
                    {
                        Label = d.Class,
                        Confidence = $"{d.Confidence:P0}",
                        Box = new Rect(d.X, d.Y, d.Width, d.Height)
                    }).ToList() ?? new List<DetectionResult>();
                    
                    DetectionUpdated?.Invoke(detections);
                });
            }
        }
        catch (Exception ex)
        {
            Log($"解析结果失败: {ex.Message}");
        }
    }
    
    private void ExecuteOpenImage()
    {
        var dialog = new OpenFileDialog
        {
            Filter = "图像文件|*.jpg;*.jpeg;*.png;*.bmp;*.gif|所有文件|*.*",
            Title = "选择图像"
        };
        
        if (dialog.ShowDialog() == true)
        {
            ProcessImageFile(dialog.FileName);
        }
    }
    
    private void ExecuteOpenVideo()
    {
        var dialog = new OpenFileDialog
        {
            Filter = "视频文件|*.mp4;*.avi;*.mov;*.mkv|所有文件|*.*",
            Title = "选择视频"
        };
        
        if (dialog.ShowDialog() == true)
        {
            ProcessVideoFile(dialog.FileName);
        }
    }
    
    private void ExecuteStartCamera()
    {
        // TODO: 实现摄像头捕获
        StatusUpdated?.Invoke("摄像头功能开发中...");
    }
    
    private void ExecuteSnapshot()
    {
        if (CurrentImage != null)
        {
            var dialog = new SaveFileDialog
            {
                Filter = "PNG|*.png|JPEG|*.jpg",
                FileName = $"snapshot_{DateTime.Now:yyyyMMdd_HHmmss}.png"
            };
            
            if (dialog.ShowDialog() == true)
            {
                SaveBitmap(CurrentImage, dialog.FileName);
                StatusUpdated?.Invoke($"截图已保存: {dialog.FileName}");
            }
        }
    }
    
    #endregion
    
    #region 图像处理
    
    public void ProcessImageFile(string filePath)
    {
        try
        {
            Log($"处理图像: {filePath}");
            StatusUpdated?.Invoke($"处理图像: {Path.GetFileName(filePath)}");
            
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.UriSource = new Uri(filePath);
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.EndInit();
            bitmap.Freeze();
            
            CurrentImage = bitmap;
            ImageUpdated?.Invoke(bitmap);
            
            // 发送到服务器
            _ = SendToServerAsync(File.ReadAllBytes(filePath));
        }
        catch (Exception ex)
        {
            Log($"处理图像失败: {ex.Message}");
            StatusUpdated?.Invoke($"错误: {ex.Message}");
        }
    }
    
    public void ProcessVideoFile(string filePath)
    {
        // TODO: 实现视频处理
        StatusUpdated?.Invoke($"视频处理: {Path.GetFileName(filePath)}");
    }
    
    private async Task SendToServerAsync(byte[] imageData)
    {
        if (!IsConnected)
        {
            Log("未连接服务器，跳过发送");
            return;
        }
        
        _stopwatch.Restart();
        _frameCount++;
        
        try
        {
            if (_wsClient?.State == WebSocketState.Open)
            {
                await SendWebSocketAsync(imageData);
            }
            
            _stopwatch.Stop();
            var latency = _stopwatch.ElapsedMilliseconds;
            Latency = latency.ToString();
            
            // 更新 FPS
            Fps = _frameCount.ToString();
            
            Log($"发送成功, 延迟: {latency}ms");
        }
        catch (Exception ex)
        {
            Log($"发送失败: {ex.Message}");
        }
    }
    
    private async Task SendWebSocketAsync(byte[] imageData)
    {
        var message = new
        {
            type = "image",
            data = Convert.ToBase64String(imageData),
            confidence = ConfidenceThreshold,
            iou = IouThreshold
        };
        
        var json = JsonSerializer.Serialize(message);
        var bytes = Encoding.UTF8.GetBytes(json);
        
        await _wsClient!.SendAsync(
            new ArraySegment<byte>(bytes),
            WebSocketMessageType.Text,
            true,
            _cts!.Token
        );
    }
    
    private void SaveBitmap(BitmapSource bitmap, string filePath)
    {
        BitmapEncoder encoder;
        var ext = Path.GetExtension(filePath).ToLower();
        
        encoder = ext switch
        {
            ".jpg" or ".jpeg" => new JpegBitmapEncoder { QualityLevel = 95 },
            ".png" => new PngBitmapEncoder(),
            ".bmp" => new BmpBitmapEncoder(),
            _ => new PngBitmapEncoder()
        };
        
        encoder.Frames.Add(BitmapFrame.Create(bitmap));
        
        using var stream = File.Create(filePath);
        encoder.Save(stream);
        
        Log($"截图已保存: {filePath}");
    }
    
    #endregion
    
    #region INotifyPropertyChanged
    
    public event PropertyChangedEventHandler? PropertyChanged;
    
    protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
    
    #endregion
    
    #region IDisposable
    
    public void Dispose()
    {
        DisconnectAsync().Wait();
        GC.SuppressFinalize(this);
    }
    
    #endregion
}

#region 辅助类

public class RelayCommand : System.Windows.Input.ICommand
{
    private readonly Action _execute;
    private readonly Func<bool>? _canExecute;
    
    public RelayCommand(Action execute, Func<bool>? canExecute = null)
    {
        _execute = execute;
        _canExecute = canExecute;
    }
    
    public event EventHandler? CanExecuteChanged
    {
        add => System.Windows.Input.CommandManager.RequerySuggested += value;
        remove => System.Windows.Input.CommandManager.RequerySuggested -= value;
    }
    
    public bool CanExecute(object? parameter) => _canExecute?.Invoke() ?? true;
    public void Execute(object? parameter) => _execute();
}

public class DetectResult
{
    public List<DetectItem>? Detections { get; set; }
}

public class DetectItem
{
    public string Class { get; set; } = "";
    public float Confidence { get; set; }
    public double X { get; set; }
    public double Y { get; set; }
    public double Width { get; set; }
    public double Height { get; set; }
}

public class DetectionResult
{
    public string Label { get; set; } = "";
    public string Confidence { get; set; } = "";
    public Rect Box { get; set; }
}

#endregion
