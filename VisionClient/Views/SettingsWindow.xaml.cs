using System.Windows;

namespace VisionClient.Views;

public partial class SettingsWindow : Window
{
    public SettingsWindow()
    {
        InitializeComponent();
        
        // 加载当前配置
        LoadSettings();
    }
    
    private void LoadSettings()
    {
        var appSettings = ConfigurationManager.AppSettings;
        
        GrpcServerBox.Text = appSettings["GrpcServer"] ?? "localhost:50051";
        WsServerBox.Text = appSettings["WsServer"] ?? "ws://localhost:8765";
        
        if (double.TryParse(appSettings["ConfidenceThreshold"], out var conf))
            ConfidenceSlider.Value = conf;
        
        if (double.TryParse(appSettings["IoUThreshold"], out var iou))
            IoUSlider.Value = iou;
        
        ShowLabelsCheck.IsChecked = appSettings["ShowLabels"] != "false";
        ShowConfidenceCheck.IsChecked = appSettings["ShowConfidence"] != "false";
        DrawBoxesCheck.IsChecked = appSettings["DrawBoxes"] != "false";
    }
    
    private void Save_Click(object sender, RoutedEventArgs e)
    {
        // 保存配置（实际项目中应该写入配置文件）
        DialogResult = true;
        Close();
    }
    
    private void Cancel_Click(object sender, RoutedEventArgs e)
    {
        DialogResult = false;
        Close();
    }
}

// 需要添加 System.Configuration.ConfigurationManager 包
public static class ConfigurationManager
{
    private static readonly Dictionary<string, string> _settings = new()
    {
        ["GrpcServer"] = "localhost:50051",
        ["WsServer"] = "ws://localhost:8765",
        ["ConfidenceThreshold"] = "0.25",
        ["IoUThreshold"] = "0.45",
        ["ShowLabels"] = "true",
        ["ShowConfidence"] = "true",
        ["DrawBoxes"] = "true"
    };
    
    public static NameValueCollection AppSettings => new(_settings);
}

public class NameValueCollection : Dictionary<string, string>
{
    public NameValueCollection(Dictionary<string, string> dict) : base(dict) { }
    
    public new string? this[string key] => TryGetValue(key, out var value) ? value : null;
}
