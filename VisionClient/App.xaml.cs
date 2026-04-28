using System.Windows;
using VisionClient.Views;

namespace VisionClient;

/// <summary>
/// Interaction logic for App.xaml
/// </summary>
public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        
        // 全局异常处理
        AppDomain.CurrentDomain.UnhandledException += (s, args) =>
        {
            var ex = args.ExceptionObject as Exception;
            LogError($"Unhandled Exception: {ex?.Message}\n{ex?.StackTrace}");
        };
        
        DispatcherUnhandledException += (s, args) =>
        {
            LogError($"Dispatcher Exception: {args.Exception.Message}\n{args.Exception.StackTrace}");
            args.Handled = true;
        };
        
        LogInfo("VisionClient 启动");
        
        // 显示主窗口
        var mainWindow = new MainWindow();
        mainWindow.Show();
    }
    
    private void LogInfo(string message)
    {
        var log = $"[{DateTime.Now:HH:mm:ss}] [INFO] {message}";
        System.Diagnostics.Debug.WriteLine(log);
    }
    
    private void LogError(string message)
    {
        var log = $"[{DateTime.Now:HH:mm:ss}] [ERROR] {message}";
        System.Diagnostics.Debug.WriteLine(log);
    }
}
