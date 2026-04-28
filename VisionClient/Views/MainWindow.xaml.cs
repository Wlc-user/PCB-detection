using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using VisionClient.ViewModels;

namespace VisionClient.Views;

/// <summary>
/// MainWindow.xaml 的交互逻辑
/// </summary>
public partial class MainWindow : Window
{
    private readonly MainViewModel _viewModel;
    
    public MainWindow()
    {
        InitializeComponent();
        
        _viewModel = new MainViewModel();
        DataContext = _viewModel;
        
        // 设置图片源
        _viewModel.ImageUpdated += image => Dispatcher.Invoke(() =>
        {
            ImageDisplay.Source = image;
            DropHintPanel.Visibility = Visibility.Collapsed;
            ImageInfoText.Text = $"{image.PixelWidth} x {image.PixelHeight}";
        });
        
        _viewModel.StatusUpdated += status => Dispatcher.Invoke(() => UpdateStatus(status));
        
        _viewModel.DetectionUpdated += results => Dispatcher.Invoke(() =>
        {
            DetectionListBox.ItemsSource = results;
            EmptyResultText.Visibility = results.Count == 0 ? Visibility.Visible : Visibility.Collapsed;
        });
        
        // 初始化
        UpdateStatus("Vision Client 已就绪");
        
        // 支持拖放
        AllowDrop = true;
        Drop += MainWindow_Drop;
        DragEnter += MainWindow_DragEnter;
        DragOver += MainWindow_DragOver;
    }
    
    private void UpdateStatus(string message)
    {
        StatusText.Text = $"[{DateTime.Now:HH:mm:ss}] {message}";
        System.Diagnostics.Debug.WriteLine($"[Status] {message}");
    }
    
    private void Settings_Click(object sender, RoutedEventArgs e)
    {
        var settingsWindow = new SettingsWindow();
        settingsWindow.Owner = this;
        settingsWindow.ShowDialog();
    }
    
    private void Confidence_Changed(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (ConfidenceValue != null)
        {
            ConfidenceValue.Text = ConfidenceSlider.Value.ToString("F2");
            _viewModel.ConfidenceThreshold = (float)ConfidenceSlider.Value;
        }
    }
    
    private void IoU_Changed(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (IoUValue != null)
        {
            IoUValue.Text = IoUSlider.Value.ToString("F2");
            _viewModel.IouThreshold = (float)IoUSlider.Value;
        }
    }
    
    #region 拖放处理
    
    private void MainWindow_DragEnter(object sender, DragEventArgs e)
    {
        if (e.Data.GetDataPresent(DataFormats.FileDrop))
        {
            e.Effects = DragDropEffects.Copy;
            DropHintPanel.Visibility = Visibility.Visible;
        }
        else
        {
            e.Effects = DragDropEffects.None;
        }
        e.Handled = true;
    }
    
    private void MainWindow_DragOver(object sender, DragEventArgs e)
    {
        e.Effects = e.Data.GetDataPresent(DataFormats.FileDrop) 
            ? DragDropEffects.Copy 
            : DragDropEffects.None;
        e.Handled = true;
    }
    
    private void MainWindow_Drop(object sender, DragEventArgs e)
    {
        DropHintPanel.Visibility = Visibility.Collapsed;
        
        if (e.Data.GetDataPresent(DataFormats.FileDrop))
        {
            var files = (string[])e.Data.GetData(DataFormats.FileDrop);
            if (files?.Length > 0)
            {
                var file = files[0];
                var ext = Path.GetExtension(file).ToLower();
                
                if (ext is ".jpg" or ".jpeg" or ".png" or ".bmp" or ".gif")
                {
                    _viewModel.ProcessImageFile(file);
                }
                else if (ext is ".mp4" or ".avi" or ".mov" or ".mkv")
                {
                    _viewModel.ProcessVideoFile(file);
                }
            }
        }
    }
    
    #endregion
    
    protected override void OnClosed(EventArgs e)
    {
        _viewModel.Dispose();
        base.OnClosed(e);
    }
}
