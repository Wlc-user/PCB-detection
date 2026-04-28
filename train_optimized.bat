@echo off
chcp 65001 >nul
echo ================================================
echo     YOLOv10 优化训练启动
echo ================================================
echo.
echo 训练配置:
echo   - 学习率: 0.001 (原0.01)
echo   - 优化器: AdamW
echo   - 马赛克: 0.5
echo   - 预训练: 使用
echo.
echo 预计时间: 30-60分钟 (根据GPU)
echo.
echo ================================================
echo.
cd /d "%~dp0"
python train_optimized_v2.py --mode retrain
pause
