@echo off
chcp 65001 >nul
title CyberPuppy 本地訓練啟動器

echo.
echo =========================================
echo 🐕 CyberPuppy 本地訓練啟動器
echo    中文網路霸凌檢測模型訓練系統
echo =========================================
echo.

:: 檢查 Python 是否安裝
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python 未安裝或未加入 PATH
    echo 請安裝 Python 3.8+ 並確保已加入系統 PATH
    pause
    exit /b 1
)

:: 檢查是否在正確目錄
if not exist "src\cyberpuppy" (
    echo ❌ 請在 CyberPuppy 專案根目錄執行此腳本
    echo 當前目錄: %CD%
    pause
    exit /b 1
)

:: 檢查虛擬環境
if defined VIRTUAL_ENV (
    echo ✅ 虛擬環境已啟用: %VIRTUAL_ENV%
) else (
    echo ⚠️  未檢測到虛擬環境
    echo 建議先啟用虛擬環境以避免套件衝突
    echo.
    set /p continue="是否繼續? (y/N): "
    if /i not "%continue%"=="y" (
        echo 已取消
        pause
        exit /b 0
    )
)

:: 檢查必要套件
echo 🔍 檢查必要套件...
python -c "import torch; print('✅ PyTorch:', torch.__version__)" 2>nul || (
    echo ❌ PyTorch 未安裝
    echo 請執行: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pause
    exit /b 1
)

python -c "import transformers; print('✅ Transformers:', transformers.__version__)" 2>nul || (
    echo ❌ Transformers 未安裝
    echo 請執行: pip install transformers
    pause
    exit /b 1
)

python -c "import tqdm; print('✅ tqdm 已安裝')" 2>nul || (
    echo ❌ tqdm 未安裝
    echo 請執行: pip install tqdm
    pause
    exit /b 1
)

python -c "import colorama; print('✅ colorama 已安裝')" 2>nul || (
    echo ⚠️  colorama 未安裝，將以無顏色模式執行
    echo 建議執行: pip install colorama
)

echo.
echo 🚀 啟動訓練程式...
echo.

:: 執行訓練腳本
python scripts\train_local.py

:: 檢查執行結果
if %errorlevel% equ 0 (
    echo.
    echo ✅ 程式執行完成
) else (
    echo.
    echo ❌ 程式執行時發生錯誤 (錯誤碼: %errorlevel%)
)

echo.
echo 按任意鍵關閉視窗...
pause >nul