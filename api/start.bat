@echo off
REM CyberPuppy API Windows 啟動腳本

echo 🐕 CyberPuppy API 啟動腳本

REM 檢查 Python 版本
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 未安裝或不在 PATH 中
    pause
    exit /b 1
)

echo ✅ Python 已安裝

REM 檢查並建立虛擬環境
if not exist "venv" (
    echo 📦 建立虛擬環境...
    python -m venv venv
)

REM 啟動虛擬環境
echo 🔧 啟動虛擬環境...
call venv\Scripts\activate.bat

REM 安裝依賴
echo 📚 安裝依賴套件...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM 設定環境變數
set PYTHONPATH=%cd%
if "%PORT%"=="" set PORT=8000
if "%LOG_LEVEL%"=="" set LOG_LEVEL=info
if "%WORKERS%"=="" set WORKERS=1

echo 🚀 啟動 CyberPuppy API 服務
echo    📍 URL: http://localhost:%PORT%
echo    📖 API 文檔: http://localhost:%PORT%/docs
echo    🩺 健康檢查: http://localhost:%PORT%/healthz
echo.

REM 啟動 uvicorn 服務器
uvicorn app:app --host 0.0.0.0 --port %PORT% --workers %WORKERS% --log-level %LOG_LEVEL% --reload

pause