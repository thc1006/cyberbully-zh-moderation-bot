@echo off
REM CyberPuppy LINE Bot Windows 啟動腳本

echo 🐕 CyberPuppy LINE Bot 啟動腳本

REM 檢查 Python 版本
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 未安裝或不在 PATH 中
    pause
    exit /b 1
)

echo ✅ Python 已安裝

REM 檢查環境變數檔案
if not exist ".env" (
    echo ⚠️  .env 檔案不存在，請參考 .env.example 建立
    if exist ".env.example" (
        echo 📋 可執行: copy .env.example .env
    )
    pause
    exit /b 1
)

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
if "%BOT_PORT%"=="" set BOT_PORT=8080
if "%BOT_HOST%"=="" set BOT_HOST=0.0.0.0
if "%LOG_LEVEL%"=="" set LOG_LEVEL=info

REM 載入 .env 檔案中的環境變數 (簡化版本)
for /f "tokens=1,2 delims==" %%a in (.env) do (
    if not "%%a"=="" (
        if not "%%a:~0,1%"=="#" (
            set "%%a=%%b"
        )
    )
)

REM 檢查必要環境變數
if "%LINE_CHANNEL_ACCESS_TOKEN%"=="" (
    echo ❌ 遺失 LINE_CHANNEL_ACCESS_TOKEN
    echo 請在 .env 檔案中設定 LINE Bot 相關變數
    pause
    exit /b 1
)

if "%LINE_CHANNEL_SECRET%"=="" (
    echo ❌ 遺失 LINE_CHANNEL_SECRET
    echo 請在 .env 檔案中設定 LINE Bot 相關變數
    pause
    exit /b 1
)

if "%CYBERPUPPY_API_URL%"=="" set CYBERPUPPY_API_URL=http://localhost:8000

echo 🔍 檢查 CyberPuppy API (%CYBERPUPPY_API_URL%)...

REM 簡化的 API 檢查 (Windows 版本)
curl -f "%CYBERPUPPY_API_URL%/healthz" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  CyberPuppy API 無法連線，請確認 API 服務已啟動
    echo 💡 可在另一個命令提示字元執行: cd ..\api && start.bat
) else (
    echo ✅ CyberPuppy API 可用
)

echo 🚀 啟動 CyberPuppy LINE Bot
echo    📍 URL: http://%BOT_HOST%:%BOT_PORT%
echo    🩺 健康檢查: http://%BOT_HOST%:%BOT_PORT%/health
echo    📊 統計資訊: http://%BOT_HOST%:%BOT_PORT%/stats
echo    🔗 Webhook: http://%BOT_HOST%:%BOT_PORT%/webhook
echo.
echo 📝 記得在 LINE Developers Console 設定 Webhook URL
echo.

REM 啟動服務
uvicorn line_bot:app --host %BOT_HOST% --port %BOT_PORT% --log-level %LOG_LEVEL% --reload

pause