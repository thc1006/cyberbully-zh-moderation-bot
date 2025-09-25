@echo off
REM CyberPuppy 本地快速啟動腳本

echo ======================================
echo CyberPuppy 本地服務啟動程式
echo ======================================
echo.

REM 檢查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 未安裝！
    exit /b 1
)

REM 檢查必要檔案
echo [INFO] 檢查必要檔案...
python scripts\check_requirements.py
if %errorlevel% neq 0 (
    echo [ERROR] 缺少必要檔案！請執行：
    echo   python scripts\download_datasets.py
    exit /b 1
)

REM 啟動 API 服務
echo.
echo [INFO] 啟動 API 服務...
echo [INFO] 服務將在 http://localhost:8000 啟動
echo.
echo 按 Ctrl+C 停止服務
echo.

cd api
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload