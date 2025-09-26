@echo off
REM CyberPuppy Docker Deployment Test Script
REM 中文網路霸凌防治系統 Docker 部署測試腳本

echo ================================
echo CyberPuppy Docker 部署測試
echo ================================
echo.

REM 檢查 Docker 是否可用
echo [1/7] 檢查 Docker 服務狀態...
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker 服務未運行，請先啟動 Docker Desktop
    echo 請手動啟動 Docker Desktop 後重新執行此腳本
    pause
    exit /b 1
)
echo ✅ Docker 服務正常運行
echo.

REM 檢查 Docker Compose 是否可用
echo [2/7] 檢查 Docker Compose...
docker-compose version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose 不可用
    pause
    exit /b 1
)
echo ✅ Docker Compose 可用
echo.

REM 檢查必要檔案
echo [3/7] 檢查必要檔案...
if not exist "Dockerfile.api" (
    echo ❌ 缺少 Dockerfile.api
    pause
    exit /b 1
)
if not exist "Dockerfile.bot" (
    echo ❌ 缺少 Dockerfile.bot
    pause
    exit /b 1
)
if not exist "docker-compose.yml" (
    echo ❌ 缺少 docker-compose.yml
    pause
    exit /b 1
)
if not exist "requirements.txt" (
    echo ❌ 缺少 requirements.txt
    pause
    exit /b 1
)
echo ✅ 所有必要檔案都存在
echo.

REM 檢查環境變數檔案
echo [4/7] 檢查環境變數設定...
if not exist "configs\docker\.env" (
    echo ⚠️  .env 檔案不存在，將複製範例檔案
    if exist "configs\docker\.env.example" (
        copy "configs\docker\.env.example" "configs\docker\.env" >nul
        echo ✅ 已建立 .env 檔案（請記得填入實際的 LINE Bot 設定）
    ) else (
        echo ❌ .env.example 檔案也不存在
        pause
        exit /b 1
    )
) else (
    echo ✅ .env 檔案已存在
)
echo.

REM 建立必要目錄
echo [5/7] 建立必要目錄...
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "data" mkdir data
echo ✅ 目錄結構準備完成
echo.

REM 驗證 Docker Compose 檔案
echo [6/7] 驗證 Docker Compose 檔案...
docker-compose config >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ docker-compose.yml 檔案格式錯誤
    echo 請檢查 YAML 語法
    pause
    exit /b 1
)
echo ✅ Docker Compose 檔案格式正確
echo.

REM 嘗試建置映像
echo [7/7] 建置 Docker 映像（不啟動服務）...
echo 建置 API 映像...
docker-compose build --no-cache api
if %errorlevel% neq 0 (
    echo ❌ API 映像建置失敗
    pause
    exit /b 1
)
echo ✅ API 映像建置成功

echo 建置 Bot 映像...
docker-compose build --no-cache bot
if %errorlevel% neq 0 (
    echo ❌ Bot 映像建置失敗
    pause
    exit /b 1
)
echo ✅ Bot 映像建置成功
echo.

echo ========================================
echo 🎉 Docker 部署測試完成！
echo ========================================
echo.
echo 下一步：
echo 1. 編輯 configs/docker/.env 填入正確的 LINE Bot 設定
echo 2. 執行: docker-compose up -d
echo 3. 測試健康檢查端點:
echo    - API: http://localhost:8000/healthz
echo    - Bot: http://localhost:8080/health
echo.
pause