@echo off
REM CyberPuppy Docker Deployment Script for Windows

echo ======================================
echo CyberPuppy Docker Deployment Script
echo ======================================
echo.

REM Check prerequisites
echo [INFO] Checking prerequisites...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed!
    exit /b 1
)

docker compose version >nul 2>&1
if %errorlevel% neq 0 (
    docker-compose --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Docker Compose is not installed!
        exit /b 1
    )
)
echo [INFO] Prerequisites check passed
echo.

REM Setup environment
echo [INFO] Setting up environment...
if not exist .env (
    echo [INFO] Creating .env file from template...
    copy .env.example .env
    echo [WARN] Please edit .env file to set your configuration
) else (
    echo [INFO] .env file already exists
)
echo.

REM Download models and data
echo [INFO] Checking models and data...
if not exist "models\macbert_base_demo\best.ckpt" (
    echo [WARN] Models not found. Downloading...
    python scripts\download_datasets.py
) else (
    echo [INFO] Models already exist
)

if not exist "data\processed\unified\train_unified.json" (
    echo [WARN] Processed data not found. Processing...
    python scripts\create_unified_training_data_v2.py
) else (
    echo [INFO] Processed data already exists
)
echo.

REM Parse arguments
set PROFILE=default
if "%1"=="--with-bot" set PROFILE=with-bot
if "%1"=="--with-cache" set PROFILE=with-cache
if "%1"=="--production" set PROFILE=production

REM Build Docker images
echo [INFO] Building Docker images...
docker compose build --no-cache api

if "%PROFILE%"=="with-bot" (
    docker compose build bot
)
echo [INFO] Docker images built successfully
echo.

REM Start services
echo [INFO] Starting services...
if "%PROFILE%"=="with-bot" (
    docker compose --profile with-bot up -d
) else if "%PROFILE%"=="with-cache" (
    docker compose --profile with-cache up -d
) else if "%PROFILE%"=="production" (
    docker compose --profile production --profile with-cache up -d
) else (
    docker compose up -d api
)
echo [INFO] Services started
echo.

REM Wait for services
echo [INFO] Waiting for services to be ready...
:wait_loop
timeout /t 2 >nul
curl -f http://localhost:8000/healthz >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] API service is ready
    goto :ready
)
echo .
goto :wait_loop

:ready
echo.
echo [INFO] Service Status:
docker compose ps
echo.

echo [INFO] Service URLs:
echo   - API: http://localhost:8000
echo   - API Docs: http://localhost:8000/docs
echo   - Health Check: http://localhost:8000/healthz

if "%PROFILE%"=="with-bot" (
    echo   - LINE Bot: http://localhost:5000
)

echo.
echo [INFO] Deployment completed successfully!
echo.
echo Quick test:
echo   curl -X POST http://localhost:8000/v1/analyze -H "Content-Type: application/json" -d "{\"text\": \"測試文本\"}"
echo.