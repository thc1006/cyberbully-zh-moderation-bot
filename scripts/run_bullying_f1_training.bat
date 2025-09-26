@echo off
REM 霸凌偵測F1優化訓練執行腳本 (Windows版本)
REM 使用改進的架構和RTX 3050優化配置

setlocal EnableDelayedExpansion

REM 顏色設定 (簡化版)
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "WARNING=[WARNING]"
set "ERROR=[ERROR]"

REM 專案根目錄
cd /d "%~dp0.."
set "PROJECT_ROOT=%CD%"

echo %INFO% 專案根目錄: %PROJECT_ROOT%

REM 檢查Python環境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %ERROR% Python 未安裝或不在PATH中
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
echo %INFO% Python 版本: %PYTHON_VERSION%

REM 檢查GPU
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits 2>nul | head -1
if %errorlevel% neq 0 (
    echo %WARNING% 未偵測到NVIDIA GPU或nvidia-smi不可用
)

REM 設定預設參數
set "CONFIG_FILE=configs\training\bullying_f1_optimization.yaml"
set "EXPERIMENT_NAME=bullying_f1_075_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "EXPERIMENT_NAME=!EXPERIMENT_NAME: =0!"
set "OUTPUT_DIR=experiments\bullying_f1_optimization"
set "DATA_DIR=data\processed\training_dataset"

REM 解析命令列參數
:parse_args
if "%~1"=="--config" (
    set "CONFIG_FILE=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--experiment-name" (
    set "EXPERIMENT_NAME=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--output-dir" (
    set "OUTPUT_DIR=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--data-dir" (
    set "DATA_DIR=%~2"
    shift
    shift
    goto parse_args
)
if "%~1"=="--help" (
    echo 使用方法: %0 [選項]
    echo.
    echo 選項:
    echo   --config FILE           配置檔案路徑 ^(預設: %CONFIG_FILE%^)
    echo   --experiment-name NAME  實驗名稱 ^(預設: 自動生成^)
    echo   --output-dir DIR        輸出目錄 ^(預設: %OUTPUT_DIR%^)
    echo   --data-dir DIR          資料目錄 ^(預設: %DATA_DIR%^)
    echo   --help                  顯示此幫助訊息
    pause
    exit /b 0
)
if "%~1" neq "" (
    echo %ERROR% 未知參數: %~1
    echo 使用 --help 查看可用選項
    pause
    exit /b 1
)

echo %INFO% 使用配置檔案: %CONFIG_FILE%
echo %INFO% 實驗名稱: %EXPERIMENT_NAME%
echo %INFO% 輸出目錄: %OUTPUT_DIR%
echo %INFO% 資料目錄: %DATA_DIR%

REM 檢查必要檔案
echo %INFO% 檢查必要檔案...

if not exist "%CONFIG_FILE%" (
    echo %ERROR% 配置檔案不存在: %CONFIG_FILE%
    pause
    exit /b 1
)

if not exist "scripts\train_bullying_f1_optimizer.py" (
    echo %ERROR% 訓練腳本不存在: scripts\train_bullying_f1_optimizer.py
    pause
    exit /b 1
)

REM 檢查資料檔案
set "TRAIN_DATA=%DATA_DIR%\train.json"
if not exist "%TRAIN_DATA%" (
    if not exist "data\processed\cold\train.json" (
        echo %ERROR% 找不到訓練資料檔案
        echo %ERROR% 請確認以下路徑之一存在:
        echo %ERROR%   - %TRAIN_DATA%
        echo %ERROR%   - data\processed\cold\train.json
        pause
        exit /b 1
    )
)

REM 檢查Python依賴
echo %INFO% 檢查Python依賴...

set "PACKAGES=torch transformers scikit-learn numpy pandas PyYAML tqdm"
for %%p in (%PACKAGES%) do (
    python -c "import %%p" >nul 2>&1
    if !errorlevel! neq 0 (
        echo %ERROR% 缺少Python套件: %%p
        echo %INFO% 請執行: pip install %%p
        pause
        exit /b 1
    )
)

REM 建立輸出目錄
echo %INFO% 建立輸出目錄...
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "logs" mkdir "logs"

REM 設定日誌檔案
set "LOG_FILE=logs\training_%EXPERIMENT_NAME%.log"

REM 執行訓練
echo %INFO% 開始訓練霸凌偵測模型...
echo %INFO% 目標: 霸凌F1≥0.75, 毒性F1≥0.78, 總體F1≥0.76
echo %INFO% 日誌檔案: %LOG_FILE%

echo ==================== 訓練開始 ==================== > "%LOG_FILE%"
echo 時間: %date% %time% >> "%LOG_FILE%"
echo 配置: %CONFIG_FILE% >> "%LOG_FILE%"
echo 實驗: %EXPERIMENT_NAME% >> "%LOG_FILE%"
echo ===================================================== >> "%LOG_FILE%"

REM 執行Python訓練腳本
python scripts\train_bullying_f1_optimizer.py ^
    --config "%CONFIG_FILE%" ^
    --experiment-name "%EXPERIMENT_NAME%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --data-dir "%DATA_DIR%" 2>&1 | tee "%LOG_FILE%"

set "TRAINING_EXIT_CODE=%errorlevel%"

echo. >> "%LOG_FILE%"
echo ==================== 訓練結束 ==================== >> "%LOG_FILE%"
echo 時間: %date% %time% >> "%LOG_FILE%"
echo 退出碼: %TRAINING_EXIT_CODE% >> "%LOG_FILE%"

if %TRAINING_EXIT_CODE% equ 0 (
    echo %SUCCESS% 訓練完成!

    REM 顯示結果摘要
    set "RESULTS_FILE=%OUTPUT_DIR%\%EXPERIMENT_NAME%\final_results.json"
    if exist "!RESULTS_FILE!" (
        echo %INFO% 結果摘要:
        python -c "import json; results=json.load(open(r'!RESULTS_FILE!', 'r', encoding='utf-8')); test_metrics=results.get('test_metrics', {}); targets=results.get('target_achieved', {}); print(f'霸凌F1: {test_metrics.get(\"bullying_f1\", 0):.4f} ({\"✅\" if targets.get(\"bullying_f1_075\", False) else \"❌\"})'); print(f'毒性F1: {test_metrics.get(\"toxicity_f1\", 0):.4f} ({\"✅\" if targets.get(\"toxicity_f1_078\", False) else \"❌\"})'); print(f'總體F1: {test_metrics.get(\"overall_macro_f1\", 0):.4f} ({\"✅\" if targets.get(\"overall_macro_f1_076\", False) else \"❌\"})')"
    )

    REM TensorBoard 資訊
    set "TENSORBOARD_DIR=%OUTPUT_DIR%\%EXPERIMENT_NAME%\tensorboard_logs"
    if exist "!TENSORBOARD_DIR!" (
        echo %INFO% TensorBoard 日誌位置: !TENSORBOARD_DIR!
        echo %INFO% 啟動TensorBoard: tensorboard --logdir "!TENSORBOARD_DIR!"
    )

    REM 模型工件位置
    set "MODEL_DIR=%OUTPUT_DIR%\%EXPERIMENT_NAME%\model_artifacts"
    if exist "!MODEL_DIR!" (
        echo %INFO% 模型檔案位置: !MODEL_DIR!
    )

) else (
    echo %ERROR% 訓練失敗 ^(退出碼: %TRAINING_EXIT_CODE%^)
    echo %INFO% 請查看日誌檔案: %LOG_FILE%
    pause
    exit /b %TRAINING_EXIT_CODE%
)

echo ===================================================== >> "%LOG_FILE%"

REM 可選: 自動啟動TensorBoard
set /p "REPLY=是否要啟動TensorBoard? (y/N): "
if /i "!REPLY!"=="y" (
    if exist "!TENSORBOARD_DIR!" (
        echo %INFO% 啟動TensorBoard...
        echo 在瀏覽器中打開: http://localhost:6006
        start tensorboard --logdir "!TENSORBOARD_DIR!" --port 6006
    ) else (
        echo %WARNING% TensorBoard日誌目錄不存在
    )
)

echo %SUCCESS% 腳本執行完成!
pause