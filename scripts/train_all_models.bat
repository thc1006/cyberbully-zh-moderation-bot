@echo off
setlocal enabledelayedexpansion

REM ===============================================
REM CyberPuppy - Automated Training Pipeline
REM ===============================================
REM Train Conservative → Aggressive → RoBERTa sequentially
REM Stop if any model reaches F1 >= 0.75
REM Generate comparison report and notifications

echo.
echo =====================================
echo CyberPuppy Automated Training Started
echo =====================================
echo Start time: %date% %time%
echo.

REM Set up paths and variables
set PROJECT_ROOT=%~dp0..
set LOGS_DIR=%PROJECT_ROOT%\logs
set SCRIPTS_DIR=%PROJECT_ROOT%\scripts
set MODELS_DIR=%PROJECT_ROOT%\models

REM Create timestamp for this training session
for /f "tokens=1-6 delims=/:. " %%a in ("%date% %time%") do (
    set timestamp=%%c%%a%%b_%%d%%e%%f
)
set LOG_FILE=%LOGS_DIR%\training_%timestamp%.log

echo Training session ID: %timestamp% | tee %LOG_FILE%
echo Log file: %LOG_FILE% | tee %LOG_FILE%
echo. | tee %LOG_FILE%

REM Check if Python environment is activated
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found or environment not activated | tee %LOG_FILE%
    echo Please activate your Python environment and try again | tee %LOG_FILE%
    pause
    exit /b 1
)

REM Start resource monitoring in background
echo Starting resource monitoring... | tee %LOG_FILE%
start /b python "%SCRIPTS_DIR%\resource_monitor.py" --session-id %timestamp%

REM Configuration list: Conservative → Aggressive → RoBERTa
set configs=conservative aggressive roberta
set config_names=Conservative Aggressive RoBERTa-Large
set best_f1=0
set best_config=none

echo Training configurations: %config_names% | tee %LOG_FILE%
echo Target F1 threshold: 0.75 | tee %LOG_FILE%
echo. | tee %LOG_FILE%

REM Train each configuration sequentially
set config_index=0
for %%c in (%configs%) do (
    set /a config_index+=1
    set current_config=%%c

    echo ===================================== | tee %LOG_FILE%
    echo Training Configuration !config_index!: %%c | tee %LOG_FILE%
    echo ===================================== | tee %LOG_FILE%
    echo Start time: %time% | tee %LOG_FILE%

    REM Train the model
    echo python train.py --config %%c --output-dir "%MODELS_DIR%\%%c_%timestamp%" --log-file "%LOG_FILE%" | tee %LOG_FILE%
    python train.py --config %%c --output-dir "%MODELS_DIR%\%%c_%timestamp%" --log-file "%LOG_FILE%"

    if errorlevel 1 (
        echo ERROR: Training failed for configuration %%c | tee %LOG_FILE%
        set training_status=FAILED
        goto :end_training
    )

    REM Extract F1 score from training results
    echo Extracting F1 score for %%c... | tee %LOG_FILE%
    python "%SCRIPTS_DIR%\extract_metrics.py" --model-dir "%MODELS_DIR%\%%c_%timestamp%" --metric f1_macro > temp_f1.txt
    set /p current_f1=<temp_f1.txt
    del temp_f1.txt

    echo Configuration %%c achieved F1: !current_f1! | tee %LOG_FILE%

    REM Check if this is the best score
    python -c "import sys; sys.exit(0 if float('!current_f1!') > float('!best_f1!') else 1)"
    if not errorlevel 1 (
        set best_f1=!current_f1!
        set best_config=%%c
        echo New best configuration: %%c (F1: !current_f1!) | tee %LOG_FILE%
    )

    REM Check early stopping condition
    python -c "import sys; sys.exit(0 if float('!current_f1!') >= 0.75 else 1)"
    if not errorlevel 1 (
        echo ✓ Target F1 threshold reached! (F1: !current_f1! >= 0.75) | tee %LOG_FILE%
        echo Stopping training early - target achieved! | tee %LOG_FILE%
        set training_status=SUCCESS_EARLY
        goto :end_training
    )

    echo Completed training %%c (F1: !current_f1!) | tee %LOG_FILE%
    echo Continuing to next configuration... | tee %LOG_FILE%
    echo. | tee %LOG_FILE%
)

set training_status=SUCCESS_COMPLETE

:end_training
echo. | tee %LOG_FILE%
echo ===================================== | tee %LOG_FILE%
echo Training Pipeline Completed | tee %LOG_FILE%
echo ===================================== | tee %LOG_FILE%
echo End time: %date% %time% | tee %LOG_FILE%
echo Status: %training_status% | tee %LOG_FILE%
echo Best configuration: %best_config% | tee %LOG_FILE%
echo Best F1 score: %best_f1% | tee %LOG_FILE%
echo. | tee %LOG_FILE%

REM Stop resource monitoring
echo Stopping resource monitoring... | tee %LOG_FILE%
taskkill /f /im python.exe /fi "WINDOWTITLE eq resource_monitor*" >nul 2>&1

REM Generate comparison report
echo Generating comparison report... | tee %LOG_FILE%
python "%SCRIPTS_DIR%\compare_models.py" --session-id %timestamp% --best-config %best_config% --best-f1 %best_f1%

if errorlevel 1 (
    echo WARNING: Comparison report generation failed | tee %LOG_FILE%
) else (
    echo ✓ Comparison report generated successfully | tee %LOG_FILE%
)

REM Send notifications
echo Sending notifications... | tee %LOG_FILE%
python "%SCRIPTS_DIR%\send_notifications.py" --session-id %timestamp% --status %training_status% --best-config %best_config% --best-f1 %best_f1%

REM Open results folder
echo Opening results folder... | tee %LOG_FILE%
explorer "%MODELS_DIR%"

REM Display final summary
echo.
echo =====================================
echo TRAINING COMPLETE!
echo =====================================
echo Session ID: %timestamp%
echo Status: %training_status%
echo Best Model: %best_config% (F1: %best_f1%)
echo Log File: %LOG_FILE%
echo Results: %MODELS_DIR%
echo.
echo Press any key to view comparison report...
pause >nul

REM Open comparison report
start "" "%MODELS_DIR%\comparison_report_%timestamp%.html"

endlocal