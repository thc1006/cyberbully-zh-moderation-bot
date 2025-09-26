@echo off
setlocal enabledelayedexpansion

REM ===============================================
REM CyberPuppy Training Automation Setup
REM ===============================================
REM Quick setup script for training automation

echo.
echo =====================================
echo CyberPuppy Training Automation Setup
echo =====================================
echo.

set PROJECT_ROOT=%~dp0..

REM Check Python environment
echo Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python and activate your environment.
    pause
    exit /b 1
)

echo ✓ Python environment detected

REM Install required packages for automation
echo.
echo Installing automation dependencies...
pip install psutil matplotlib seaborn pandas plyer requests pygame >nul 2>&1
if errorlevel 1 (
    echo WARNING: Some packages may not have installed correctly
) else (
    echo ✓ Automation dependencies installed
)

REM Create necessary directories
echo.
echo Creating directory structure...
mkdir "%PROJECT_ROOT%\logs" 2>nul
mkdir "%PROJECT_ROOT%\models" 2>nul
mkdir "%PROJECT_ROOT%\config" 2>nul
mkdir "%PROJECT_ROOT%\assets\sounds" 2>nul
echo ✓ Directory structure created

REM Create notification config if it doesn't exist
echo.
echo Setting up notification configuration...
python "%PROJECT_ROOT%\scripts\send_notifications.py" --session-id "setup" --status "FAILED" --test >nul 2>&1
echo ✓ Notification configuration initialized

REM Show available options
echo.
echo =====================================
echo Setup Complete! Available Commands:
echo =====================================
echo.
echo 1. IMMEDIATE TRAINING:
echo    train_all_models.bat
echo.
echo 2. SCHEDULE OVERNIGHT TRAINING:
echo    python scripts\schedule_training.py create --time 23:00 --frequency daily
echo.
echo 3. SCHEDULE WEEKEND TRAINING:
echo    python scripts\schedule_training.py create --time 20:00 --frequency weekly
echo.
echo 4. TEST NOTIFICATIONS:
echo    python scripts\send_notifications.py --session-id test --status SUCCESS_EARLY --test
echo.
echo 5. VIEW SCHEDULED TASKS:
echo    python scripts\schedule_training.py list
echo.
echo 6. MANUAL TASK MANAGEMENT:
echo    python scripts\schedule_training.py enable     (Enable scheduled task)
echo    python scripts\schedule_training.py disable    (Disable scheduled task)
echo    python scripts\schedule_training.py delete     (Remove scheduled task)
echo    python scripts\schedule_training.py run        (Run task immediately)
echo    python scripts\schedule_training.py info       (Show task details)
echo.

REM Ask user what they want to do
echo What would you like to do now?
echo.
echo [1] Run training immediately
echo [2] Schedule overnight training (11 PM daily)
echo [3] Schedule weekend training (8 PM weekly)
echo [4] Test notifications only
echo [5] Exit (manual setup later)
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto :run_immediate
if "%choice%"=="2" goto :schedule_overnight
if "%choice%"=="3" goto :schedule_weekend
if "%choice%"=="4" goto :test_notifications
if "%choice%"=="5" goto :manual_exit

echo Invalid choice. Exiting...
goto :manual_exit

:run_immediate
echo.
echo Starting immediate training...
echo This will train Conservative → Aggressive → RoBERTa models sequentially.
echo The process will stop early if any model achieves F1 >= 0.75.
echo.
echo Press any key to start training, or Ctrl+C to cancel...
pause >nul
start "" "%PROJECT_ROOT%\scripts\train_all_models.bat"
goto :end

:schedule_overnight
echo.
echo Setting up overnight training schedule...
python "%PROJECT_ROOT%\scripts\schedule_training.py" create --time 23:00 --frequency daily --enabled
if errorlevel 1 (
    echo ERROR: Failed to create scheduled task
    goto :manual_exit
)
echo.
echo ✅ Overnight training scheduled for 11:00 PM daily
echo Your computer will wake up and start training automatically.
echo.
echo To modify the schedule, use:
echo   python scripts\schedule_training.py delete
echo   python scripts\schedule_training.py create --time HH:MM --frequency daily
echo.
goto :end

:schedule_weekend
echo.
echo Setting up weekend training schedule...
python "%PROJECT_ROOT%\scripts\schedule_training.py" create --time 20:00 --frequency weekly --enabled
if errorlevel 1 (
    echo ERROR: Failed to create scheduled task
    goto :manual_exit
)
echo.
echo ✅ Weekend training scheduled for 8:00 PM weekly
echo Training will run automatically every week.
echo.
goto :end

:test_notifications
echo.
echo Testing notification systems...
python "%PROJECT_ROOT%\scripts\send_notifications.py" --session-id "test" --status "SUCCESS_EARLY" --best-config "test" --best-f1 0.85 --test
echo.
echo Notification test completed. Check if you received:
echo - Desktop notification
echo - Sound notification
echo - Email (if configured)
echo.
echo To configure email notifications, edit:
echo   config\notifications.json
echo.
goto :end

:manual_exit
echo.
echo Manual setup instructions:
echo.
echo 1. To run training immediately:
echo    Double-click: scripts\train_all_models.bat
echo.
echo 2. To schedule training:
echo    python scripts\schedule_training.py create --time 23:00 --frequency daily
echo.
echo 3. To configure notifications:
echo    Edit: config\notifications.json
echo.
echo 4. For help with any script:
echo    python scripts\[script_name].py --help
echo.
goto :end

:end
echo.
echo =====================================
echo Important Tips:
echo =====================================
echo.
echo 💡 HANDS-OFF USAGE:
echo   1. Run train_all_models.bat
echo   2. Go to sleep 😴
echo   3. Wake up to results 📊
echo.
echo 📊 RESULTS LOCATION:
echo   - Training logs: logs\training_YYYYMMDD_HHMMSS.log
echo   - Models: models\[config]_[timestamp]\
echo   - Comparison report: models\comparison_report_[timestamp].html
echo   - Resource monitoring: logs\resource_monitor_[timestamp].log
echo.
echo 🔔 NOTIFICATIONS:
echo   - Desktop notifications for completion/alerts
echo   - Optional email notifications (configure in config\notifications.json)
echo   - Sound alerts for success/failure
echo   - Resource usage alerts during training
echo.
echo 🛡️ SAFETY FEATURES:
echo   - Early stopping at F1 >= 0.75
echo   - Resource monitoring and alerts
echo   - Automatic cleanup on errors
echo   - Comprehensive logging
echo.
echo 📈 MONITORING:
echo   - Real-time resource usage tracking
echo   - Training progress in logs
echo   - Email alerts for critical resource usage
echo   - Desktop notifications for completion
echo.
echo For technical support, check the logs directory for detailed information.
echo.
pause
endlocal