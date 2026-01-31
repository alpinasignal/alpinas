@echo off
echo ========================================
echo   Alpina Signal - Quick Start
echo ========================================
echo.

echo Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Starting Telegram Bot
echo ========================================
echo.
echo Bot will start in 3 seconds...
echo Find your bot in Telegram and send /start
echo.
timeout /t 3

python main.py bot

pause
