@echo off
echo ========================================
echo   Installing Alpina Signal Dependencies
echo ========================================
echo.

echo Installing packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed!
    echo.
    echo Try manually:
    echo   python -m pip install torch numpy pandas scikit-learn
    echo   python -m pip install fastapi uvicorn
    echo   python -m pip install python-telegram-bot
    echo   python -m pip install sqlalchemy aiosqlite
    echo   python -m pip install loguru requests python-dotenv
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Installation Complete!
echo ========================================
echo.
echo Now you can run:
echo   python main.py bot
echo.
pause
