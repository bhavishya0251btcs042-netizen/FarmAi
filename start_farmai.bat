@echo off
title FarmAI - Crop Recommendation System
color 0A

echo.
echo  ========================================
echo    FarmAI - Crop Recommendation System
echo  ========================================
echo.

cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not installed or not in PATH.
    echo  Please install Python from https://python.org
    pause
    exit /b 1
)

echo  [1/3] Installing dependencies (first run installs AI model ~500MB)...
python -m pip install "bcrypt==4.0.1" --quiet
python -m pip install "pymongo[srv]==4.6.1" --quiet
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu --quiet
python -m pip install transformers==4.36.0 --quiet
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo  [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo  [2/3] Starting FarmAI backend server...
echo.

start "FarmAI Server" cmd /k "cd /d "%~dp0" && python -m uvicorn crop:app --host 127.0.0.1 --port 8000"

timeout /t 5 /nobreak >nul


start "" "http://127.0.0.1:8000"

echo.
echo  ========================================
echo    FarmAI is running at http://127.0.0.1:8000
echo  ========================================
echo.
echo  NOTE: First disease scan will take ~30 seconds to load the AI model.
echo  Keep the "FarmAI Server" window open.
echo.
pause
