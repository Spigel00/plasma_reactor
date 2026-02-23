@echo off
REM Setup script for Ollama integration with Plasma Control System

echo =====================================================
echo Plasma Control System - Ollama Setup
echo =====================================================
echo.

REM Check if Ollama is installed
where ollama >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Ollama is not installed or not in PATH
    echo.
    echo To use the Ollama integration:
    echo 1. Download Ollama from https://ollama.ai
    echo 2. Install and run Ollama
    echo 3. The system will automatically pull llama3.2
    echo.
    echo Once installed, you can:
    echo   - Run: ollama serve (in one terminal)
    echo   - Run the plasma control pipeline in another terminal
    echo.
    pause
    exit /b 0
)

echo ✓ Ollama found in PATH
echo.

REM Try to connect to running Ollama instance
echo Checking if Ollama service is running...
timeout /t 2 /nobreak >nul

python -c "import requests; requests.get('http://localhost:11434/api/tags', timeout=2); print('✓ Ollama service is running')" 2>nul

if %errorlevel% neq 0 (
    echo.
    echo ⚠ Ollama service not running
    echo.
    echo Start Ollama with:
    echo   Command: ollama serve
    echo.
    echo Then the system will automatically pull llama3.2 on first run
    echo.
    pause
    exit /b 0
)

echo.
echo ✓ Ollama service is running
echo.
echo Pulling llama3.2 model (this may take a few minutes on first run)...
echo.

ollama pull llama3.2

if %errorlevel% equ 0 (
    echo.
    echo =========================================
    echo ✓ Setup complete!
    echo =========================================
    echo.
    echo You can now run:
    echo   python run_complete_plasma_control.py
    echo.
    echo This will generate AI-powered reports using Llama 3.2
    echo.
) else (
    echo.
    echo ⚠ Failed to pull model
    echo Make sure Ollama service is running: ollama serve
    echo.
)

pause
