@echo off
title Multilingual AI Language Translator
color 0B

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║       🌐  MULTILINGUAL AI LANGUAGE TRANSLATOR  🌐       ║
echo  ║   Speak . Translate . Connect — 49 Languages            ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

:: Move to project root
cd /d "%~dp0"

:: Set PYTHONPATH to project root
set PYTHONPATH=%~dp0

:: Suppress HuggingFace symlink warning on Windows
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo  [1/3] Checking Python...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo  ERROR: Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

echo.
echo  [2/3] Checking dependencies...
python -c "import gradio, torch; print(f'  Gradio {gradio.__version__} | PyTorch {torch.__version__}')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo  Installing dependencies...
    pip install -r requirements.txt --quiet
)

echo.
echo  [3/3] Starting Gradio App...
echo.
echo  ┌──────────────────────────────────────────────┐
echo  │  App will open at: http://localhost:7860     │
echo  │  Press Ctrl+C to stop the server            │
echo  └──────────────────────────────────────────────┘
echo.

python app/gradio_app.py

pause
