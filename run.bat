@echo off
cd /d "%~dp0"

if not exist ".venv_local\Scripts\python.exe" (
    echo Creating virtual environment in .venv_local...
    py -3 -m venv .venv_local 2>nul || python -m venv .venv_local 2>nul
    if errorlevel 1 (
        echo ERROR: Could not create virtual environment. Make sure Python 3 is installed and on PATH.
        pause
        exit /b 1
    )
)

call ".venv_local\Scripts\activate.bat"

python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing Python dependencies from requirements.txt...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies.
        pause
        exit /b 1
    )
)

echo Starting Flask application...
python app.py
pause
