@echo off
cd /d "%~dp0"

:: Use existing virtual environment or create it automatically
if not exist ".venv_local\Scripts\python.exe" (
    echo Creating virtual environment .venv_local...
    if exist "%SystemRoot%\py.exe" (
        py -3 -m venv .venv_local
    ) else (
        if exist "%SystemRoot%\System32\python.exe" (
            python -m venv .venv_local
        ) else (
            echo ERROR: Python was not found on this machine.
            echo Install Python 3 and retry.
            pause
            exit /b 1
        )
    )
    echo Installing requirements...
    .venv_local\Scripts\python.exe -m pip install --upgrade pip
    .venv_local\Scripts\python.exe -m pip install -r requirements.txt
)

call .venv_local\Scripts\activate.bat
python app.py
pause
