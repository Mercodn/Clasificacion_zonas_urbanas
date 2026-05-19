@echo off
cd /d "%~dp0"
call .venv_local\Scripts\activate.bat
python app.py
pause
