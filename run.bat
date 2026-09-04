@echo off
REM Double-click this to start the Return-Risk Console.
cd /d "%~dp0backend"

if exist "..\venv\Scripts\activate.bat" (
  call "..\venv\Scripts\activate.bat"
)

python app.py
pause
