@echo off
cd /d "%~dp0examples"
"%~dp0.venv\Scripts\python.exe" kronos_gui.py
if errorlevel 1 pause
