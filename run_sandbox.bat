@echo off
cd /d "%~dp0"
python witness_sandbox.py
if %ERRORLEVEL% NEQ 0 (
    echo Error exit code: %ERRORLEVEL%
    pause
)
