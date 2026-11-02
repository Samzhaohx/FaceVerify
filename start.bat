@echo off
REM ============================================================
REM  One-click launcher. Always runs from THIS file's directory,
REM  so it works no matter where you open the console.
REM ============================================================
cd /d "%~dp0"

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

echo [1/3] Project : %CD%
echo [2/3] Python  : %PY%

"%PY%" -c "import flask, cv2, numpy, onnxruntime" 2>nul
if errorlevel 1 (
    echo.
    echo [!] Dependencies are missing for this interpreter.
    echo     Install them with:
    echo         "%PY%" -m pip install -r requirements-cpu.txt
    echo.
    pause
    exit /b 1
)

if not exist "encodings\faces.pkl" (
    echo.
    echo [!] No face gallery found ^(encodings\faces.pkl^).
    echo     Enroll faces first:
    echo         "%PY%" register.py --name yourname
    echo.
)

echo [3/3] Starting web server, open the URL printed below ...
echo.
"%PY%" app.py %*

pause
