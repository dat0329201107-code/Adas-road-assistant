@echo off
chcp 65001 >nul
echo ===================================================================
echo     NHAN DIEN LOAI DUONG VIET NAM VA 4 TRANG THAI CANH BAO
echo ===================================================================

if exist "%~dp0src\adas" (
    cd /d "%~dp0"
    set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
) else if exist "%~dp0adas-road-assistant\src\adas" (
    cd /d "%~dp0adas-road-assistant"
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

if not exist "%PYTHON_EXE%" (
    if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
    if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
)

set "VIDEO_PATH=%~1"
set "SPEED=%~2"

:: Smart shortcut aliases
if "%VIDEO_PATH%"=="hem_ngo" set "VIDEO_PATH=data/videos/demo_hem_ngo.mp4"
if "%VIDEO_PATH%"=="cao_toc" set "VIDEO_PATH=data/videos/demo_cao_toc.mp4"
if "%VIDEO_PATH%"=="duong_doi" set "VIDEO_PATH=data/videos/demo_duong_doi_do_thi.mp4"
if "%VIDEO_PATH%"=="duong_doi_do_thi" set "VIDEO_PATH=data/videos/demo_duong_doi_do_thi.mp4"
if "%VIDEO_PATH%"=="pho_2chieu" set "VIDEO_PATH=data/videos/demo_pho_hai_chieu.mp4"
if "%VIDEO_PATH%"=="pho_hai_chieu" set "VIDEO_PATH=data/videos/demo_pho_hai_chieu.mp4"
if "%VIDEO_PATH%"=="ngoai_kdc" set "VIDEO_PATH=data/videos/demo_ngoai_khu_dan_cu.mp4"
if "%VIDEO_PATH%"=="ngoai_khu_dan_cu" set "VIDEO_PATH=data/videos/demo_ngoai_khu_dan_cu.mp4"

if "%VIDEO_PATH%"=="" (
    set "VIDEO_PATH=data/videos/demo_duong_doi_do_thi.mp4"
    echo [THONG BAO] Khong co tham so video, chay mac dinh video duong doi do thi:
    echo   %VIDEO_PATH%
    echo.
)

set "SPEED_OPT="
if not "%SPEED%"=="" set "SPEED_OPT=--speed %SPEED%"

"%PYTHON_EXE%" scripts/classify_video_road.py --video "%VIDEO_PATH%" --play %SPEED_OPT% %3 %4 %5 %6

echo.
pause
