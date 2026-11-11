@echo off
chcp 65001 >nul
echo ===================================================================
echo     KHOI DONG ADAS ROAD ASSISTANT (4 TRANG THAI CANH BAO + GIAI THICH)
echo ===================================================================

if exist "%~dp0src\adas" (
    cd /d "%~dp0"
    set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
    set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
) else if exist "%~dp0adas-road-assistant\src\adas" (
    cd /d "%~dp0adas-road-assistant"
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
) else (
    set "PYTHON_EXE=python"
    set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
)

if not exist "%PYTHON_EXE%" (
    if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
    if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
)

if "%~1"=="--help" (
    "%PYTHON_EXE%" -m adas --help
    exit /b 0
)
if "%~1"=="-h" (
    "%PYTHON_EXE%" -m adas --help
    exit /b 0
)

set "SOURCE=%~1"
set "SPEED=%~2"

:: Smart shortcut aliases
if "%SOURCE%"=="hem_ngo" set "SOURCE=data/videos/demo_hem_ngo.mp4"
if "%SOURCE%"=="cao_toc" set "SOURCE=data/videos/demo_cao_toc.mp4"
if "%SOURCE%"=="duong_doi" set "SOURCE=data/videos/demo_duong_doi_do_thi.mp4"
if "%SOURCE%"=="duong_doi_do_thi" set "SOURCE=data/videos/demo_duong_doi_do_thi.mp4"
if "%SOURCE%"=="pho_2chieu" set "SOURCE=data/videos/demo_pho_hai_chieu.mp4"
if "%SOURCE%"=="pho_hai_chieu" set "SOURCE=data/videos/demo_pho_hai_chieu.mp4"
if "%SOURCE%"=="ngoai_kdc" set "SOURCE=data/videos/demo_ngoai_khu_dan_cu.mp4"
if "%SOURCE%"=="ngoai_khu_dan_cu" set "SOURCE=data/videos/demo_ngoai_khu_dan_cu.mp4"

if "%SOURCE%"=="" set "SOURCE=data/videos/demo_duong_doi_do_thi.mp4"
if "%SPEED%"=="" set "SPEED=55"

echo ▶ Nguon video : %SOURCE%
echo ▶ Toc do xe  : %SPEED% km/h (Bam +/- tren ban phim de thay doi truc tiep)
echo ▶ 4 Trang thai canh bao: Dang xac dinh ^| An toan ^| Chu y ^| Vuot toc
echo.

"%PYTHON_EXE%" -m adas --source "%SOURCE%" --demo-speed %SPEED% %3 %4 %5 %6 %7 %8 %9
if errorlevel 1 (
    echo.
    echo [LOI] Chuong trinh bi dung hoac xay ra loi!
    pause
)
