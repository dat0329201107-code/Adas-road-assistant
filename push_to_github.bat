@echo off
chcp 65001 >nul
echo ===================================================================
echo    DANG DAY TOAN BO DU AN LEN GITHUB: Adas-road-assistant
echo ===================================================================
cd /d "%~dp0"
echo ▶ Repo: https://github.com/dat0329201107-code/Adas-road-assistant.git
echo ▶ Branch: main
echo.
git push -u origin main --force
if errorlevel 1 (
    echo.
    echo ===================================================================
    echo [HUONG DAN XAC THUC GITHUB]
    echo 1. Neu hien cua so dang nhap, chon 'Sign in with your browser'
    echo 2. Neu yeu cau Personal Access Token (PAT):
    echo    Vao https://github.com/settings/tokens tao token voi quyen 'repo'
    echo    roi dan vao khi duoc hoi password.
    echo ===================================================================
    pause
) else (
    echo.
    echo ===================================================================
    echo [THANH CONG] Da day toan bo du an len Adas-road-assistant thanh cong!
    echo Hay vao trinh duyet F5 lai trang GitHub de xem ket qua.
    echo ===================================================================
    pause
)
