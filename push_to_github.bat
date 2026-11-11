@echo off
chcp 65001 >nul
echo =======================================================
echo    DANG DAY CODE LEN GITHUB: Notification-for-Driver
echo =======================================================
cd /d "%~dp0"
echo ▶ Branch: main
echo ▶ Remote: https://github.com/DikeyoZero/Notification-for-Driver-based-on-type-of-road-.git
echo.
git push origin main
if errorlevel 1 (
    echo.
    echo [THONG BAO] Neu GitHub yeu cau dang nhap, hay chon 'Sign in with your browser'
    echo hoac su dung Personal Access Token (PAT).
    pause
) else (
    echo.
    echo [THANH CONG] Da day toan bo code len GitHub thanh cong!
    pause
)
