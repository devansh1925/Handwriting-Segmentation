@echo off
REM setup.bat
REM Script to install Tesseract OCR and Python dependencies on Windows

echo =====================================
echo Installing Tesseract OCR...
echo =====================================

REM Check if Chocolatey is installed (used for easy Tesseract install)
where choco >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Chocolatey is not installed. Installing Chocolatey...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
)

REM Install Tesseract OCR using Chocolatey
choco install tesseract -y

echo =====================================
echo Installing Python dependencies...
echo =====================================
pip install --upgrade pip
pip install -r requirements.txt

echo =====================================
echo Setup Completed!
echo =====================================
echo Make sure to add Tesseract installation path to your environment PATH.
echo Example path: C:\Program Files\Tesseract-OCR
pause
