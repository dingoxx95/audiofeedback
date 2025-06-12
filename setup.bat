@echo off
REM Audio Feedback Analyzer - Windows Setup Script
setlocal enabledelayedexpansion

echo.
echo ==================================================
echo   Audio Feedback Analyzer - Windows Setup
echo ==================================================
echo.

REM Colors (limited in batch)
set "SUCCESS=[SUCCESS]"
set "ERROR=[ERROR]"
set "INFO=[INFO]"
set "WARNING=[WARNING]"

echo This script will set up the Audio Feedback Analyzer on Windows.
echo.
echo It will:
echo - Check Python installation
echo - Set up virtual environment
echo - Install Python packages
echo - Install and configure Ollama
echo - Download AI model
echo - Test the installation
echo.

set /p "continue=Continue? (y/N): "
if /i not "%continue%"=="y" (
    echo Setup cancelled.
    exit /b 0
)

echo.
echo %INFO% Starting setup...

REM Check Python installation
echo.
echo %INFO% Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Python not found in PATH.
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Python 3.8+ is required.
    echo Please upgrade your Python installation.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"') do set PYTHON_VERSION=%%i
echo %SUCCESS% Python %PYTHON_VERSION% found

REM Check if virtual environment already exists
if exist "audiofeedback_env" (
    echo %INFO% Virtual environment already exists
    set "VENV_EXISTS=1"
) else (
    echo %INFO% Creating virtual environment...
    python -m venv audiofeedback_env
    if errorlevel 1 (
        echo %ERROR% Failed to create virtual environment
        pause
        exit /b 1
    )
    set "VENV_EXISTS=0"
)

REM Activate virtual environment
echo %INFO% Activating virtual environment...
call audiofeedback_env\Scripts\activate.bat
if errorlevel 1 (
    echo %ERROR% Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo %INFO% Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

REM Install requirements
echo %INFO% Installing Python packages...
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo %WARNING% requirements.txt not found, installing core packages...
    pip install librosa pydub numpy scipy matplotlib seaborn requests ollama soundfile
)

if errorlevel 1 (
    echo %ERROR% Failed to install Python packages
    echo Try running: pip install -r requirements.txt
    pause
    exit /b 1
)

echo %SUCCESS% Python packages installed

REM Check FFmpeg
echo.
echo %INFO% Checking FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo %WARNING% FFmpeg not found in PATH
    echo.
    echo Please install FFmpeg:
    echo 1. Download from https://ffmpeg.org/download.html
    echo 2. Extract to a folder (e.g., C:\ffmpeg)
    echo 3. Add C:\ffmpeg\bin to your PATH environment variable
    echo.
    echo Or install with Chocolatey: choco install ffmpeg
    echo.
    set /p "continue_ffmpeg=Continue without FFmpeg? (y/N): "
    if /i not "!continue_ffmpeg!"=="y" (
        echo Setup paused. Please install FFmpeg and run setup again.
        pause
        exit /b 1
    )
) else (
    echo %SUCCESS% FFmpeg found
)

REM Install Ollama
echo.
echo %INFO% Checking Ollama installation...
where ollama >nul 2>&1
if errorlevel 1 (
    echo %INFO% Ollama not found. Installing...
    echo.
    echo Please download and install Ollama from:
    echo https://ollama.ai/download
    echo.
    echo After installation, restart this script.
    pause
    exit /b 1
) else (
    echo %SUCCESS% Ollama found
)

REM Start Ollama service
echo %INFO% Starting Ollama service...
start /b ollama serve >nul 2>&1

REM Wait for Ollama to start
echo %INFO% Waiting for Ollama to start...
timeout /t 5 /nobreak >nul

REM Download model
echo %INFO% Downloading Gemma 2 9B model (this may take a while)...
echo This is a large download (~5GB). Please be patient.
ollama pull gemma2:9b
if errorlevel 1 (
    echo %ERROR% Failed to download model
    echo Try running manually: ollama pull gemma2:9b
    pause
    exit /b 1
)

echo %SUCCESS% Model downloaded successfully

REM Generate test audio files
echo.
echo %INFO% Generating test audio files...
if exist "generate_test_audio.py" (
    python generate_test_audio.py -o .\test_audio\
    echo %SUCCESS% Test audio files created in .\test_audio\
) else (
    echo %WARNING% Test audio generator not found, skipping...
)

REM Run tests
echo.
echo %INFO% Running system tests...
if exist "test_complete_system.py" (
    python test_complete_system.py
) else (
    echo %INFO% Running basic tests...
    python -c "import librosa, pydub, numpy, matplotlib, ollama; print('✅ All packages imported successfully')"
    python -c "import ollama; client = ollama.Client(); models = client.list(); print('✅ Ollama connection successful')"
    echo %SUCCESS% Basic tests passed
)

REM Create convenience batch files
echo.
echo %INFO% Creating convenience scripts...

echo @echo off > run_analyzer.bat
echo call audiofeedback_env\Scripts\activate.bat >> run_analyzer.bat
echo python audiofeedback.py %%* >> run_analyzer.bat
echo Created run_analyzer.bat

echo @echo off > run_batch_process.bat
echo call audiofeedback_env\Scripts\activate.bat >> run_batch_process.bat
echo python batch_process.py %%* >> run_batch_process.bat
echo Created run_batch_process.bat

echo @echo off > activate_env.bat
echo call audiofeedback_env\Scripts\activate.bat >> activate_env.bat
echo cmd /k >> activate_env.bat
echo Created activate_env.bat

REM Display completion message
echo.
echo ==================================================
echo   Setup Complete!
echo ==================================================
echo.
echo Your Audio Feedback Analyzer is ready to use!
echo.
echo Quick start:
echo 1. Double-click 'activate_env.bat' to open activated environment
echo 2. Or use convenience scripts:
echo    - run_analyzer.bat your_song.wav
echo    - run_batch_process.bat .\audio_folder\ -o .\results\
echo.
echo Manual usage:
echo 1. Activate environment: audiofeedback_env\Scripts\activate.bat
echo 2. Run analyzer: python audiofeedback.py your_song.wav
echo 3. With visualization: python audiofeedback.py your_song.wav -v -o .\results\
echo.
echo Test with generated files:
echo run_analyzer.bat .\test_audio\01_sine_440hz.wav -v
echo.
echo Available models:
echo - gemma2:9b (recommended, already downloaded)
echo - gemma2:27b (higher quality, larger download)
echo.
echo For more information, see README.md and INSTALLATION.md
echo.

REM Check if we should keep the window open
if "%1"=="--no-pause" goto end
pause

:end
echo %SUCCESS% Setup completed successfully!
