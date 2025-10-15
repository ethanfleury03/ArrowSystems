@echo off
REM DuraFlex Technical Assistant - Startup Script (Windows)
REM Usage: start.bat

echo ==================================
echo DuraFlex Technical Assistant
echo ==================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Virtual environment not found!
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies are installed
if not exist "venv\installed.flag" (
    echo Installing dependencies...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    type nul > venv\installed.flag
    echo Dependencies installed
) else (
    echo Dependencies already installed
)

REM Check if storage exists
if not exist "storage\" (
    echo.
    echo Storage directory not found!
    echo You need to run ingestion first:
    echo   python ingest.py
    echo.
    set /p REPLY="Do you want to run ingestion now? (y/n): "
    if /i "%REPLY%"=="y" (
        python ingest.py
    ) else (
        echo Starting without index - queries will fail
    )
)

REM Start application
echo.
echo Starting DuraFlex Technical Assistant...
echo Application will be available at: http://localhost:8501
echo Default login: admin / admin123
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py

