@echo off
title 6. LSTM - Air Quality Prediction Frontend
color 0b
echo =====================================================================
echo           6. LSTM - AIR QUALITY PREDICTION WITH STREAMLIT
echo =====================================================================
echo.
echo [1/2] Checking Python and packages...
python -c "import streamlit, tensorflow, plotly; print('[OK] All required packages verified.')"
if %errorlevel% neq 0 (
    echo [ERROR] Dependencies check failed. Please ensure tensorflow, streamlit, and plotly are installed.
    pause
    exit /b
)

echo.
echo [2/2] Launching Streamlit Web Dashboard at http://localhost:8501 ...
echo.
python -m streamlit run app.py
pause
