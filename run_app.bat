@echo off
echo Starting Frequent Items Mining Application...
echo.

echo Starting FastAPI backend server...
start "FastAPI Backend" cmd /k "python main.py"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo Starting Streamlit frontend...
start "Streamlit Frontend" cmd /k "streamlit run streamlit_app.py"

echo.
echo Both services are starting...
echo FastAPI Backend: http://localhost:8000
echo Streamlit Frontend: http://localhost:8501
echo API Documentation: http://localhost:8000/docs
echo.
echo Press any key to exit...
pause > nul