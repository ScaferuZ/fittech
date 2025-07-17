@echo off
echo ========================================
echo FitTech AI - Complete Training Pipeline
echo ========================================
echo.

echo Step 1: Training Models...
python train_model.py
if %errorlevel% neq 0 (
    echo Error during training!
    pause
    exit /b 1
)

echo.
echo Step 2: Generating Visualizations...
python run_visualizations.py
if %errorlevel% neq 0 (
    echo Error during visualization generation!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Pipeline completed successfully!
echo ========================================
echo.
echo Generated files:
echo - models/xgfitness_ai_model.pkl (Production)
echo - models/research_model_comparison.pkl (Research)
echo - training_data.csv
echo - model_comparison_summary.csv
echo - thesis_model_comparison_summary.txt
echo - visualizations/run_*/ (Visualization files)
echo.
pause 