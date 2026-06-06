@echo off
setlocal

cd /d "%~dp0.."

if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

echo Running EDA...
%PYTHON% scripts\4_eda.py --config config\config.yaml

echo.
echo EDA selesai. Cek output di outputs\eda\
echo File penting:
echo   outputs\eda\class_distribution.png
echo   outputs\eda\class_weights.json
echo   outputs\eda\dataset_stats_clipped.json
echo   outputs\eda\dominant_class_split_summary.csv
echo   outputs\eda\eda_warnings.txt
echo   outputs\eda\bad_tiles.txt
echo   outputs\eda\band_stats_plot.png
echo   outputs\eda\sample_visual_inspection.png
echo   outputs\eda\red_nir_scatter.png
echo   outputs\tiles\train_list_clean.txt
echo   outputs\tiles\val_list_clean.txt

endlocal
