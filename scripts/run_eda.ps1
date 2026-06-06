param(
    [string]$Config = "config/config.yaml"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv/Scripts/python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "Running EDA with config: $Config"
& $Python "scripts/4_eda.py" --config $Config

Write-Host ""
Write-Host "EDA selesai. Cek output di:"
Write-Host "  outputs/eda/"
Write-Host ""
Write-Host "File penting:"
Write-Host "  outputs/eda/class_distribution.png"
Write-Host "  outputs/eda/class_weights.json"
Write-Host "  outputs/eda/dataset_stats_clipped.json"
Write-Host "  outputs/eda/dominant_class_split_summary.csv"
Write-Host "  outputs/eda/eda_warnings.txt"
Write-Host "  outputs/eda/bad_tiles.txt"
Write-Host "  outputs/eda/band_stats_plot.png"
Write-Host "  outputs/eda/sample_visual_inspection.png"
Write-Host "  outputs/eda/red_nir_scatter.png"
Write-Host "  outputs/tiles/train_list_clean.txt"
Write-Host "  outputs/tiles/val_list_clean.txt"
