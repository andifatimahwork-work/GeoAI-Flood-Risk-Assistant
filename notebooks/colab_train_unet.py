# %% [markdown]
# # GeoAI Flood Susceptibility - Training U-Net di Google Colab
#
# Jalankan file ini cell-by-cell di Colab. Tujuan tahap ini adalah menghasilkan:
#
# - `unet_effb0_jabodetabek_exp2_best_model.pth`
# - checkpoint tiap 10 epoch
# - `training_log.csv`
#
# Pastikan folder project sudah ada di Google Drive dan `config/config.yaml` sudah menunjuk ke path Drive yang benar.

# %%
from google.colab import drive
drive.mount("/content/drive")

# %%
import subprocess
import sys

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "segmentation-models-pytorch",
        "rasterio",
        "PyYAML",
        "tqdm",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ]
)

# %%
import os
from pathlib import Path

PROJECT_DIR = Path("/content/drive/MyDrive/GeoAI_Flood_Susceptibility")
os.chdir(PROJECT_DIR)
print("Working directory:", Path.cwd())

# %%
sys.path.insert(0, str(PROJECT_DIR / "src"))

from geoai_flood.config import load_config
from geoai_flood.train import train_from_config

cfg = load_config(PROJECT_DIR / "config/config.yaml")
cfg["_project_root"] = str(PROJECT_DIR)
# Keep experiment-2 paths from config, resolved relative to PROJECT_DIR by the training code.

best = train_from_config(cfg)
print("Best model saved at:", best)

# %% [markdown]
# Setelah selesai, pastikan file berikut ada di Drive:
#
# - `outputs/checkpoints_exp2/unet_effb0_jabodetabek_exp2_best_model.pth`
# - `outputs/training_log_exp2.csv`
