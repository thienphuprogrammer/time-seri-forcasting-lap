"""
Configuration settings for time series forecasting package.
"""

import os
from pathlib import Path
from typing import Optional

# Package information
PACKAGE_NAME = "time_series_forecasting"
VERSION = "1.0.0"
AUTHOR = "DAT301m Lab 4 Team"

# Directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"

# Data processing settings
DEFAULT_DATETIME_COL = "Datetime"
DEFAULT_TARGET_COL = "MW"
DEFAULT_TRAIN_SPLIT = 0.7
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_TEST_SPLIT = 0.15

# Model training settings
DEFAULT_INPUT_WIDTH = 24  # 24 hours
DEFAULT_LABEL_WIDTH = 1   # 1 hour
DEFAULT_SHIFT = 1         # 1 hour
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 10
DEFAULT_LEARNING_RATE = 0.001

# PJM specific settings
PJM_REGIONS = [
    "AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ", 
    "EKPC", "FE", "NI", "PJME", "PJMW", "PJM_Load"
]

# Plotting settings
PLOT_STYLE = "seaborn-v0_8"
PLOT_FIGSIZE = (12, 8)
PLOT_DPI = 300
PLOT_FORMAT = "png"

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "time_series_forecasting.log"

# Model settings
BASELINE_MODELS = ["linear", "arima", "sarima"]
DEEP_LEARNING_MODELS = ["rnn", "gru", "lstm"]
ATTENTION_MODELS = ["transformer", "attention"]

# Performance metrics
METRICS = ["rmse", "mae", "mape", "r2"]

# Random seed for reproducibility
RANDOM_SEED = 42

# GPU settings
USE_GPU = True
GPU_MEMORY_GROWTH = True

# Create directories if they don't exist
def create_directories():
    """Create necessary directories."""
    for directory in [DATA_DIR, RESULTS_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# Environment variables
def get_env_var(name: str, default: Optional[str] = None) -> str:
    """Get environment variable with default value."""
    return os.environ.get(name, default or "")

# Configuration dictionary
CONFIG = {
    "package": {
        "name": PACKAGE_NAME,
        "version": VERSION,
        "author": AUTHOR,
    },
    "directories": {
        "project_root": PROJECT_ROOT,
        "data": DATA_DIR,
        "results": RESULTS_DIR,
        "models": MODELS_DIR,
        "plots": PLOTS_DIR,
        "reports": REPORTS_DIR,
    },
    "data_processing": {
        "datetime_col": DEFAULT_DATETIME_COL,
        "target_col": DEFAULT_TARGET_COL,
        "train_split": DEFAULT_TRAIN_SPLIT,
        "val_split": DEFAULT_VAL_SPLIT,
        "test_split": DEFAULT_TEST_SPLIT,
    },
    "model_training": {
        "input_width": DEFAULT_INPUT_WIDTH,
        "label_width": DEFAULT_LABEL_WIDTH,
        "shift": DEFAULT_SHIFT,
        "batch_size": DEFAULT_BATCH_SIZE,
        "epochs": DEFAULT_EPOCHS,
        "patience": DEFAULT_PATIENCE,
        "learning_rate": DEFAULT_LEARNING_RATE,
    },
    "pjm": {
        "regions": PJM_REGIONS,
    },
    "plotting": {
        "style": PLOT_STYLE,
        "figsize": PLOT_FIGSIZE,
        "dpi": PLOT_DPI,
        "format": PLOT_FORMAT,
    },
    "logging": {
        "level": LOG_LEVEL,
        "format": LOG_FORMAT,
        "file": LOG_FILE,
    },
    "models": {
        "baseline": BASELINE_MODELS,
        "deep_learning": DEEP_LEARNING_MODELS,
        "attention": ATTENTION_MODELS,
    },
    "metrics": METRICS,
    "random_seed": RANDOM_SEED,
    "gpu": {
        "use_gpu": USE_GPU,
        "memory_growth": GPU_MEMORY_GROWTH,
    },
}

# Export configuration
__all__ = ["CONFIG", "create_directories", "get_env_var"] 