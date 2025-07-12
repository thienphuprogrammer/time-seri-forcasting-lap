"""
Time Series Forecasting Package

A professional time series forecasting solution for PJM energy consumption data.
This package provides comprehensive tools for data processing, model training,
and analysis of time series data.

Authors: DAT301m Lab 4 Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "DAT301m Lab 4 Team"
__email__ = "lab4@example.com"
__description__ = "Professional Time Series Forecasting Solution for PJM Energy Data"

# Import main components
from .core.data_processor import DataProcessor
from .core.window_generator import WindowGenerator
from .models.model_factory import ModelFactory
from .models.model_trainer import ModelTrainer
from .pipeline.forecasting_pipeline import ForecastingPipeline
from .analysis.pjm_analyzer import PJMDataAnalyzer
from .analysis.lab4_interface import DAT301mLab4Interface

# Package-level configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Public API
__all__ = [
    "DataProcessor",
    "WindowGenerator", 
    "ModelFactory",
    "ModelTrainer",
    "ForecastingPipeline",
    "PJMDataAnalyzer",
    "DAT301mLab4Interface",
] 