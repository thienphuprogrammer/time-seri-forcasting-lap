"""
Test package for time series forecasting.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Test data fixtures
@pytest.fixture
def sample_data():
    """Create sample time series data for testing."""
    dates = pd.date_range('2020-01-01', periods=1000, freq='H')
    values = np.random.normal(1000, 100, size=1000)
    return pd.DataFrame({
        'Datetime': dates,
        'MW': values
    })

@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        'input_width': 24,
        'label_width': 1,
        'shift': 1,
        'batch_size': 32,
        'epochs': 10,  # Reduced for testing
        'patience': 5
    }

# Test utilities
def create_test_data(n_samples=1000, noise_level=0.1):
    """Create synthetic test data."""
    np.random.seed(42)
    time_steps = np.arange(n_samples)
    
    # Create trend + seasonal + noise
    trend = 0.1 * time_steps
    seasonal = 100 * np.sin(2 * np.pi * time_steps / 24)
    noise = noise_level * np.random.normal(0, 1, n_samples)
    
    values = 1000 + trend + seasonal + noise
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='H')
    
    return pd.DataFrame({
        'Datetime': dates,
        'MW': values
    })

__all__ = ['sample_data', 'sample_config', 'create_test_data'] 