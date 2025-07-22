"""
Data Preparation Module for Time Series Analysis
"""

import pandas as pd # type: ignore
import numpy as np # type: ignore
from typing import Tuple, Optional, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler # type: ignore

def load_and_preprocess_data(file_path: str, datetime_col: str = 'Datetime', target_col: str = 'MW') -> pd.DataFrame:
    """
    Load and preprocess time series data
    
    Args:
        file_path: Path to the data file
        datetime_col: Name of datetime column
        target_col: Name of target column
        
    Returns:
        Preprocessed DataFrame
    """
    # Load data
    data = pd.read_csv(file_path)
    
    # Rename columns for consistency
    data.columns = ['Datetime', 'MW']
    
    # Convert datetime and set as index
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.sort_values('Datetime').reset_index(drop=True)
    data.set_index('Datetime', inplace=True)
    
    # Handle missing values
    if data.isnull().sum().sum() > 0:
        data = data.interpolate(method='time')
    
    return data

def normalize_data(data: pd.DataFrame, method: str = 'minmax', target_col: str = 'MW') -> Tuple[pd.DataFrame, Any]:
    """
    Normalize time series data
    
    Args:
        data: Input DataFrame
        method: Normalization method ('minmax' or 'standard')
        target_col: Name of target column
        
    Returns:
        Tuple of (normalized DataFrame, scaler object)
    """
    processed_data = data.copy()
    
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Unsupported normalization method. Use 'minmax' or 'standard'")
    
    processed_data[target_col] = scaler.fit_transform(processed_data[[target_col]])
    
    return processed_data, scaler

def create_data_splits(data: pd.DataFrame, 
                      train_split: float = 0.7,
                      val_split: float = 0.15,
                      test_split: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation/test splits for time series data
    
    Args:
        data: Input DataFrame
        train_split: Training data proportion
        val_split: Validation data proportion
        test_split: Test data proportion
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    n = len(data)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    train_df = data.iloc[:train_end]
    val_df = data.iloc[train_end:val_end]
    test_df = data.iloc[val_end:]
    
    return train_df, val_df, test_df 