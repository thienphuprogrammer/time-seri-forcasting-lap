from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from time_series_forecasting.utils.io import ensure_dir, list_files, get_file_info, save_pickle, load_pickle, save_json, load_json

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a file.
    """
    return pd.read_csv(file_path)

def save_data(data: pd.DataFrame, file_path: str) -> None:
    """
    Save data to a file.
    """
    data.to_csv(file_path, index=False)

def validate_data(data: pd.DataFrame) -> bool:
    """
    Validate data.
    """
    return data.notna().all()

def convert_datetime(data: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """
    Convert datetime column to datetime type.
    """
    data[datetime_col] = pd.to_datetime(data[datetime_col])
    return data


def load_time_series_data(file_path: str,
                         datetime_col: str = 'Datetime',
                         target_col: Optional[str] = None,
                         datetime_format: str = '%Y-%m-%d %H:%M:%S') -> pd.DataFrame:
    """
    Load time series data from file.
    
    Args:
        file_path: Path to data file
        datetime_col: Name of datetime column
        target_col: Name of target column
        datetime_format: Format of datetime strings
        
    Returns:
        DataFrame with processed data
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Parse datetime
    df[datetime_col] = pd.to_datetime(df[datetime_col], format=datetime_format)
    df.set_index(datetime_col, inplace=True)
    
    # Select target column if specified
    if target_col and target_col in df.columns:
        df = df[[target_col]]
    
    return df

def handle_missing_data(df: pd.DataFrame,
                       method: str = 'interpolate') -> pd.DataFrame:
    """
    Handle missing values in time series data.
    
    Args:
        df: Input DataFrame
        method: Method to handle missing values
        
    Returns:
        DataFrame with handled missing values
    """
    if method == 'interpolate':
        return df.interpolate(method='time')
    elif method == 'forward':
        return df.fillna(method='ffill')
    elif method == 'backward':
        return df.fillna(method='bfill')
    elif method == 'mean':
        return df.fillna(df.mean())
    else:
        raise ValueError(f"Unknown missing data handling method: {method}")

def normalize_data(data: np.ndarray,
                  method: str = 'minmax',
                  feature_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, Any]:
    """
    Normalize time series data.
    
    Args:
        data: Input data
        method: Normalization method
        feature_range: Range for MinMaxScaler
        
    Returns:
        Tuple of (normalized data, scaler)
    """
    # Reshape 1D data if needed
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    # Create scaler
    if method == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Fit and transform
    normalized_data = scaler.fit_transform(data)
    
    return normalized_data, scaler

def create_sequences(data: np.ndarray,
                    input_width: int,
                    label_width: int,
                    shift: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input/output sequences for time series model.
    
    Args:
        data: Input time series data
        input_width: Number of input time steps
        label_width: Number of output time steps
        shift: Shift between input and output
        
    Returns:
        Tuple of (input sequences, output sequences)
    """
    # Total size of window
    total_size = input_width + shift
    
    # Create sequences
    data_size = len(data)
    num_sequences = data_size - total_size + 1
    
    X = np.zeros((num_sequences, input_width, data.shape[1]))
    y = np.zeros((num_sequences, label_width))
    
    for i in range(num_sequences):
        X[i] = data[i:i+input_width]
        y[i] = data[i+shift:i+shift+label_width, 0]  # Assuming target is first column
    
    return X, y

def split_data(data: np.ndarray,
               train_split: float = 0.7,
               val_split: float = 0.15,
               test_split: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split time series data into train/validation/test sets.
    
    Args:
        data: Input data
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        
    Returns:
        Tuple of (train, validation, test) sets
    """
    assert np.isclose(train_split + val_split + test_split, 1.0)
    
    n = len(data)
    train_size = int(n * train_split)
    val_size = int(n * val_split)
    
    train = data[:train_size]
    val = data[train_size:train_size+val_size]
    test = data[train_size+val_size:]
    
    return train, val, test 