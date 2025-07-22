"""
Data Cleaner Module for Time Series Data
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

class DataCleaner:
    """
    Class for cleaning and preprocessing time series data.
    """
    
    def __init__(self):
        """Initialize DataCleaner."""
        self.data = None
        self.cleaning_history = []
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input data by applying all cleaning steps.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.data = data.copy()
        
        # Apply cleaning steps
        self.parse_datetime()
        self.handle_missing_values()
        self.remove_duplicates()
        self.handle_outliers()
        
        return self.data
    
    def parse_datetime(self, datetime_col: str = 'Datetime', format: str = '%Y-%m-%d %H:%M:%S') -> pd.DataFrame:
        """
        Parse datetime column and set as index.
        
        Args:
            datetime_col: Name of datetime column
            format: Datetime format string
            
        Returns:
            DataFrame with parsed datetime index
        """
        if self.data is None:
            raise ValueError("No data to process")
            
        # Convert to datetime
        self.data[datetime_col] = pd.to_datetime(self.data[datetime_col], format=format)
        
        # Set as index
        self.data.set_index(datetime_col, inplace=True)
        
        # Sort by datetime
        self.data.sort_index(inplace=True)
        
        self.cleaning_history.append({
            'step': 'parse_datetime',
            'column': datetime_col,
            'format': format
        })
        
        print(f"Parsed datetime. Date range: {self.data.index.min()} to {self.data.index.max()}")
        
        return self.data
    
    def handle_missing_values(self, method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            method: Method to handle missing data ('interpolate', 'ffill', 'bfill', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        if self.data is None:
            raise ValueError("No data to process")
            
        missing_before = self.data.isnull().sum().sum()
        print(f"Missing values before handling: {missing_before}")
        
        if method == 'interpolate':
            self.data = self.data.interpolate(method='time')
        elif method == 'ffill':
            self.data = self.data.fillna(method='ffill')
        elif method == 'bfill':
            self.data = self.data.fillna(method='bfill')
        elif method == 'drop':
            self.data = self.data.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
            
        missing_after = self.data.isnull().sum().sum()
        print(f"Missing values after handling: {missing_after}")
        
        self.cleaning_history.append({
            'step': 'handle_missing_values',
            'method': method,
            'missing_before': missing_before,
            'missing_after': missing_after
        })
        
        return self.data
    
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Returns:
            DataFrame with duplicates removed
        """
        if self.data is None:
            raise ValueError("No data to process")
            
        duplicates_before = len(self.data)
        self.data = self.data[~self.data.index.duplicated(keep='first')]
        duplicates_after = len(self.data)
        
        duplicates_removed = duplicates_before - duplicates_after
        print(f"Removed {duplicates_removed} duplicate rows")
        
        self.cleaning_history.append({
            'step': 'remove_duplicates',
            'duplicates_removed': duplicates_removed
        })
        
        return self.data
    
    def handle_outliers(self, method: str = 'zscore', threshold: float = 3.0) -> pd.DataFrame:
        """
        Handle outliers in the dataset.
        
        Args:
            method: Method to detect outliers ('zscore', 'iqr')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        if self.data is None:
            raise ValueError("No data to process")
            
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        outliers_removed = 0
        
        for col in numeric_cols:
            if method == 'zscore':
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outliers = z_scores > threshold
            elif method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (self.data[col] < (Q1 - threshold * IQR)) | (self.data[col] > (Q3 + threshold * IQR))
            else:
                raise ValueError(f"Unknown method: {method}")
            
            outliers_removed += outliers.sum()
            self.data.loc[outliers, col] = np.nan
        
        # Handle the newly created missing values
        self.handle_missing_values()
        
        print(f"Removed {outliers_removed} outliers using {method} method")
        
        self.cleaning_history.append({
            'step': 'handle_outliers',
            'method': method,
            'threshold': threshold,
            'outliers_removed': outliers_removed
        })
        
        return self.data
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """
        Get summary of cleaning operations.
        
        Returns:
            Dictionary containing cleaning summary
        """
        return {
            'cleaning_steps': len(self.cleaning_history),
            'cleaning_history': self.cleaning_history,
            'final_shape': self.data.shape if self.data is not None else None
        } 