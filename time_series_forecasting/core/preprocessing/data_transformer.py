"""
Data Transformer Module for Time Series Data
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

class DataTransformer:
    """
    Class for transforming time series data (scaling, normalization, etc.).
    """
    
    def __init__(self):
        """Initialize DataTransformer."""
        self.data = None
        self.scalers = {}
        self.transform_history = []
    
    def fit_transform(self, data: pd.DataFrame, method: str = 'minmax', columns: Optional[list] = None) -> pd.DataFrame:
        """
        Fit and transform the data using specified method.
        
        Args:
            data: Input DataFrame
            method: Transformation method ('minmax', 'standard', 'robust')
            columns: Columns to transform (if None, transforms all numeric columns)
            
        Returns:
            Transformed DataFrame
        """
        self.data = data.copy()
        
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        # Create and fit scalers
        for col in columns:
            scaler = self._get_scaler(method)
            self.data[col] = scaler.fit_transform(self.data[[col]])
            self.scalers[col] = scaler
            
            self.transform_history.append({
                'step': 'fit_transform',
                'method': method,
                'column': col
            })
        
        print(f"Transformed {len(columns)} columns using {method} scaling")
        
        return self.data
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted scalers.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self.scalers:
            raise ValueError("No fitted scalers available. Call fit_transform first.")
        
        result = data.copy()
        
        for col, scaler in self.scalers.items():
            if col in result.columns:
                result[col] = scaler.transform(result[[col]])
        
        return result
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform data using fitted scalers.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Inverse transformed DataFrame
        """
        if not self.scalers:
            raise ValueError("No fitted scalers available. Call fit_transform first.")
        
        result = data.copy()
        
        for col, scaler in self.scalers.items():
            if col in result.columns:
                result[col] = scaler.inverse_transform(result[[col]])
        
        return result
    
    def _get_scaler(self, method: str):
        """
        Get scaler instance based on method.
        
        Args:
            method: Scaling method
            
        Returns:
            Scaler instance
        """
        if method == 'minmax':
            return MinMaxScaler()
        elif method == 'standard':
            return StandardScaler()
        elif method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def get_transform_info(self) -> Dict[str, Any]:
        """
        Get information about the transformations.
        
        Returns:
            Dictionary containing transformation information
        """
        return {
            'transform_steps': len(self.transform_history),
            'transform_history': self.transform_history,
            'transformed_columns': list(self.scalers.keys())
        }
    
    def save_scalers(self, path: str) -> None:
        """
        Save fitted scalers to file.
        
        Args:
            path: Path to save scalers
        """
        import joblib
        joblib.dump(self.scalers, path)
        print(f"Saved scalers to {path}")
    
    def load_scalers(self, path: str) -> None:
        """
        Load fitted scalers from file.
        
        Args:
            path: Path to load scalers from
        """
        import joblib
        self.scalers = joblib.load(path)
        print(f"Loaded scalers from {path}")
    
    def __repr__(self) -> str:
        """String representation of DataTransformer."""
        status = 'fitted' if self.scalers else 'not fitted'
        return f"DataTransformer(status='{status}', columns={list(self.scalers.keys()) if self.scalers else None})" 