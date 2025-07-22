"""
Data Loader Module for Time Series Data
"""

import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path

class DataLoader:
    """
    Class for loading time series data from various sources.
    """
    
    def __init__(self, data_path: Optional[str] = None, region: Optional[str] = None):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to the data file
            region: Region to filter data
        """
        self.data_path = Path(data_path) if data_path else None
        self.region = region
        self.data = None
        self.original_columns = None
    
    def load_csv(self, data_path: Optional[str] = None, region: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            data_path: Path to CSV file
            region: Region to filter
            
        Returns:
            Loaded DataFrame
        """
        if data_path:
            self.data_path = Path(data_path)
        if region:
            self.region = region
            
        if not self.data_path:
            raise ValueError("Data path must be provided")
            
        # Load data
        self.data = pd.read_csv(self.data_path)
        self.original_columns = list(self.data.columns)
        
        # Auto-detect and standardize PJM data format
        if len(self.data.columns) == 2:
            datetime_col = self.data.columns[0]
            target_col = self.data.columns[1]
            
            # Rename columns for consistency
            self.data.columns = ['Datetime', 'MW']
            
            print(f"Auto-detected PJM format:")
            print(f"  {datetime_col} -> 'Datetime'")
            print(f"  {target_col} -> 'MW'")
            
            # Extract region from target column name if not specified
            if not self.region and '_MW' in target_col:
                self.region = target_col.replace('_MW', '')
                print(f"  Detected region: {self.region}")
        
        # Filter by region if specified
        if self.region and len(self.data.columns) > 2:
            if 'region' in self.data.columns:
                self.data = self.data[self.data['region'] == self.region]
            elif 'Region' in self.data.columns:
                self.data = self.data[self.data['Region'] == self.region]
        
        print(f"Loaded data shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        return self.data
    
    def load_parquet(self, data_path: Optional[str] = None, region: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from Parquet file.
        
        Args:
            data_path: Path to Parquet file
            region: Region to filter
            
        Returns:
            Loaded DataFrame
        """
        if data_path:
            self.data_path = Path(data_path)
        if region:
            self.region = region
            
        if not self.data_path:
            raise ValueError("Data path must be provided")
            
        # Load data
        self.data = pd.read_parquet(self.data_path)
        self.original_columns = list(self.data.columns)
        
        # Filter by region if specified
        if self.region and len(self.data.columns) > 2:
            if 'region' in self.data.columns:
                self.data = self.data[self.data['region'] == self.region]
            elif 'Region' in self.data.columns:
                self.data = self.data[self.data['Region'] == self.region]
        
        print(f"Loaded data shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        return self.data
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded data.
        
        Returns:
            Dictionary containing data information
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'original_columns': self.original_columns,
            'dtypes': self.data.dtypes.to_dict(),
            'region': self.region,
            'file_path': str(self.data_path)
        }
    
    def __repr__(self) -> str:
        """String representation of DataLoader."""
        status = 'loaded' if self.data is not None else 'not loaded'
        return f"DataLoader(data_path='{self.data_path}', region='{self.region}', status='{status}')" 