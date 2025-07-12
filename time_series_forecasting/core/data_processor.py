import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Class for handling data loading, preprocessing, and visualization for time series forecasting.
    """
    
    def __init__(self, data_path: Optional[str] = None, region: Optional[str] = None):
        """
        Initialize DataProcessor.
        
        Args:
            data_path: Path to the CSV file containing energy consumption data
            region: Specific region to filter data (e.g., 'PJM_Load_hourly')
        """
        self.data_path = data_path
        self.region = region
        self.data = None
        self.scaler = None
        self.processed_data = None
        self.original_target_col = None
        
    def load_data(self, data_path: Optional[str] = None, region: Optional[str] = None) -> pd.DataFrame:
        """
        Load and filter energy consumption data.
        
        Args:
            data_path: Path to the CSV file
            region: Region to filter (if None, uses self.region)
            
        Returns:
            DataFrame with energy consumption data
        """
        if data_path:
            self.data_path = data_path
        if region:
            self.region = region
            
        if not self.data_path:
            raise ValueError("Data path must be provided")
            
        # Load data
        self.data = pd.read_csv(self.data_path)
        
        # Auto-detect and standardize PJM data format
        if len(self.data.columns) == 2:
            datetime_col = self.data.columns[0]
            target_col = self.data.columns[1]
            
            # Store original column name for reference
            self.original_target_col = target_col
            
            # Rename columns for consistency
            self.data.columns = ['Datetime', 'MW']
            
            print(f"Auto-detected PJM format:")
            print(f"  {datetime_col} -> 'Datetime'")
            print(f"  {target_col} -> 'MW'")
            
            # Extract region from target column name if not specified
            if not self.region and '_MW' in target_col:
                self.region = target_col.replace('_MW', '')
                print(f"  Detected region: {self.region}")
        
        # Filter by region if specified (for multi-region files)
        if self.region and len(self.data.columns) > 2:
            if 'region' in self.data.columns:
                self.data = self.data[self.data['region'] == self.region]
            elif 'Region' in self.data.columns:
                self.data = self.data[self.data['Region'] == self.region]
        
        print(f"Loaded data shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        return self.data
    
    def parse_datetime(self, datetime_col: str = 'Datetime', format: str = '%Y-%m-%d %H:%M:%S') -> pd.DataFrame:
        """
        Parse datetime column and set as index.
        
        Args:
            datetime_col: Name of the datetime column
            format: Datetime format string
            
        Returns:
            DataFrame with parsed datetime index
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
            
        # Convert to datetime
        self.data[datetime_col] = pd.to_datetime(self.data[datetime_col], format=format)
        
        # Set as index
        self.data.set_index(datetime_col, inplace=True)
        
        # Sort by datetime
        self.data.sort_index(inplace=True)
        
        print(f"Parsed datetime. Date range: {self.data.index.min()} to {self.data.index.max()}")
        
        return self.data
    
    def handle_missing_data(self, method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            method: Method to handle missing data ('interpolate', 'ffill', 'bfill', 'drop')
            
        Returns:
            DataFrame with missing data handled
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
            
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
        
        return self.data
    
    def normalize_data(self, method: str = 'minmax', columns: list = None) -> pd.DataFrame:
        """
        Normalize the data using specified method.
        
        Args:
            method: Normalization method ('minmax', 'standard', 'robust')
            columns: Columns to normalize (if None, normalizes all numeric columns)
            
        Returns:
            DataFrame with normalized data
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
            
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            
        data_to_normalize = self.data[columns].copy()
        
        if method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        # Fit and transform
        normalized_data = self.scaler.fit_transform(data_to_normalize)
        self.processed_data = pd.DataFrame(normalized_data, 
                                         index=self.data.index, 
                                         columns=columns)
        
        print(f"Normalized data using {method} scaling")
        
        return self.processed_data
    
    def create_visualizations(self, save_path: str = None) -> Dict[str, Any]:
        """
        Create comprehensive visualizations for time series analysis.
        
        Args:
            save_path: Path to save plots (if None, displays them)
            
        Returns:
            Dictionary containing plot objects
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
            
        plots = {}
        
        # Get the main energy consumption column
        energy_col = None
        for col in self.data.columns:
            if any(keyword in col.lower() for keyword in ['mw', 'load', 'consumption', 'energy']):
                energy_col = col
                break
        
        if energy_col is None:
            energy_col = self.data.select_dtypes(include=[np.number]).columns[0]
        
        # 1. Time series plot
        fig, ax = plt.subplots(figsize=(15, 6))
        self.data[energy_col].plot(ax=ax)
        ax.set_title(f'{energy_col} Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Energy Consumption')
        plt.tight_layout()
        plots['time_series'] = fig
        
        if save_path:
            plt.savefig(f"{save_path}_time_series.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Seasonal decomposition
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Resample to daily if hourly data
        if len(self.data) > 8760:  # More than a year of hourly data
            daily_data = self.data[energy_col].resample('D').mean()
        else:
            daily_data = self.data[energy_col]
        
        decomposition = seasonal_decompose(daily_data, period=24 if len(daily_data) > 24 else 7)
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        decomposition.observed.plot(ax=axes[0])
        axes[0].set_title('Original')
        decomposition.trend.plot(ax=axes[1])
        axes[1].set_title('Trend')
        decomposition.seasonal.plot(ax=axes[2])
        axes[2].set_title('Seasonal')
        decomposition.resid.plot(ax=axes[3])
        axes[3].set_title('Residual')
        plt.tight_layout()
        plots['decomposition'] = fig
        
        if save_path:
            plt.savefig(f"{save_path}_decomposition.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Distribution analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        axes[0].hist(self.data[energy_col], bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_title(f'Distribution of {energy_col}')
        axes[0].set_xlabel('Energy Consumption')
        axes[0].set_ylabel('Frequency')
        
        # Box plot
        axes[1].boxplot(self.data[energy_col])
        axes[1].set_title(f'Box Plot of {energy_col}')
        axes[1].set_ylabel('Energy Consumption')
        
        plt.tight_layout()
        plots['distribution'] = fig
        
        if save_path:
            plt.savefig(f"{save_path}_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Interactive plotly visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Time Series', 'Daily Pattern', 'Weekly Pattern', 'Monthly Pattern'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Time series
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data[energy_col], name='Energy Consumption'),
            row=1, col=1
        )
        
        # Daily pattern
        daily_pattern = self.data[energy_col].groupby(self.data.index.hour).mean()
        fig.add_trace(
            go.Scatter(x=daily_pattern.index, y=daily_pattern.values, name='Daily Pattern'),
            row=1, col=2
        )
        
        # Weekly pattern
        weekly_pattern = self.data[energy_col].groupby(self.data.index.dayofweek).mean()
        fig.add_trace(
            go.Scatter(x=weekly_pattern.index, y=weekly_pattern.values, name='Weekly Pattern'),
            row=2, col=1
        )
        
        # Monthly pattern
        monthly_pattern = self.data[energy_col].groupby(self.data.index.month).mean()
        fig.add_trace(
            go.Scatter(x=monthly_pattern.index, y=monthly_pattern.values, name='Monthly Pattern'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Energy Consumption Patterns")
        plots['interactive'] = fig
        
        if save_path:
            fig.write_html(f"{save_path}_interactive.html")
        
        return plots
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
            
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'date_range': {
                'start': self.data.index.min(),
                'end': self.data.index.max(),
                'duration': self.data.index.max() - self.data.index.min()
            },
            'numeric_stats': self.data.describe().to_dict()
        }
        
        return info
    
    def prepare_for_forecasting(self, target_column: str = None) -> pd.DataFrame:
        """
        Prepare the final dataset for forecasting.
        
        Args:
            target_column: Target column for forecasting
            
        Returns:
            DataFrame ready for forecasting
        """
        if self.processed_data is not None:
            return self.processed_data
        
        if self.data is None:
            raise ValueError("Data must be loaded first")
            
        # If no target column specified, use the first numeric column
        if target_column is None:
            target_column = self.data.select_dtypes(include=[np.number]).columns[0]
        
        # Create a simple DataFrame with just the target column
        self.processed_data = pd.DataFrame({
            'target': self.data[target_column]
        }, index=self.data.index)
        
        return self.processed_data 