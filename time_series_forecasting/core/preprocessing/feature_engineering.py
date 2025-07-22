"""
Feature Engineering Module for Time Series Data
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

class TimeSeriesFeatureEngineer:
    """
    Class for creating time series features.
    """
    
    def __init__(self):
        """Initialize TimeSeriesFeatureEngineer."""
        self.data = None
        self.features = {}
        self.feature_history = []
    
    def create_features(self, data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Create time series features.
        
        Args:
            data: Input DataFrame with datetime index
            features: List of features to create
            
        Returns:
            DataFrame with additional features
        """
        self.data = data.copy()
        
        for feature in features:
            if feature in self._available_features():
                feature_func = getattr(self, f"_create_{feature}_features")
                new_features = feature_func()
                self.features[feature] = list(new_features.columns)
                self.data = pd.concat([self.data, new_features], axis=1)
                
                self.feature_history.append({
                    'feature_type': feature,
                    'created_features': list(new_features.columns)
                })
        
        return self.data
    
    def _create_time_features(self) -> pd.DataFrame:
        """Create time-based features."""
        features = pd.DataFrame(index=self.data.index)
        
        # Basic time features
        features['hour'] = self.data.index.hour
        features['day'] = self.data.index.day
        features['month'] = self.data.index.month
        features['year'] = self.data.index.year
        features['dayofweek'] = self.data.index.dayofweek
        features['quarter'] = self.data.index.quarter
        
        # Cyclical features
        features['hour_sin'] = np.sin(2 * np.pi * features['hour']/24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour']/24)
        features['month_sin'] = np.sin(2 * np.pi * features['month']/12)
        features['month_cos'] = np.cos(2 * np.pi * features['month']/12)
        features['dayofweek_sin'] = np.sin(2 * np.pi * features['dayofweek']/7)
        features['dayofweek_cos'] = np.cos(2 * np.pi * features['dayofweek']/7)
        
        return features
    
    def _create_lag_features(self, lags: List[int] = [1, 24, 48, 168]) -> pd.DataFrame:
        """Create lagged features."""
        features = pd.DataFrame(index=self.data.index)
        
        for col in self.data.select_dtypes(include=[np.number]).columns:
            for lag in lags:
                features[f'{col}_lag_{lag}'] = self.data[col].shift(lag)
        
        return features
    
    def _create_rolling_features(self, windows: List[int] = [6, 12, 24, 48]) -> pd.DataFrame:
        """Create rolling statistics features."""
        features = pd.DataFrame(index=self.data.index)
        
        for col in self.data.select_dtypes(include=[np.number]).columns:
            for window in windows:
                features[f'{col}_rolling_mean_{window}'] = self.data[col].rolling(window=window).mean()
                features[f'{col}_rolling_std_{window}'] = self.data[col].rolling(window=window).std()
                features[f'{col}_rolling_min_{window}'] = self.data[col].rolling(window=window).min()
                features[f'{col}_rolling_max_{window}'] = self.data[col].rolling(window=window).max()
        
        return features
    
    def _create_diff_features(self) -> pd.DataFrame:
        """Create difference features."""
        features = pd.DataFrame(index=self.data.index)
        
        for col in self.data.select_dtypes(include=[np.number]).columns:
            features[f'{col}_diff_1'] = self.data[col].diff()
            features[f'{col}_diff_24'] = self.data[col].diff(24)  # Daily difference
            features[f'{col}_diff_168'] = self.data[col].diff(168)  # Weekly difference
            
            # Percentage changes
            features[f'{col}_pct_change_1'] = self.data[col].pct_change()
            features[f'{col}_pct_change_24'] = self.data[col].pct_change(24)
        
        return features
    
    def _create_seasonal_features(self) -> pd.DataFrame:
        """Create seasonal decomposition features."""
        from statsmodels.tsa.seasonal import seasonal_decompose
        features = pd.DataFrame(index=self.data.index)
        
        for col in self.data.select_dtypes(include=[np.number]).columns:
            try:
                decomposition = seasonal_decompose(self.data[col], period=24)
                features[f'{col}_trend'] = decomposition.trend
                features[f'{col}_seasonal'] = decomposition.seasonal
                features[f'{col}_residual'] = decomposition.resid
            except:
                print(f"Could not create seasonal features for {col}")
        
        return features
    
    def _available_features(self) -> List[str]:
        """Get list of available feature types."""
        return [
            'time',
            'lag',
            'rolling',
            'diff',
            'seasonal'
        ]
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about created features.
        
        Returns:
            Dictionary containing feature information
        """
        return {
            'feature_types': list(self.features.keys()),
            'total_features': sum(len(features) for features in self.features.values()),
            'features_by_type': self.features,
            'feature_history': self.feature_history
        }
    
    def __repr__(self) -> str:
        """String representation of TimeSeriesFeatureEngineer."""
        status = 'features created' if self.features else 'no features'
        return f"TimeSeriesFeatureEngineer(status='{status}', feature_types={list(self.features.keys())})" 