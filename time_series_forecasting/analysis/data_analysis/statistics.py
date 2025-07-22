"""
Statistics Module for Time Series Analysis
"""

import pandas as pd # type: ignore
import numpy as np # type: ignore
from typing import Dict, Any, Optional, List
from statsmodels.tsa.seasonal import seasonal_decompose # type: ignore

def get_basic_statistics(data: pd.DataFrame, target_col: str = 'MW') -> Dict[str, Any]:
    """
    Get basic statistics of time series data.
    
    Args:
        data: Input DataFrame
        target_col: Target column name
        
    Returns:
        Dictionary containing basic statistics
    """
    stats = {
        'total_records': len(data),
        'date_range': {
            'start': data.index.min(),
            'end': data.index.max(),
            'duration': data.index.max() - data.index.min()
        },
        'missing_values': data[target_col].isnull().sum(),
        'duplicates': data.index.duplicated().sum(),
        'descriptive_stats': data[target_col].describe().to_dict()
    }
    
    return stats

def get_seasonal_statistics(data: pd.DataFrame, target_col: str = 'MW') -> Dict[str, pd.Series]:
    """
    Get seasonal statistics of time series data.
    
    Args:
        data: Input DataFrame
        target_col: Target column name
        
    Returns:
        Dictionary containing seasonal statistics
    """
    stats = {
        'monthly_avg': data[target_col].groupby(data.index.month).mean(),
        'daily_avg': data[target_col].groupby(data.index.dayofweek).mean(),
        'hourly_avg': data[target_col].groupby(data.index.hour).mean(),
        'monthly_std': data[target_col].groupby(data.index.month).std(),
        'daily_std': data[target_col].groupby(data.index.dayofweek).std(),
        'hourly_std': data[target_col].groupby(data.index.hour).std()
    }
    
    return stats

def get_rolling_statistics(data: pd.DataFrame,
                         window: int = 24*7,
                         target_col: str = 'MW') -> Dict[str, pd.Series]:
    """
    Get rolling statistics of time series data.
    
    Args:
        data: Input DataFrame
        window: Rolling window size
        target_col: Target column name
        
    Returns:
        Dictionary containing rolling statistics
    """
    stats = {
        'rolling_mean': data[target_col].rolling(window=window).mean(),
        'rolling_std': data[target_col].rolling(window=window).std(),
        'rolling_min': data[target_col].rolling(window=window).min(),
        'rolling_max': data[target_col].rolling(window=window).max()
    }
    
    return stats

def analyze_data_distribution(data: pd.DataFrame, target_col: str = 'MW') -> Dict[str, Any]:
    """
    Analyze distribution of time series data.
    
    Args:
        data: Input DataFrame
        target_col: Target column name
        
    Returns:
        Dictionary containing distribution analysis
    """
    from scipy import stats # type: ignore
    
    series = data[target_col].dropna()
    
    analysis = {
        'normality_test': {
            'statistic': float(stats.normaltest(series)[0]),
            'p_value': float(stats.normaltest(series)[1])
        },
        'skewness': float(series.skew()),
        'kurtosis': float(series.kurtosis()),
        'percentiles': {
            '1%': float(np.percentile(series, 1)),
            '5%': float(np.percentile(series, 5)),
            '25%': float(np.percentile(series, 25)),
            '50%': float(np.percentile(series, 50)),
            '75%': float(np.percentile(series, 75)),
            '95%': float(np.percentile(series, 95)),
            '99%': float(np.percentile(series, 99))
        }
    }
    
    return analysis

def detect_anomalies(data: pd.DataFrame,
                    target_col: str = 'MW',
                    method: str = 'zscore',
                    threshold: float = 3.0) -> Dict[str, Any]:
    """
    Detect anomalies in time series data.
    
    Args:
        data: Input DataFrame
        target_col: Target column name
        method: Detection method ('zscore' or 'iqr')
        threshold: Detection threshold
        
    Returns:
        Dictionary containing anomaly detection results
    """
    series = data[target_col].dropna()
    
    if method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        anomalies = z_scores > threshold
    elif method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        anomalies = (series < (Q1 - threshold * IQR)) | (series > (Q3 + threshold * IQR))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    results = {
        'total_anomalies': int(anomalies.sum()),
        'anomaly_indices': data.index[anomalies].tolist(),
        'anomaly_values': series[anomalies].tolist(),
        'anomaly_percentage': float(anomalies.mean() * 100)
    }
    
    return results

def detect_seasonality(data: pd.DataFrame,
                      target_col: str = 'MW',
                      periods: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Detect seasonality in time series data.
    
    Args:
        data: Input DataFrame
        target_col: Target column name
        periods: List of periods to check
        
    Returns:
        Dictionary containing seasonality detection results
    """
    from statsmodels.tsa.stattools import acf # type: ignore
    
    series = data[target_col].dropna()
    periods = periods or [24, 24*7, 24*30]  # Default: daily, weekly, monthly
    
    # Calculate autocorrelation
    acf_values = acf(series, nlags=max(periods))
    
    # Check seasonality for each period
    seasonality = {}
    for period in periods:
        if period >= len(acf_values):
            continue
        
        correlation = acf_values[period]
        seasonality[period] = {
            'correlation': float(correlation),
            'strength': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.3 else 'weak'
        }
    
    # Perform seasonal decomposition
    try:
        decomposition = seasonal_decompose(series, period=24)
        seasonal_strength = float(np.std(decomposition.seasonal) / np.std(series))
    except:
        seasonal_strength = None
    
    results = {
        'seasonality_by_period': seasonality,
        'seasonal_strength': seasonal_strength,
        'has_seasonality': any(
            isinstance(s['correlation'], float) and s['correlation'] > 0.3 
            for s in seasonality.values()
        )
    }
    
    return results

def detect_trends(data: pd.DataFrame, target_col: str = 'MW') -> Dict[str, Any]:
    """
    Detect trends in time series data.
    
    Args:
        data: Input DataFrame
        target_col: Target column name
        
    Returns:
        Dictionary containing trend detection results
    """
    from scipy import stats # type: ignore
    
    series = data[target_col].dropna()
    
    # Linear regression
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
    
    # Mann-Kendall test
    from scipy.stats import kendalltau # type: ignore
    tau, mk_p_value = kendalltau(X.flatten(), y)
    
    # Calculate trend strength
    trend_strength = abs(r_value)
    trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'no trend'
    
    results = {
        'linear_regression': {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value**2),
            'p_value': float(p_value),
            'std_error': float(std_err)
        },
        'mann_kendall': {
            'tau': float(tau),
            'p_value': float(mk_p_value)
        },
        'trend_summary': {
            'direction': trend_direction,
            'strength': trend_strength,
            'significant': p_value < 0.05 and mk_p_value < 0.05
        }
    }
    
    return results 