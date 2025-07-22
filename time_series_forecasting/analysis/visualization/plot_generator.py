"""
Plot Generator Module for Time Series Analysis
"""

import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import os

def plot_time_series(data: pd.DataFrame,
                    target_col: str = 'MW',
                    title: Optional[str] = None,
                    figsize: Tuple[int, int] = (15, 6),
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot time series data.
    
    Args:
        data: Input DataFrame
        target_col: Target column name
        title: Plot title
        figsize: Figure size
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot time series
    data[target_col].plot(ax=ax)
    
    # Customize plot
    ax.set_title(title or f'{target_col} Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel(target_col)
    ax.grid(True)
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_seasonal_patterns(data: pd.DataFrame,
                         target_col: str = 'MW',
                         figsize: Tuple[int, int] = (15, 10),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot seasonal patterns in time series data.
    
    Args:
        data: Input DataFrame
        target_col: Target column name
        figsize: Figure size
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Monthly pattern
    monthly_avg = data[target_col].groupby(data.index.month).mean()
    monthly_std = data[target_col].groupby(data.index.month).std()
    axes[0, 0].errorbar(monthly_avg.index, monthly_avg.values, yerr=monthly_std.values, fmt='o-')
    axes[0, 0].set_title('Monthly Pattern')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel(target_col)
    
    # Daily pattern
    daily_avg = data[target_col].groupby(data.index.dayofweek).mean()
    daily_std = data[target_col].groupby(data.index.dayofweek).std()
    axes[0, 1].errorbar(daily_avg.index, daily_avg.values, yerr=daily_std.values, fmt='o-')
    axes[0, 1].set_title('Daily Pattern')
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel(target_col)
    
    # Hourly pattern
    hourly_avg = data[target_col].groupby(data.index.hour).mean()
    hourly_std = data[target_col].groupby(data.index.hour).std()
    axes[1, 0].errorbar(hourly_avg.index, hourly_avg.values, yerr=hourly_std.values, fmt='o-')
    axes[1, 0].set_title('Hourly Pattern')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].set_ylabel(target_col)
    
    # Heatmap
    pivot_data = data.pivot_table(
        values=target_col,
        index=data.index.hour,
        columns=data.index.dayofweek,
        aggfunc='mean'
    )
    sns.heatmap(pivot_data, ax=axes[1, 1], cmap='YlOrRd', center=pivot_data.mean().mean())
    axes[1, 1].set_title('Hour vs Day Heatmap')
    axes[1, 1].set_xlabel('Day of Week')
    axes[1, 1].set_ylabel('Hour')
    
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_distribution(data: pd.DataFrame,
                     target_col: str = 'MW',
                     figsize: Tuple[int, int] = (15, 5),
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of time series data.
    
    Args:
        data: Input DataFrame
        target_col: Target column name
        figsize: Figure size
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Histogram
    sns.histplot(data[target_col], kde=True, ax=axes[0])
    axes[0].set_title('Distribution')
    axes[0].set_xlabel(target_col)
    
    # Box plot
    sns.boxplot(y=data[target_col], ax=axes[1])
    axes[1].set_title('Box Plot')
    axes[1].set_ylabel(target_col)
    
    # Q-Q plot
    from scipy import stats # type: ignore
    stats.probplot(data[target_col].dropna(), dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot')
    
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_trends(data: pd.DataFrame,
                target_col: str = 'MW',
                window: int = 24*7,
                figsize: Tuple[int, int] = (15, 10),
                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot trends in time series data.
    
    Args:
        data: Input DataFrame
        target_col: Target column name
        window: Rolling window size
        figsize: Figure size
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Original data with trend line
    data[target_col].plot(ax=axes[0], alpha=0.5, label='Original')
    
    # Add trend line
    X = np.arange(len(data)).reshape(-1, 1)
    from sklearn.linear_model import LinearRegression # type: ignore
    model = LinearRegression()
    model.fit(X, data[target_col])
    trend = model.predict(X)
    axes[0].plot(data.index, trend, 'r--', label='Trend')
    axes[0].set_title('Time Series with Trend')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel(target_col)
    axes[0].legend()
    
    # Rolling statistics
    rolling_mean = data[target_col].rolling(window=window).mean()
    rolling_std = data[target_col].rolling(window=window).std()
    
    axes[1].plot(data.index, rolling_mean, label='Rolling Mean')
    axes[1].fill_between(
        data.index,
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        alpha=0.2,
        label='±1 Std Dev'
    )
    axes[1].set_title(f'Rolling Statistics (Window: {window} hours)')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel(target_col)
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_anomalies(data: pd.DataFrame,
                   target_col: str = 'MW',
                   method: str = 'zscore',
                   threshold: float = 3.0,
                   figsize: Tuple[int, int] = (15, 6),
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot time series data with anomalies highlighted.
    
    Args:
        data: Input DataFrame
        target_col: Target column name
        method: Anomaly detection method
        threshold: Detection threshold
        figsize: Figure size
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot original data
    data[target_col].plot(ax=ax, alpha=0.5, label='Original')
    
    # Detect anomalies
    series = data[target_col]
    if method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        anomalies = z_scores > threshold
    elif method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        anomalies = (series < (Q1 - threshold * IQR)) | (series > (Q3 + threshold * IQR))
    
    # Plot anomalies
    ax.scatter(data.index[anomalies], series[anomalies], color='red', label='Anomalies')
    
    ax.set_title(f'Time Series with Anomalies ({method.upper()})')
    ax.set_xlabel('Date')
    ax.set_ylabel(target_col)
    ax.legend()
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_correlation(data: pd.DataFrame,
                    target_col: str = 'MW',
                    lags: int = 48,
                    figsize: Tuple[int, int] = (15, 5),
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot autocorrelation and partial autocorrelation.
    
    Args:
        data: Input DataFrame
        target_col: Target column name
        lags: Number of lags
        figsize: Figure size
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # type: ignore
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot ACF
    plot_acf(data[target_col].dropna(), lags=lags, ax=axes[0])
    axes[0].set_title('Autocorrelation')
    
    # Plot PACF
    plot_pacf(data[target_col].dropna(), lags=lags, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation')
    
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_report_plots(data: pd.DataFrame,
                       target_col: str = 'MW',
                       output_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
    """
    Create all plots for reporting.
    
    Args:
        data: Input DataFrame
        target_col: Target column name
        output_dir: Directory to save plots
        
    Returns:
        Dictionary of plot figures
    """

    if output_dir:
        output_dir = Path(output_dir) # type: ignore
        output_dir.mkdir(parents=True, exist_ok=True) # type: ignore
    else:
        output_dir = Path(os.getcwd(), 'plots').resolve() # type: ignore        
        output_dir.mkdir(parents=True, exist_ok=True) # type: ignore
    
    plots = {}
    
    # Time series plot
    plots['time_series'] = plot_time_series(
        data,
        target_col=target_col,
        save_path=str(output_dir / 'time_series.png') # type: ignore
    )
    
    # Seasonal patterns
    plots['seasonal'] = plot_seasonal_patterns(
        data,
        target_col=target_col,
        save_path=str(output_dir / 'seasonal_patterns.png') # type: ignore
    )
    
    # Distribution
    plots['distribution'] = plot_distribution(
        data,
        target_col=target_col,
        save_path=str(output_dir / 'distribution.png') # type: ignore
    )
    
    # Trends
    plots['trends'] = plot_trends(
        data,
        target_col=target_col,
        save_path=str(output_dir / 'trends.png') # type: ignore
    )
    
    # Anomalies
    plots['anomalies'] = plot_anomalies(
        data,
        target_col=target_col,
        save_path=str(output_dir / 'anomalies.png') # type: ignore
    )
    
    # Correlation
    plots['correlation'] = plot_correlation(
        data,
        target_col=target_col,
        save_path=str(output_dir / 'correlation.png') # type: ignore
    )
    
    return plots 