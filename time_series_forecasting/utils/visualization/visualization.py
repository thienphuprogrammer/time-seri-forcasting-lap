from typing import Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def plot_metrics(metrics: Dict[str, float],
                 figsize: tuple = (15, 5)) -> None:
    """
    Plot metrics.
    """
    plt.figure(figsize=figsize)
    plt.bar(metrics.keys(), metrics.values())
    plt.show()

def plot_data_distribution(data: pd.DataFrame,
                           columns: List[str],
                           figsize: tuple = (15, 5)) -> None:
    """
    Plot data distribution for each column.
    """
    plt.figure(figsize=figsize)
    for col in columns:
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

def plot_correlation_matrix(data: pd.DataFrame,
                           figsize: tuple = (15, 5)) -> None:
    """
    Plot correlation matrix.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.show()


def plot_time_series(timestamps: np.ndarray,
                    values: np.ndarray,
                    title: str = 'Time Series Plot',
                    xlabel: str = 'Time',
                    ylabel: str = 'Value',
                    figsize: tuple = (15, 5)) -> None:
    """
    Plot time series data.
    
    Args:
        timestamps: Array of timestamps
        values: Array of values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(timestamps, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_training_history(history: Dict[str, List[float]],
                         metrics: List[str] = ['loss', 'mae'],
                         figsize: tuple = (15, 5),
                         save_path: Optional[str] = None,
                         show: bool = True) -> None:
    """
    Plot training history for deep learning models.
    
    Args:
        history: Training history dictionary
        metrics: List of metrics to plot
        figsize: Figure size
        save_path: Optional file path to save the generated plot
        show: Whether to display the plot. If False, the figure will be closed after saving.
    """
    # Handle missing or None history gracefully
    if history is None or len(history) == 0:
        if show:
            print("[plot_training_history] No history data to plot.")
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        # Skip metrics that are not present in history
        if metric not in history and f'val_{metric}' not in history:
            ax.axis('off')  # Hide unused subplot
            continue

        if metric in history:
            ax.plot(history[metric], label='Training')
        if f'val_{metric}' in history:
            ax.plot(history[f'val_{metric}'], label='Validation')

        ax.set_title(f'{metric.upper()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    # Save figure if requested
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_predictions(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    title: str = 'Predictions vs Actual',
                    max_points: int = 1000,
                    figsize: tuple = (15, 10)) -> None:
    """
    Plot model predictions against actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        max_points: Maximum number of points to plot
        figsize: Figure size
    """
    # Limit points for plotting
    if len(y_true) > max_points:
        indices = np.linspace(0, len(y_true)-1, max_points, dtype=int)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Time series plot
    ax1.plot(y_true, label='Actual', alpha=0.7)
    ax1.plot(y_pred, label='Predicted', alpha=0.7)
    ax1.set_title(f'{title} - Time Series')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True)
    
    # Scatter plot
    ax2.scatter(y_true, y_pred, alpha=0.5)
    ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax2.set_title(f'{title} - Scatter Plot')
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Predicted Values')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_model_comparison(model_metrics: Dict[str, Dict[str, float]],
                         metrics: List[str] = ['RMSE', 'MAE', 'R2', 'MAPE'],
                         figsize: tuple = (15, 10)) -> None:
    """
    Create comparison plots for multiple models.
    
    Args:
        model_metrics: Dictionary of model metrics
        metrics: List of metrics to plot
        figsize: Figure size
    """
    # Create subplots
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        metric_values = [m[metric] for m in model_metrics.values()]
        model_names = list(model_metrics.keys())
        
        axes[i].bar(model_names, metric_values)
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True)
    
    # Remove empty subplots
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

def plot_forecast_errors(forecast_errors: Dict[str, Dict[str, float]],
                        figsize: tuple = (15, 5)) -> None:
    """
    Plot forecast errors for different horizons.
    
    Args:
        forecast_errors: Dictionary of forecast errors
        figsize: Figure size
    """
    horizons = list(forecast_errors.keys())
    metrics = list(forecast_errors[horizons[0]].keys())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        values = [forecast_errors[h][metric] for h in horizons]
        ax.plot(range(1, len(horizons) + 1), values, marker='o')
        ax.set_title(f'{metric} by Horizon')
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel(metric)
        ax.grid(True)
    
    plt.tight_layout()
    plt.show() 