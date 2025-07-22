from typing import Dict, Any
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for time series predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    # Ensure 2D arrays
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # Calculate basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }

def compare_models(model_metrics: Dict[str, Dict[str, float]], 
                  metric: str = 'RMSE') -> Dict[str, Any]:
    """
    Compare models based on a specific metric.
    
    Args:
        model_metrics: Dictionary of model metrics
        metric: Metric to compare (default: RMSE)
        
    Returns:
        Dictionary with best model and its score
    """
    if not model_metrics:
        return {'model': None, 'score': None}
    
    best_model = min(
        model_metrics.items(),
        key=lambda x: x[1].get(metric, float('inf'))
    )
    
    return {
        'model': best_model[0],
        'score': best_model[1].get(metric)
    }

def calculate_forecast_error(y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           horizon: int) -> Dict[str, np.ndarray]:
    """
    Calculate forecast error metrics for different horizons.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        horizon: Forecast horizon
        
    Returns:
        Dictionary of error metrics per horizon
    """
    errors = {}
    
    # Calculate errors for each horizon
    for h in range(1, horizon + 1):
        y_true_h = y_true[h-1::horizon]
        y_pred_h = y_pred[h-1::horizon]
        
        # Calculate metrics
        mae_h = mean_absolute_error(y_true_h, y_pred_h)
        mse_h = mean_squared_error(y_true_h, y_pred_h)
        rmse_h = np.sqrt(mse_h)
        
        errors[f'horizon_{h}'] = {
            'MAE': mae_h,
            'MSE': mse_h,
            'RMSE': rmse_h
        }
    
    return errors 