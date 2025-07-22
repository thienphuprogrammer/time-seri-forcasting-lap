"""
Base Model Module for Time Series Forecasting
"""

from abc import ABC, abstractmethod
import pandas as pd # type: ignore
import numpy as np # type: ignore
from typing import Dict, Any, Optional, Union, Tuple, List
from ...core import DataTransformer, WindowGenerator # type: ignore

class BaseTimeSeriesModel(ABC):
    """
    Abstract base class for all time series forecasting models.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base model.
        
        Args:
            name: Model name
            config: Model configuration
        """
        self.name = name
        self.config = config or {}
        self.is_fitted = False
        self.data_transformer = DataTransformer()
        self.history: List[Dict[str, Any]] = []
        
    @abstractmethod
    def build(self, input_shape: Optional[tuple] = None) -> None:
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def fit(self, 
            train_data: Union[pd.DataFrame, WindowGenerator],
            validation_data: Optional[Union[pd.DataFrame, WindowGenerator]] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_data: Training data
            validation_data: Validation data
            **kwargs: Additional training arguments
            
        Returns:
            Training history
        """
        pass
    
    @abstractmethod
    def predict(self, 
                data: Union[pd.DataFrame, WindowGenerator],
                **kwargs) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            data: Input data
            **kwargs: Additional prediction arguments
            
        Returns:
            Model predictions
        """
        pass
    
    def evaluate(self, 
                 data: Union[pd.DataFrame, WindowGenerator],
                 metrics: Optional[list] = None,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            data: Test data
            metrics: List of metrics to compute
            **kwargs: Additional evaluation arguments
            
        Returns:
            Dictionary of metric scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        # Get predictions
        y_pred = self.predict(data, **kwargs)
        
        # Get actual values
        if isinstance(data, WindowGenerator):
            _, y_true = next(iter(data.test))
            y_true = y_true.numpy()
        else:
            y_true = data.values
        
        # Calculate metrics
        results = {}
        metrics = metrics or ['mae', 'mse', 'rmse', 'mape']
        
        for metric in metrics:
            if metric == 'mae':
                results['mae'] = np.mean(np.abs(y_true - y_pred))
            elif metric == 'mse':
                results['mse'] = np.mean((y_true - y_pred) ** 2)
            elif metric == 'rmse':
                results['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
            elif metric == 'mape':
                mask = y_true != 0
                results['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        return results
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Save path
        """
        import joblib # type: ignore
        
        save_dict = {
            'name': self.name,
            'config': self.config,
            'is_fitted': self.is_fitted,
            'history': self.history,
            'data_transformer': self.data_transformer
        }
        
        joblib.dump(save_dict, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: Load path
        """
        import joblib # type: ignore
        
        load_dict = joblib.load(path)
        
        self.name = load_dict['name']
        self.config = load_dict['config']
        self.is_fitted = load_dict['is_fitted']
        self.history = load_dict['history']
        self.data_transformer = load_dict['data_transformer']
        
        print(f"Model loaded from {path}")
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'name': self.name,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
    
    def set_params(self, **params) -> None:
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.config[key] = value
    
    def __repr__(self) -> str:
        """String representation."""
        status = 'fitted' if self.is_fitted else 'not fitted'
        return f"{self.__class__.__name__}(name='{self.name}', status='{status}')"