"""
Traditional Machine Learning Models for Time Series Forecasting
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

from time_series_forecasting.models.base.base_model import BaseTimeSeriesModel

class LinearRegressionModel(BaseTimeSeriesModel):
    """
    Linear Regression model for time series forecasting.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Linear Regression model.
        
        Args:
            name: Model name
            config: Model configuration
        """
        super().__init__(name, config)
        self.model = LinearRegression(**config if config else {})
    
    def build(self, input_shape: Optional[tuple] = None) -> None:
        """
        Build model (not needed for sklearn models).
        """
        pass

    def fit(self,
            train_data: Union[np.ndarray, Any],
            validation_data: Optional[Union[np.ndarray, Any]] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_data: Training data (X, y)
            validation_data: Validation data (not used)
            **kwargs: Additional training arguments
            
        Returns:
            Training history
        """
        X_train, y_train = self._prepare_data(train_data)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Create minimal history
        history = {
            'loss': [self.model.score(X_train, y_train)],
            'val_loss': [] if validation_data is None else [self.model.score(*self._prepare_data(validation_data))]
        }
        self.history = [history]
        
        return history
    
    def predict(self,
                data: Union[np.ndarray, Any],
                **kwargs) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            data: Input data
            **kwargs: Additional prediction arguments
            
        Returns:
            Model predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = self._prepare_data(data)[0]
        return self.model.predict(X)
    
    def evaluate(self,
                 data: Union[np.ndarray, Any],
                 metrics: Optional[list] = None,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            data: Test data
            metrics: List of metrics to evaluate
            **kwargs: Additional evaluation arguments
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        X, y = self._prepare_data(data)
        y_pred = self.predict(X)
        
        from time_series_forecasting.metrics.evaluation_metrics import calculate_metrics
        return calculate_metrics(y, y_pred)
    
    def _prepare_data(self, data: Union[np.ndarray, Any]) -> tuple:
        """
        Prepare data for sklearn model.
        
        Args:
            data: Input data
            
        Returns:
            Tuple of (X, y)
        """
        if isinstance(data, (tuple, list)):
            return data[0], data[1]
        elif hasattr(data, 'numpy'):
            # TensorFlow dataset
            X, y = [], []
            for batch in data:
                batch_x, batch_y = batch
                X.append(batch_x.numpy())
                y.append(batch_y.numpy())
            return np.vstack(X), np.vstack(y)
        elif isinstance(data, pd.DataFrame):
            # pandas DataFrame - create lagged features for X and use target as y
            # Use last column as target, create features from all columns
            target_col = data.columns[-1]
            
            # Create simple lagged features for linear regression
            X_data = []
            y_data = []
            
            # Use simple windowing approach
            input_width = 24  # Use last 24 hours as features
            
            for i in range(input_width, len(data)):
                # Get features (lagged values)
                features = data.iloc[i-input_width:i].values.flatten()
                target = data.iloc[i][target_col]
                
                X_data.append(features)
                y_data.append(target)
            
            return np.array(X_data), np.array(y_data)
        elif isinstance(data, np.ndarray):
            # Already numpy array - assume it's X data for prediction
            if data.ndim == 1:
                return data.reshape(1, -1), None
            return data, None
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")

class ARIMAModel(BaseTimeSeriesModel):
    """
    ARIMA model for time series forecasting.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ARIMA model.
        
        Args:
            name: Model name
            config: Model configuration with ARIMA order (p,d,q)
        """
        super().__init__(name, config)
        self.order = config.get('order', (1,1,1)) if config else (1,1,1)
    
    def build(self, input_shape: Optional[tuple] = None) -> None:
        """
        Build model (not needed for ARIMA).
        """
        pass
    
    def fit(self,
            train_data: Union[np.ndarray, Any],
            validation_data: Optional[Union[np.ndarray, Any]] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_data: Training data
            validation_data: Validation data (not used)
            **kwargs: Additional training arguments
            
        Returns:
            Training history
        """
        y_train = self._prepare_data(train_data)
        self.model = ARIMA(y_train, order=self.order).fit()
        self.is_fitted = True
        
        # Create minimal history
        history = {
            'aic': [self.model.aic],
            'bic': [self.model.bic]
        }
        self.history = [history]
        
        return history
    
    def predict(self,
                data: Union[np.ndarray, Any],
                **kwargs) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            data: Input data or number of steps to forecast
            **kwargs: Additional prediction arguments
            
        Returns:
            Model predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        steps = kwargs.get('steps', 1)
        return self.model.forecast(steps)
    
    def evaluate(self,
                 data: Union[np.ndarray, Any],
                 metrics: Optional[list] = None,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            data: Test data
            metrics: List of metrics to evaluate
            **kwargs: Additional evaluation arguments
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        y_true = self._prepare_data(data)
        y_pred = self.predict(data, steps=len(y_true))
        
        from time_series_forecasting.metrics.evaluation_metrics import calculate_metrics
        return calculate_metrics(y_true, y_pred)
    
    def _prepare_data(self, data: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Prepare data for ARIMA model.
        
        Args:
            data: Input data
            
        Returns:
            1D array of time series values
        """
        if isinstance(data, (tuple, list)):
            return data[1].ravel()
        elif isinstance(data, np.ndarray):
            return data.ravel()
        elif isinstance(data, pd.DataFrame):
            # pandas DataFrame - use target column values
            target_col = data.columns[-1]
            return data[target_col].values
        elif hasattr(data, 'numpy'):
            # TensorFlow dataset
            y = []
            for batch in data:
                _, batch_y = batch
                y.append(batch_y.numpy())
            return np.concatenate(y).ravel()
        else:
            raise ValueError("Unsupported data format") 