"""
RNN-based Deep Learning Models for Time Series Forecasting
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, Union
from time_series_forecasting.models.base.base_model import BaseTimeSeriesModel

class BaseRNNModel(BaseTimeSeriesModel):
    """
    Base class for RNN-based models.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base RNN model.
        
        Args:
            name: Model name
            config: Model configuration
        """
        super().__init__(name, config)
        self.units = config.get('units', 64)
        self.layers = config.get('layers', 2)
        self.dropout = config.get('dropout', 0.2)
        self.learning_rate = config.get('learning_rate', 0.001)
    
    def build(self, input_shape: tuple) -> None:
        """
        Build model architecture.
        
        Args:
            input_shape: Shape of input data
        """
        raise NotImplementedError("Subclasses must implement build()")
    
    def fit(self,
            train_data: Union[np.ndarray, tf.data.Dataset],
            val_data: Optional[Union[np.ndarray, tf.data.Dataset]] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            **kwargs: Additional training arguments
            
        Returns:
            Training history
        """
        if not isinstance(self.model, tf.keras.Model):
            raise ValueError("Model must be built before training")
        
        epochs = kwargs.get('epochs', 100)
        patience = kwargs.get('patience', 10)
        
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss' if val_data is not None else 'loss',
            patience=patience,
            restore_best_weights=True
        )
        
        # Train model
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=[early_stopping],
            verbose=kwargs.get('verbose', 1)
        )
        
        self.is_fitted = True
        self.history = history.history
        return self.history
    
    def predict(self,
                data: Union[np.ndarray, tf.data.Dataset],
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
        
        return self.model.predict(data, **kwargs)
    
    def evaluate(self,
                 data: Union[np.ndarray, tf.data.Dataset],
                 metrics: Optional[list] = None) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            data: Test data
            metrics: List of metrics to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        if isinstance(data, tf.data.Dataset):
            X, y = [], []
            for batch_x, batch_y in data:
                X.append(batch_x.numpy())
                y.append(batch_y.numpy())
            X = np.vstack(X)
            y = np.vstack(y)
        else:
            X, y = data
        
        y_pred = self.predict(X)
        
        from time_series_forecasting.metrics.evaluation_metrics import calculate_metrics
        return calculate_metrics(y, y_pred)
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        self.model = tf.keras.models.load_model(path)
        self.is_fitted = True

class RNNModel(BaseRNNModel):
    """
    Simple RNN model for time series forecasting.
    """
    
    def build(self, input_shape: tuple) -> None:
        """
        Build RNN model architecture.
        
        Args:
            input_shape: Shape of input data
        """
        model = tf.keras.Sequential()
        
        # Add RNN layers
        for i in range(self.layers):
            return_sequences = i < self.layers - 1
            model.add(tf.keras.layers.SimpleRNN(
                units=self.units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                input_shape=input_shape if i == 0 else None
            ))
        
        # Add output layer
        model.add(tf.keras.layers.Dense(1))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model

class GRUModel(BaseRNNModel):
    """
    GRU model for time series forecasting.
    """
    
    def build(self, input_shape: tuple) -> None:
        """
        Build GRU model architecture.
        
        Args:
            input_shape: Shape of input data
        """
        model = tf.keras.Sequential()
        
        # Add GRU layers
        for i in range(self.layers):
            return_sequences = i < self.layers - 1
            model.add(tf.keras.layers.GRU(
                units=self.units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                input_shape=input_shape if i == 0 else None
            ))
        
        # Add output layer
        model.add(tf.keras.layers.Dense(1))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model

class LSTMModel(BaseRNNModel):
    """
    LSTM model for time series forecasting.
    """
    
    def build(self, input_shape: tuple) -> None:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data
        """
        model = tf.keras.Sequential()
        
        # Add LSTM layers
        for i in range(self.layers):
            return_sequences = i < self.layers - 1
            model.add(tf.keras.layers.LSTM(
                units=self.units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                input_shape=input_shape if i == 0 else None
            ))
        
        # Add output layer
        model.add(tf.keras.layers.Dense(1))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model 