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
        config = config or {}
        
        # CUDA-optimized defaults to reduce register spilling
        self.units = config.get('units', 256)  # Balanced for GPU registers
        self.layers = config.get('layers', 2)  # 2 layers optimal for memory
        self.dropout = config.get('dropout', 0.2)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 64)  # Reduced for register efficiency
        self.optimizer = config.get('optimizer', 'adam')
        
        # CUDA optimization flags
        self.mixed_precision = config.get('mixed_precision', True)
        self.xla_compile = config.get('xla_compile', False)  # Disable to reduce register pressure
        
    def build(self, input_shape: Optional[tuple] = None) -> None:
        """
        Build model architecture.
        
        Args:
            input_shape: Shape of input data
        """
        raise NotImplementedError("Subclasses must implement build()")
    
    def _configure_gpu_optimization(self):
        """Configure GPU optimizations to reduce register spilling."""
        # Enable mixed precision if requested
        if self.mixed_precision:
            try:
                # Mixed precision reduces register usage
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("✅ Mixed precision enabled (reduces register usage)")
            except Exception as e:
                print(f"❌ Mixed precision failed: {e}")
    
    def fit(self,
            train_data: Union[np.ndarray, tf.data.Dataset],
            validation_data: Optional[Union[np.ndarray, tf.data.Dataset]] = None,
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
        
        # Apply GPU optimizations
        self._configure_gpu_optimization()
        
        epochs = kwargs.get('epochs', 100)
        patience = kwargs.get('patience', 15)
        
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss' if validation_data is not None else 'loss',
            patience=patience,
            restore_best_weights=True
        )
        
        # Learning rate reduction with more conservative settings
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if validation_data is not None else 'loss',
            factor=0.7,  # More conservative reduction
            patience=patience//3,
            min_lr=1e-6,  # Higher minimum LR
            verbose=1
        )
        
        # Train model with CUDA optimizations
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=[early_stopping, lr_scheduler],
            verbose=kwargs.get('verbose', 1)
        )
        
        self.is_fitted = True
        self.history = history.history
        return history.history
    
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
        
        if isinstance(data, tf.data.Dataset):
            X, y = [], []
            for batch_x, batch_y in data:
                X.append(batch_x.numpy())
                y.append(batch_y.numpy())
            X = np.vstack(X)
            y_array = np.vstack(y)
            
            # Flatten y to 2D for metrics calculation (samples, features)
            if y_array.ndim > 2:
                y_array = y_array.reshape(-1, y_array.shape[-1])
            y = y_array
        else:
            X, y = data
        
        y_pred_raw = self.predict(X)
        
        # Ensure y_pred is also 2D for metrics calculation
        if y_pred_raw.ndim > 2:
            y_pred = y_pred_raw.reshape(-1, y_pred_raw.shape[-1])
        else:
            y_pred = y_pred_raw
        
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
    
    def build(self, input_shape: Optional[tuple] = None) -> None:
        """
        Build RNN model architecture.
        
        Args:
            input_shape: Shape of input data
        """
        # Build CUDA-optimized RNN model
        model = tf.keras.Sequential()
        
        # Add Input layer first (recommended by Keras)
        if input_shape is not None:
            model.add(tf.keras.layers.Input(shape=input_shape))
        
        # Add RNN layers with CUDA optimizations
        for i in range(self.layers):
            return_sequences = i < self.layers - 1
            model.add(tf.keras.layers.SimpleRNN(
                units=self.units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                recurrent_dropout=0.1,  # Light recurrent dropout
                # CUDA optimization: use CuDNN when possible
                use_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal'
            ))
            
            # Light batch normalization only between layers
            if i < self.layers - 1:
                model.add(tf.keras.layers.BatchNormalization())
        
        # Add output layer
        model.add(tf.keras.layers.Dense(1))
        
        # Use standard optimizer for CUDA efficiency
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model

class GRUModel(BaseRNNModel):
    """
    GRU model for time series forecasting.
    """
    
    def build(self, input_shape: Optional[tuple] = None) -> None:
        """
        Build GRU model architecture.
        
        Args:
            input_shape: Shape of input data
        """
        # Build CUDA-optimized GRU model
        model = tf.keras.Sequential()
        
        # Add Input layer first (recommended by Keras)
        if input_shape is not None:
            model.add(tf.keras.layers.Input(shape=input_shape))
        
        # Add GRU layers with CUDA optimizations
        for i in range(self.layers):
            return_sequences = i < self.layers - 1
            model.add(tf.keras.layers.GRU(
                units=self.units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                recurrent_dropout=0.1,
                reset_after=True,  # GPU-optimized GRU variant
                # CUDA optimization settings
                use_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal'
            ))
            
            # Light batch normalization
            if i < self.layers - 1:
                model.add(tf.keras.layers.BatchNormalization())
        
        # Add output layer
        model.add(tf.keras.layers.Dense(1))
        
        # Standard optimizer for efficiency
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model

class LSTMModel(BaseRNNModel):
    """
    LSTM model for time series forecasting.
    """
    
    def build(self, input_shape: Optional[tuple] = None) -> None:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data
        """
        # Build CUDA-optimized LSTM model
        model = tf.keras.Sequential()
        
        # Add Input layer first (recommended by Keras)
        if input_shape is not None:
            model.add(tf.keras.layers.Input(shape=input_shape))
        
        # Add LSTM layers with CUDA optimizations
        for i in range(self.layers):
            return_sequences = i < self.layers - 1
            model.add(tf.keras.layers.LSTM(
                units=self.units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                recurrent_dropout=0.1,
                implementation=2,  # GPU-optimized LSTM implementation
                # CUDA optimization settings
                use_bias=True,
                unit_forget_bias=True,  # LSTM-specific optimization
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal'
            ))
            
            # Light batch normalization
            if i < self.layers - 1:
                model.add(tf.keras.layers.BatchNormalization())
        
        # Add output layer  
        model.add(tf.keras.layers.Dense(1))
        
        # Standard optimizer for efficiency
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model 