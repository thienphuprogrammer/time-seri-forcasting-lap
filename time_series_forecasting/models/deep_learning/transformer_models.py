"""
Transformer Models for Time Series Forecasting
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, Union
from time_series_forecasting.models.base.base_model import BaseTimeSeriesModel

class TransformerModel(BaseTimeSeriesModel):
    """
    Transformer model for time series forecasting.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize transformer model.
        
        Args:
            name: Model name
            config: Model configuration
        """
        super().__init__(name, config)
        self.num_heads = config.get('num_heads', 8)
        self.d_model = config.get('d_model', 128)
        self.num_layers = config.get('num_layers', 4)
        self.dff = config.get('dff', 512)
        self.dropout = config.get('dropout', 0.1)
        self.learning_rate = config.get('learning_rate', 0.001)
    
    def build(self, input_shape: Optional[tuple] = None) -> None:
        """
        Build transformer model architecture.
        
        Args:
            input_shape: Shape of input data
        """
        if input_shape is None:
            raise ValueError("input_shape is required for Transformer models")
            
        # Input layers
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Project input to d_model dimension
        x = tf.keras.layers.Dense(self.d_model)(inputs)
        
        # Create positional encoding for d_model dimension
        max_length, _ = input_shape
        pos_encoding = self._positional_encoding(max_length, self.d_model)
        
        # Add positional encoding
        x = x + pos_encoding
        
        # Transformer blocks
        for _ in range(self.num_layers):
            x = self._transformer_block(x)
        
        # Global average pooling and output
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        # Create and compile model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
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
    
    def _positional_encoding(self, length: int, depth: int) -> tf.Tensor:
        """
        Create positional encoding.
        
        Args:
            length: Sequence length
            depth: Feature dimension
            
        Returns:
            Positional encoding tensor
        """
        # Create position indices
        positions = np.arange(length)[:, np.newaxis]
        
        # Create depth indices (only use even indices for sin/cos pairs)
        depths = np.arange(0, depth, 2)[np.newaxis, :] / depth
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        
        # Create positional encoding with interleaved sin/cos
        pos_encoding = np.zeros((length, depth))
        pos_encoding[:, 0::2] = np.sin(angle_rads)  # Even indices get sin
        pos_encoding[:, 1::2] = np.cos(angle_rads)  # Odd indices get cos
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def _transformer_block(self, x: tf.Tensor) -> tf.Tensor:
        """
        Create a transformer block.
        
        Args:
            x: Input tensor
            
        Returns:
            Transformed tensor
        """
        # Multi-head attention
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model
        )(x, x)
        attn = tf.keras.layers.Dropout(self.dropout)(attn)
        out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
        
        # Feed-forward network
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dff, activation='relu'),
            tf.keras.layers.Dense(self.d_model)
        ])
        
        ffn_out = ffn(out1)
        ffn_out = tf.keras.layers.Dropout(self.dropout)(ffn_out)
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_out) 