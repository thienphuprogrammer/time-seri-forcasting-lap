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
        config = config or {}
        # CUDA-optimized defaults to reduce register spilling
        self.num_heads = config.get('num_heads', 8)
        self.d_model = config.get('d_model', 256)  # Balanced for GPU registers
        self.num_layers = config.get('num_layers', 4)  # Reduced layers for register efficiency
        self.dff = config.get('dff', 512)  # Reduced for memory efficiency
        self.dropout = config.get('dropout', 0.2)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)  # Reduced for register efficiency
        self.optimizer = config.get('optimizer', 'adam')  # Standard optimizer for CUDA efficiency
        
        # CUDA optimization flags
        self.mixed_precision = config.get('mixed_precision', True)
        self.gradient_checkpointing = config.get('gradient_checkpointing', False)
    
    def build(self, input_shape: Optional[tuple] = None) -> None:
        """
        Build transformer model architecture.
        
        Args:
            input_shape: Shape of input data
        """
        if input_shape is None:
            raise ValueError("input_shape is required for Transformer models")
        
        # Enable GPU acceleration - remove CPU constraint
        # Input layers
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Project input to d_model dimension
        x = tf.keras.layers.Dense(self.d_model)(inputs)
        
        # Create positional encoding for d_model dimension
        max_length, _ = input_shape
        pos_encoding = self._positional_encoding(max_length, self.d_model)
        
        # Add positional encoding
        x = x + pos_encoding
        
        # Apply input dropout
        x = tf.keras.layers.Dropout(self.dropout)(x)
        
        # Transformer blocks
        for _ in range(self.num_layers):
            x = self._transformer_block(x)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Add final dense layers for better representation
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Use standard optimizer for CUDA efficiency
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
    
    def _configure_gpu_optimization(self):
        """Configure GPU optimizations to reduce register spilling."""
        # Enable mixed precision if requested
        if self.mixed_precision:
            try:
                # Mixed precision reduces register usage
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("✅ Mixed precision enabled for Transformer (reduces register usage)")
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
        
        epochs = kwargs.get('epochs', 100)  # Reduced epochs for CUDA efficiency
        patience = kwargs.get('patience', 15)  # Balanced patience
        
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss' if validation_data is not None else 'loss',
            patience=patience,
            restore_best_weights=True
        )
        
        # Learning rate scheduling for transformers
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if validation_data is not None else 'loss',
            factor=0.5,
            patience=patience//3,
            min_lr=1e-7,
            verbose=1
        )
        
        # Optional: Warmup learning rate schedule
        warmup_steps = kwargs.get('warmup_steps', 1000)
        if warmup_steps > 0:
            lr_warmup = tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: self.learning_rate * min(1.0, epoch / warmup_steps)
            )
            callbacks = [early_stopping, lr_scheduler, lr_warmup]
        else:
            callbacks = [early_stopping, lr_scheduler]
        
        # Train model
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
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
        
        y_pred = self.predict(X)
        
        # Ensure y_pred is also 2D for metrics calculation
        if y_pred.ndim > 2:
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        
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
        # Multi-head attention optimized for CUDA
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,  # Balanced key dimension
            dropout=self.dropout,
            use_bias=True,  # CUDA optimization
            kernel_initializer='glorot_uniform'
        )(x, x)
        attn = tf.keras.layers.Dropout(self.dropout)(attn)
        out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
        
        # Memory-efficient feed-forward network
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dff, 
                                activation='relu',
                                kernel_initializer='glorot_uniform',
                                use_bias=True),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Dense(self.d_model,
                                kernel_initializer='glorot_uniform',
                                use_bias=True)
        ])
        
        ffn_out = ffn(out1)
        ffn_out = tf.keras.layers.Dropout(self.dropout)(ffn_out)
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_out) 