import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras # type: ignore
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class ModelFactory:
    """
    Factory class for creating different types of forecasting models.
    """
    
    def __init__(self):
        """Initialize ModelFactory."""
        self.models = {}
        
    def create_linear_regression(self, **kwargs) -> LinearRegression:
        """
        Create a Linear Regression model.
        
        Args:
            **kwargs: Additional arguments for LinearRegression
            
        Returns:
            LinearRegression model
        """
        model = LinearRegression(**kwargs)
        self.models['linear_regression'] = model
        return model
    
    def create_arima(self, endog, order: Tuple[int, int, int], **kwargs) -> ARIMA:
        """
        Create an ARIMA model.
        
        Args:
            endog: Endogenous variable (time series data)
            order: (p, d, q) order of the ARIMA model
            **kwargs: Additional arguments for ARIMA
            
        Returns:
            ARIMA model
        """
        model = ARIMA(endog, order=order, **kwargs)
        self.models['arima'] = model
        return model
    
    def create_sarima(self, order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int], **kwargs) -> SARIMAX:
        """
        Create a SARIMA model.
        
        Args:
            order: (p, d, q) order of the SARIMA model
            seasonal_order: (P, D, Q, s) seasonal order
            **kwargs: Additional arguments for SARIMAX
            
        Returns:
            SARIMAX model
        """
        model = SARIMAX(order=order, seasonal_order=seasonal_order, **kwargs)
        self.models['sarima'] = model
        return model
    
    def create_random_forest(self, **kwargs) -> RandomForestRegressor:
        """
        Create a Random Forest model.
        
        Args:
            **kwargs: Additional arguments for RandomForestRegressor
            
        Returns:
            RandomForestRegressor model
        """
        model = RandomForestRegressor(**kwargs)
        self.models['random_forest'] = model
        return model
    
    def create_xgboost(self, **kwargs) -> XGBRegressor:
        """
        Create an XGBoost model.
        
        Args:
            **kwargs: Additional arguments for XGBRegressor
            
        Returns:
            XGBRegressor model
        """
        model = XGBRegressor(**kwargs)
        self.models['xgboost'] = model
        return model
    
    def create_rnn(self, 
                   input_shape: Tuple[int, int], 
                   units: int = 64, 
                   layers: int = 2,
                   dropout: float = 0.2,
                   **kwargs) -> tf.keras.Model: # type: ignore
        """
        Create a Recurrent Neural Network model.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            units: Number of units in RNN layers
            layers: Number of RNN layers
            dropout: Dropout rate
            **kwargs: Additional arguments for model compilation
            
        Returns:
            Compiled RNN model
        """
        model = keras.Sequential()
        
        # Add RNN layers
        for i in range(layers):
            return_sequences = i < layers - 1
            model.add(keras.layers.SimpleRNN(
                units=units,
                return_sequences=return_sequences,
                dropout=dropout,
                input_shape=input_shape if i == 0 else None
            ))
        
        # Add output layer
        model.add(keras.layers.Dense(1))
        
        # Compile model
        model.compile(
            optimizer=kwargs.get('optimizer', 'adam'),
            loss=kwargs.get('loss', 'mse'),
            metrics=kwargs.get('metrics', ['mae'])
        )
        
        self.models['rnn'] = model
        return model
    
    def create_gru(self, 
                   input_shape: Tuple[int, int], 
                   units: int = 64, 
                   layers: int = 2,
                   dropout: float = 0.2,
                   **kwargs) -> tf.keras.Model: # type: ignore
        """
        Create a GRU model.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            units: Number of units in GRU layers
            layers: Number of GRU layers
            dropout: Dropout rate
            **kwargs: Additional arguments for model compilation
            
        Returns:
            Compiled GRU model
        """
        model = keras.Sequential()
        
        # Add GRU layers
        for i in range(layers):
            return_sequences = i < layers - 1
            model.add(keras.layers.GRU(
                units=units,
                return_sequences=return_sequences,
                dropout=dropout,
                input_shape=input_shape if i == 0 else None
            ))
        
        # Add output layer
        model.add(keras.layers.Dense(1))
        
        # Compile model
        model.compile(
            optimizer=kwargs.get('optimizer', 'adam'),
            loss=kwargs.get('loss', 'mse'),
            metrics=kwargs.get('metrics', ['mae'])
        )
        
        self.models['gru'] = model
        return model
    
    def create_lstm(self, 
                    input_shape: Tuple[int, int], 
                    units: int = 64, 
                    layers: int = 2,
                    dropout: float = 0.2,
                    **kwargs) -> tf.keras.Model: # type: ignore
        """
        Create an LSTM model.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            units: Number of units in LSTM layers
            layers: Number of LSTM layers
            dropout: Dropout rate
            **kwargs: Additional arguments for model compilation
            
        Returns:
            Compiled LSTM model
        """
        model = keras.Sequential()
        
        # Add LSTM layers
        for i in range(layers):
            return_sequences = i < layers - 1
            model.add(keras.layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=dropout,
                input_shape=input_shape if i == 0 else None
            ))
        
        # Add output layer
        model.add(keras.layers.Dense(1))
        
        # Compile model
        model.compile(
            optimizer=kwargs.get('optimizer', 'adam'),
            loss=kwargs.get('loss', 'mse'),
            metrics=kwargs.get('metrics', ['mae'])
        )
        
        self.models['lstm'] = model
        return model
    
    def create_cnn_lstm(self, 
                        input_shape: Tuple[int, int], 
                        cnn_filters: int = 64,
                        cnn_kernel_size: int = 3,
                        lstm_units: int = 64,
                        dropout: float = 0.2,
                        **kwargs) -> tf.keras.Model: # type: ignore
        """
        Create a CNN-LSTM hybrid model.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            cnn_filters: Number of CNN filters
            cnn_kernel_size: Size of CNN kernel
            lstm_units: Number of LSTM units
            dropout: Dropout rate
            **kwargs: Additional arguments for model compilation
            
        Returns:
            Compiled CNN-LSTM model
        """
        model = keras.Sequential([
            keras.layers.Conv1D(
                filters=cnn_filters,
                kernel_size=cnn_kernel_size,
                activation='relu',
                input_shape=input_shape
            ),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.LSTM(units=lstm_units, dropout=dropout),
            keras.layers.Dense(1)
        ])
        
        # Compile model
        model.compile(
            optimizer=kwargs.get('optimizer', 'adam'),
            loss=kwargs.get('loss', 'mse'),
            metrics=kwargs.get('metrics', ['mae'])
        )
        
        self.models['cnn_lstm'] = model
        return model
    
    def create_transformer(self, 
                          input_shape: Tuple[int, int],
                          num_heads: int = 8,
                          d_model: int = 128,
                          num_layers: int = 4,
                          dff: int = 512,
                          dropout: float = 0.1,
                          **kwargs) -> tf.keras.Model: # type: ignore
        """
        Create a Transformer model for time series forecasting.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            num_heads: Number of attention heads
            d_model: Dimension of the model
            num_layers: Number of transformer layers
            dff: Dimension of feed-forward network
            dropout: Dropout rate
            **kwargs: Additional arguments for model compilation
            
        Returns:
            Compiled Transformer model
        """
        # Positional encoding
        def positional_encoding(length, depth):
            positions = np.arange(length)[:, np.newaxis]
            depths = np.arange(depth)[np.newaxis, :] / depth
            angle_rates = 1 / (10000**depths)
            angle_rads = positions * angle_rates
            pos_encoding = np.concatenate(
                [np.sin(angle_rads), np.cos(angle_rads)],
                axis=-1)
            return tf.cast(pos_encoding, dtype=tf.float32)
        
        # Transformer block
        class TransformerBlock(keras.layers.Layer):
            def __init__(self, d_model, num_heads, dff, dropout=0.1):
                super().__init__()
                self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
                self.ffn = keras.Sequential([
                    keras.layers.Dense(dff, activation='relu'),
                    keras.layers.Dense(d_model)
                ])
                self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
                self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
                self.dropout1 = keras.layers.Dropout(dropout)
                self.dropout2 = keras.layers.Dropout(dropout)
            
            def call(self, inputs, training):
                attn_output = self.att(inputs, inputs)
                attn_output = self.dropout1(attn_output, training=training)
                out1 = self.layernorm1(inputs + attn_output)
                ffn_output = self.ffn(out1)
                ffn_output = self.dropout2(ffn_output, training=training)
                return self.layernorm2(out1 + ffn_output)
        
        # Build model
        inputs = keras.layers.Input(shape=input_shape)
        
        # Add positional encoding
        pos_encoding = positional_encoding(input_shape[0], input_shape[1])
        x = inputs + pos_encoding
        
        # Add transformer blocks
        for _ in range(num_layers):
            x = TransformerBlock(d_model, num_heads, dff, dropout)(x)
        
        # Global average pooling and output
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=kwargs.get('optimizer', 'adam'),
            loss=kwargs.get('loss', 'mse'),
            metrics=kwargs.get('metrics', ['mae'])
        )
        
        self.models['transformer'] = model
        return model
    
    def create_seq2seq_attention(self, 
                                input_shape: Tuple[int, int],
                                encoder_units: int = 128,
                                decoder_units: int = 128,
                                output_length: int = 1,
                                dropout: float = 0.2,
                                **kwargs) -> tf.keras.Model: # type: ignore
        """
        Create a Seq2Seq model with attention mechanism.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            encoder_units: Number of units in encoder LSTM
            decoder_units: Number of units in decoder LSTM
            output_length: Length of output sequence
            dropout: Dropout rate
            **kwargs: Additional arguments for model compilation
            
        Returns:
            Compiled Seq2Seq model with attention
        """
        # Encoder
        encoder_inputs = keras.layers.Input(shape=input_shape)
        encoder_lstm = keras.layers.LSTM(
            encoder_units, 
            return_sequences=True, 
            return_state=True,
            dropout=dropout
        )
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = keras.layers.Input(shape=(output_length, input_shape[1]))
        decoder_lstm = keras.layers.LSTM(
            decoder_units, 
            return_sequences=True, 
            return_state=True,
            dropout=dropout
        )
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        
        # Attention mechanism
        attention = keras.layers.Attention()
        attention_output = attention([decoder_outputs, encoder_outputs])
        
        # Concatenate attention output with decoder output
        decoder_concat_input = keras.layers.concatenate([decoder_outputs, attention_output])
        
        # Dense layer for final output
        decoder_dense = keras.layers.Dense(1, activation='linear')
        decoder_outputs = decoder_dense(decoder_concat_input)
        
        # Define the model
        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        # Compile model
        model.compile(
            optimizer=kwargs.get('optimizer', 'adam'),
            loss=kwargs.get('loss', 'mse'),
            metrics=kwargs.get('metrics', ['mae'])
        )
        
        self.models['seq2seq_attention'] = model
        return model
    
    def create_ensemble_model(self, models: list, method: str = 'average') -> 'EnsembleModel':
        """
        Create an ensemble model combining multiple models.
        
        Args:
            models: List of trained models
            method: Ensemble method ('average', 'weighted', 'voting')
            
        Returns:
            EnsembleModel instance
        """
        ensemble = EnsembleModel(models, method)
        self.models['ensemble'] = ensemble
        return ensemble
    
    def get_model(self, model_name: str) -> Any:
        """
        Get a model by name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model instance
        """
        return self.models.get(model_name)
    
    def list_models(self) -> list:
        """
        List all created models.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())


class EnsembleModel:
    """
    Ensemble model that combines predictions from multiple models.
    """
    
    def __init__(self, models: list, method: str = 'average'):
        """
        Initialize EnsembleModel.
        
        Args:
            models: List of trained models
            method: Ensemble method ('average', 'weighted', 'voting')
        """
        self.models = models
        self.method = method
        self.weights = None
        
    def fit(self, X, y, weights: Optional[list] = None):
        """
        Fit the ensemble model (for weighted averaging).
        
        Args:
            X: Input features
            y: Target values
            weights: Weights for each model (for weighted averaging)
        """
        if self.method == 'weighted' and weights is not None:
            self.weights = weights
        return self
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        
        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                predictions.append(pred)
            else:
                # For TensorFlow models
                pred = model(X).numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if self.method == 'average':
            return np.mean(predictions, axis=0)
        elif self.method == 'weighted' and self.weights is not None:
            return np.average(predictions, axis=0, weights=self.weights)
        elif self.method == 'voting':
            # For classification, use voting; for regression, use median
            return np.median(predictions, axis=0)
        else:
            return np.mean(predictions, axis=0) 