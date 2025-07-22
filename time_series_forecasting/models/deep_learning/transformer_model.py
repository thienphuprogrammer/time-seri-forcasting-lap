from typing import Dict, Any
import tensorflow as tf
from tensorflow import keras

from .deep_models import DeepLearningBaseModel

class MultiHeadAttention(keras.layers.Layer):
    """Multi-head attention layer."""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.query_dense = keras.layers.Dense(d_model)
        self.key_dense = keras.layers.Dense(d_model)
        self.value_dense = keras.layers.Dense(d_model)
        
        self.dense = keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def call(self, inputs, training=True):
        query, key, value = inputs
        batch_size = tf.shape(query)[0]
        
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # Scaled dot-product attention
        scaled_attention = tf.matmul(query, key, transpose_b=True)
        scaled_attention = scaled_attention / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        
        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
        
        output = tf.matmul(attention_weights, value)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        return self.dense(output)

class TransformerBlock(keras.layers.Layer):
    """Transformer block with multi-head attention and feed forward network."""
    
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = keras.Sequential([
            keras.layers.Dense(dff, activation='relu'),
            keras.layers.Dense(d_model)
        ])
        
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = keras.layers.Dropout(dropout)
        self.dropout2 = keras.layers.Dropout(dropout)
        
    def call(self, inputs, training=True):
        # Multi-head attention
        attn_output = self.mha([inputs, inputs, inputs], training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerModel(DeepLearningBaseModel):
    """Transformer model for time series forecasting."""
    
    def __init__(self,
                 num_heads: int = 8,
                 d_model: int = 128,
                 num_layers: int = 4,
                 dff: int = 512,
                 dropout: float = 0.1):
        super().__init__('transformer')
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_layers = num_layers
        self.dff = dff
        self.dropout = dropout
    
    def build(self, input_shape: tuple) -> None:
        """Build transformer model."""
        inputs = keras.Input(shape=input_shape)
        x = inputs
        
        # Positional encoding could be added here
        
        # Add transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout=self.dropout
            )(x)
        
        # Global average pooling and output
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(1)(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.compile_model() 