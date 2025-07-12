import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt

class WindowGenerator:
    """
    Class for generating time series windows for forecasting.
    Supports both single-step and multi-step forecasting.
    """
    
    def __init__(self, 
                 input_width: int, 
                 label_width: int, 
                 shift: int,
                 train_split: float = 0.7,
                 val_split: float = 0.15,
                 test_split: float = 0.15,
                 label_columns: list = None):
        """
        Initialize WindowGenerator.
        
        Args:
            input_width: Number of time steps to use as input
            label_width: Number of time steps to predict
            shift: Number of time steps to shift the label window
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            label_columns: Columns to use as labels (if None, uses all columns)
        """
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.label_columns = label_columns
        
        # Total window size
        self.total_window_size = input_width + shift
        
        # Input and label slices
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
        # Store data and splits
        self.data = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Input DataFrame with datetime index
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        self.data = data
        n = len(data)
        
        # Calculate split indices
        train_end = int(n * self.train_split)
        val_end = int(n * (self.train_split + self.val_split))
        
        # Split data
        self.train_df = data.iloc[:train_end]
        self.val_df = data.iloc[train_end:val_end]
        self.test_df = data.iloc[val_end:]
        
        print(f"Data splits - Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")
        
        return self.train_df, self.val_df, self.test_df
    
    def make_dataset(self, data: pd.DataFrame, shuffle: bool = True, batch_size: int = 32) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from DataFrame.
        
        Args:
            data: Input DataFrame
            shuffle: Whether to shuffle the dataset
            batch_size: Batch size for the dataset
            
        Returns:
            TensorFlow dataset
        """
        if len(data) < self.total_window_size:
            raise ValueError(f"Data length ({len(data)}) must be >= total_window_size ({self.total_window_size})")
        
        # Convert to numpy array
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data.values,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=batch_size
        )
        
        # Split into inputs and labels
        ds = ds.map(self.split_window)
        
        return ds
    
    def split_window(self, features: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Split window into inputs and labels.
        
        Args:
            features: Input tensor with shape (batch_size, total_window_size, features)
            
        Returns:
            Tuple of (inputs, labels)
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        
        # If label_columns specified, only use those columns for labels
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.data.columns.get_loc(name)] for name in self.label_columns],
                axis=-1
            )
        
        # If label_columns specified, only use those columns for inputs
        if self.label_columns is not None:
            inputs = tf.stack(
                [inputs[:, :, self.data.columns.get_loc(name)] for name in self.label_columns],
                axis=-1
            )
        
        return inputs, labels
    
    def get_train_dataset(self, batch_size: int = 32) -> tf.data.Dataset:
        """Get training dataset."""
        if self.train_df is None:
            raise ValueError("Data must be split first using split_data()")
        return self.make_dataset(self.train_df, shuffle=True, batch_size=batch_size)
    
    def get_val_dataset(self, batch_size: int = 32) -> tf.data.Dataset:
        """Get validation dataset."""
        if self.val_df is None:
            raise ValueError("Data must be split first using split_data()")
        return self.make_dataset(self.val_df, shuffle=False, batch_size=batch_size)
    
    def get_test_dataset(self, batch_size: int = 32) -> tf.data.Dataset:
        """Get test dataset."""
        if self.test_df is None:
            raise ValueError("Data must be split first using split_data()")
        return self.make_dataset(self.test_df, shuffle=False, batch_size=batch_size)
    
    def plot(self, model: tf.keras.Model = None, plot_col: str = 'target', max_subplots: int = 3):
        """
        Plot the model's predictions.
        
        Args:
            model: Trained model to use for predictions
            plot_col: Column to plot
            max_subplots: Maximum number of subplots to show
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        inputs, labels = self.get_test_dataset().take(1).get_single_element()
        
        plt.figure(figsize=(12, 8))
        plot_col_index = self.data.columns.get_loc(plot_col)
        max_n = min(max_subplots, len(inputs))
        
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)
            
            if self.label_columns:
                label_col_index = self.label_columns.index(plot_col)
            else:
                label_col_index = plot_col_index
            
            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                       edgecolors='k', label='Labels', c='#2ca02c', s=64)
            
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                           marker='X', edgecolors='k', label='Predictions',
                           c='#ff7f0e', s=64)
            
            if n == 0:
                plt.legend()
        
        plt.xlabel('Time [h]')
        plt.tight_layout()
        plt.show()
    
    def get_example_batch(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get a single batch for example purposes.
        
        Returns:
            Tuple of (inputs, labels)
        """
        if self.train_df is None:
            raise ValueError("Data must be split first using split_data()")
        
        for inputs, labels in self.get_train_dataset().take(1):
            return inputs, labels
    
    def get_window_info(self) -> Dict[str, Any]:
        """
        Get information about the window configuration.
        
        Returns:
            Dictionary with window information
        """
        return {
            'input_width': self.input_width,
            'label_width': self.label_width,
            'shift': self.shift,
            'total_window_size': self.total_window_size,
            'input_indices': self.input_indices.tolist(),
            'label_indices': self.label_indices.tolist(),
            'label_columns': self.label_columns
        }
    
    @property
    def example(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Property to get example batch."""
        return self.get_example_batch()
    
    @property
    def train(self) -> tf.data.Dataset:
        """Property to get training dataset."""
        return self.get_train_dataset()
    
    @property
    def val(self) -> tf.data.Dataset:
        """Property to get validation dataset."""
        return self.get_val_dataset()
    
    @property
    def test(self) -> tf.data.Dataset:
        """Property to get test dataset."""
        return self.get_test_dataset() 