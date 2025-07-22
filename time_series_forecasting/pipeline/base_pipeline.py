from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

from time_series_forecasting.core.data_processor import DataProcessor
from time_series_forecasting.core.window_generator import WindowGenerator
from time_series_forecasting.models.factory.model_factory import ModelFactory
from time_series_forecasting.models.training.model_trainer import ModelTrainer

class BasePipeline:
    """
    Base pipeline class that handles initialization and core functionality.
    """
    
    def __init__(self, 
                 data_path: Optional[str] = None,
                 region: Optional[str] = None,
                 input_width: int = 24,
                 label_width: int = 1,
                 shift: int = 1):
        """
        Initialize BasePipeline.
        
        Args:
            data_path: Path to the data file
            region: Region to filter data
            input_width: Number of input time steps
            label_width: Number of prediction time steps
            shift: Shift between input and label windows
        """
        self.data_path = data_path
        self.region = region
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        
        # Initialize components
        self.data_processor = DataProcessor(data_path, region)
        self.window_generator = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift
        )
        self.model_factory = ModelFactory()
        self.model_trainer = ModelTrainer(self.window_generator)
        
        # Store results
        self.processed_data = None
        self.trained_models = {}
        self.evaluation_results = {}

    def _dataset_to_numpy(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert TensorFlow dataset to numpy arrays.
        
        Args:
            dataset: TensorFlow dataset
            
        Returns:
            Tuple of (X, y) numpy arrays
        """
        X_list, y_list = [], []
        
        for X, y in dataset:
            X_list.append(X.numpy())
            y_list.append(y.numpy())
        
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        
        # Reshape y to 2D if it's 1D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        return X, y

    def prepare_datasets(self, 
                        train_split: float = 0.7,
                        val_split: float = 0.15,
                        test_split: float = 0.15,
                        batch_size: int = 32) -> Tuple[Any, Any, Any]:
        """
        Prepare training, validation, and test datasets.
        
        Args:
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            batch_size: Batch size for datasets
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        print("\n" + "=" * 60)
        print("STEP 3: DATASET PREPARATION")
        print("=" * 60)
        
        # Update window generator splits
        self.window_generator.train_split = train_split
        self.window_generator.val_split = val_split
        self.window_generator.test_split = test_split
        
        # Split data
        if self.processed_data is None:
            raise ValueError("No processed data available. Please run load_and_preprocess_data() first.")
        train_df, val_df, test_df = self.window_generator.split_data(self.processed_data)
        
        # Create datasets
        train_dataset = self.window_generator.get_train_dataset(batch_size)
        val_dataset = self.window_generator.get_val_dataset(batch_size)
        test_dataset = self.window_generator.get_test_dataset(batch_size)
        
        print(f"Datasets prepared:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        
        return train_dataset, val_dataset, test_dataset

    def generate_comprehensive_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive report of all results.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Report text
        """
        print("\n" + "=" * 60)
        print("STEP 8: GENERATING COMPREHENSIVE REPORT")
        print("=" * 60)
        
        # Generate model comparison
        model_names = list(self.model_trainer.evaluation_results.keys())
        comparison_df = self.model_trainer.compare_models(model_names, save_path if save_path is not None else '')
        
        # Generate detailed report
        report = self.model_trainer.generate_report(save_path if save_path is not None else '')
        
        # Add pipeline-specific information
        pipeline_info = f"""
PIPELINE CONFIGURATION:
- Data path: {self.data_path}
- Region: {self.region}
- Input width: {self.input_width}
- Label width: {self.label_width}
- Shift: {self.shift}
- Data shape: {self.processed_data.shape if self.processed_data is not None else 'N/A'}

WINDOW CONFIGURATION:
{self.window_generator.get_window_info()}

DATA PROCESSING:
- Normalization method: {self.data_processor.scaler.__class__.__name__ if self.data_processor.scaler else 'N/A'}
- Missing data handling: Completed
- Data visualization: Completed
"""
        
        full_report = pipeline_info + "\n" + report
        
        if save_path:
            with open(f"{save_path}_full_report.txt", 'w') as f:
                f.write(full_report)
        
        print("Comprehensive report generated.")
        
        return full_report 