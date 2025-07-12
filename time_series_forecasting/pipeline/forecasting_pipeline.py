import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from ..core.data_processor import DataProcessor
from ..core.window_generator import WindowGenerator
from ..models.model_factory import ModelFactory
from ..models.model_trainer import ModelTrainer

class ForecastingPipeline:
    """
    Main pipeline class that orchestrates the entire time series forecasting workflow.
    """
    
    def __init__(self, 
                 data_path: Optional[str] = None,
                 region: Optional[str] = None,
                 input_width: int = 24,
                 label_width: int = 1,
                 shift: int = 1):
        """
        Initialize ForecastingPipeline.
        
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
        
    def load_and_preprocess_data(self, 
                                datetime_col: str = 'Datetime',
                                datetime_format: str = '%Y-%m-%d %H:%M:%S',
                                missing_method: str = 'interpolate',
                                normalize_method: str = 'minmax',
                                target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Load and preprocess the data.
        
        Args:
            datetime_col: Name of datetime column
            datetime_format: Format of datetime strings
            missing_method: Method to handle missing data
            normalize_method: Method to normalize data
            target_column: Target column for forecasting
            
        Returns:
            Preprocessed DataFrame
        """
        print("=" * 60)
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("=" * 60)
        
        # Load data
        self.data_processor.load_data(self.data_path, self.region)
        
        # Parse datetime
        self.data_processor.parse_datetime(datetime_col, datetime_format)
        
        # Handle missing data
        self.data_processor.handle_missing_data(missing_method)
        
        # Normalize data
        self.data_processor.normalize_data(normalize_method)
        
        # Prepare for forecasting
        self.processed_data = self.data_processor.prepare_for_forecasting(target_column if target_column is not None else 'MW')
        
        print(f"Data preprocessing completed. Shape: {self.processed_data.shape}")
        
        return self.processed_data
    
    def create_visualizations(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create comprehensive visualizations of the data.
        
        Args:
            save_path: Path to save visualizations
            
        Returns:
            Dictionary of plot objects
        """
        print("\n" + "=" * 60)
        print("STEP 2: DATA VISUALIZATION")
        print("=" * 60)
        
        plots = self.data_processor.create_visualizations(save_path if save_path is not None else '')
        
        print("Visualizations completed.")
        
        return plots
    
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
    
    def train_baseline_models(self, 
                             train_dataset: Any,
                             val_dataset: Any,
                             test_dataset: Any,
                             **kwargs) -> Dict[str, Any]:
        """
        Train baseline models (Linear Regression, ARIMA, SARIMA).
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            **kwargs: Additional arguments for model training
            
        Returns:
            Dictionary of trained models
        """
        print("\n" + "=" * 60)
        print("STEP 4: BASELINE MODELS TRAINING")
        print("=" * 60)
        
        # Get example batch for input shape
        example_inputs, example_labels = self.window_generator.get_example_batch()
        input_shape = example_inputs.shape[1:]
        
        # Convert datasets to numpy arrays for sklearn models
        X_train, y_train = self._dataset_to_numpy(train_dataset)
        X_val, y_val = self._dataset_to_numpy(val_dataset)
        X_test, y_test = self._dataset_to_numpy(test_dataset)
        
        # 1. Linear Regression
        print("\nTraining Linear Regression...")
        lr_model = self.model_factory.create_linear_regression()
        self.model_trainer.train_sklearn_model(
            lr_model, 'linear_regression', X_train, y_train
        )
        
        # Evaluate Linear Regression
        lr_metrics = self.model_trainer.evaluate_model(
            lr_model, 'linear_regression', X_test, y_test, 
            self.data_processor.scaler
        )
        
        # 2. ARIMA (if data is suitable)
        if self.processed_data is not None and len(self.processed_data) > 100:  # Need sufficient data for ARIMA
            print("\nTraining ARIMA...")
            try:
                # Use the original data (not windowed) for ARIMA
                original_data = self.data_processor.data.iloc[:, 0] if self.data_processor.data is not None else None  # First column
                
                # Determine ARIMA order (simple approach)
                order = (1, 1, 1)  # Can be optimized
                arima_model = self.model_factory.create_arima(original_data, order=order)
                
                # Fit ARIMA
                fitted_arima = arima_model.fit()
                self.trained_models['arima'] = fitted_arima
                
                # For ARIMA, we need to evaluate differently
                print("ARIMA training completed.")
                
            except Exception as e:
                print(f"ARIMA training failed: {e}")
        
        # 3. Random Forest
        print("\nTraining Random Forest...")
        rf_model = self.model_factory.create_random_forest(n_estimators=100, random_state=42)
        self.model_trainer.train_sklearn_model(
            rf_model, 'random_forest', X_train, y_train
        )
        
        # Evaluate Random Forest
        rf_metrics = self.model_trainer.evaluate_model(
            rf_model, 'random_forest', X_test, y_test,
            self.data_processor.scaler
        )
        
        print("\nBaseline models training completed.")
        
        return self.trained_models
    
    def train_deep_learning_models(self, 
                                  train_dataset: Any,
                                  val_dataset: Any,
                                  test_dataset: Any,
                                  epochs: int = 100,
                                  patience: int = 10,
                                  **kwargs) -> Dict[str, Any]:
        """
        Train deep learning models (RNN, GRU, LSTM, CNN-LSTM).
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            epochs: Maximum number of epochs
            patience: Patience for early stopping
            **kwargs: Additional arguments for model training
            
        Returns:
            Dictionary of trained models
        """
        print("\n" + "=" * 60)
        print("STEP 5: DEEP LEARNING MODELS TRAINING")
        print("=" * 60)
        
        # Get example batch for input shape
        example_inputs, example_labels = self.window_generator.get_example_batch()
        input_shape = tuple(example_inputs.shape[1:].as_list()) if example_inputs is not None else (0, 0)  # type: ignore
        input_shape = (input_shape[0] or 0, input_shape[1] or 0) if len(input_shape) >= 2 else (0, 0)
        
        # Convert test dataset to numpy for evaluation
        X_test, y_test = self._dataset_to_numpy(test_dataset)
        
        # 1. RNN
        print("\nTraining RNN...")
        rnn_model = self.model_factory.create_rnn(
            input_shape=input_shape,
            units=64,
            layers=2,
            dropout=0.2
        )
        
        self.model_trainer.train_tensorflow_model(
            rnn_model, 'rnn', train_dataset, val_dataset,
            epochs=epochs, patience=patience
        )
        
        # Evaluate RNN
        rnn_metrics = self.model_trainer.evaluate_model(
            rnn_model, 'rnn', X_test, y_test,
            self.data_processor.scaler
        )
        
        # 2. GRU
        print("\nTraining GRU...")
        gru_model = self.model_factory.create_gru(
            input_shape=input_shape,
            units=64,
            layers=2,
            dropout=0.2
        )
        
        self.model_trainer.train_tensorflow_model(
            gru_model, 'gru', train_dataset, val_dataset,
            epochs=epochs, patience=patience
        )
        
        # Evaluate GRU
        gru_metrics = self.model_trainer.evaluate_model(
            gru_model, 'gru', X_test, y_test,
            self.data_processor.scaler
        )
        
        # 3. LSTM
        print("\nTraining LSTM...")
        lstm_model = self.model_factory.create_lstm(
            input_shape=input_shape,
            units=64,
            layers=2,
            dropout=0.2
        )
        
        self.model_trainer.train_tensorflow_model(
            lstm_model, 'lstm', train_dataset, val_dataset,
            epochs=epochs, patience=patience
        )
        
        # Evaluate LSTM
        lstm_metrics = self.model_trainer.evaluate_model(
            lstm_model, 'lstm', X_test, y_test,
            self.data_processor.scaler
        )
        
        # 4. CNN-LSTM
        print("\nTraining CNN-LSTM...")
        cnn_lstm_model = self.model_factory.create_cnn_lstm(
            input_shape=input_shape,
            cnn_filters=64,
            cnn_kernel_size=3,
            lstm_units=64,
            dropout=0.2
        )
        
        self.model_trainer.train_tensorflow_model(
            cnn_lstm_model, 'cnn_lstm', train_dataset, val_dataset,
            epochs=epochs, patience=patience
        )
        
        # Evaluate CNN-LSTM
        cnn_lstm_metrics = self.model_trainer.evaluate_model(
            cnn_lstm_model, 'cnn_lstm', X_test, y_test,
            self.data_processor.scaler
        )
        
        print("\nDeep learning models training completed.")
        
        return self.trained_models
    
    def train_transformer_model(self, 
                               train_dataset: Any,
                               val_dataset: Any,
                               test_dataset: Any,
                               epochs: int = 100,
                               patience: int = 10,
                               **kwargs) -> Any:
        """
        Train a Transformer model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            epochs: Maximum number of epochs
            patience: Patience for early stopping
            **kwargs: Additional arguments for model training
            
        Returns:
            Trained transformer model
        """
        print("\n" + "=" * 60)
        print("STEP 6: TRANSFORMER MODEL TRAINING")
        print("=" * 60)
        
        # Get example batch for input shape
        example_inputs, example_labels = self.window_generator.get_example_batch()
        input_shape = tuple(example_inputs.shape[1:].as_list()) if example_inputs is not None else (0, 0)  # type: ignore
        input_shape = (input_shape[0] or 0, input_shape[1] or 0) if len(input_shape) >= 2 else (0, 0)
        
        # Convert test dataset to numpy for evaluation
        X_test, y_test = self._dataset_to_numpy(test_dataset)
        
        # Create and train Transformer
        print("\nTraining Transformer...")
        transformer_model = self.model_factory.create_transformer(
            input_shape=input_shape,
            num_heads=8,
            d_model=128,
            num_layers=4,
            dff=512,
            dropout=0.1
        )
        
        self.model_trainer.train_tensorflow_model(
            transformer_model, 'transformer', train_dataset, val_dataset,
            epochs=epochs, patience=patience
        )
        
        # Evaluate Transformer
        transformer_metrics = self.model_trainer.evaluate_model(
            transformer_model, 'transformer', X_test, y_test,
            self.data_processor.scaler
        )
        
        print("\nTransformer model training completed.")
        
        return transformer_model
    
    def create_ensemble_model(self, 
                             model_names: List[str],
                             method: str = 'average',
                             **kwargs) -> Any:
        """
        Create an ensemble model from trained models.
        
        Args:
            model_names: Names of models to ensemble
            method: Ensemble method ('average', 'weighted', 'voting')
            **kwargs: Additional arguments
            
        Returns:
            Ensemble model
        """
        print("\n" + "=" * 60)
        print("STEP 7: ENSEMBLE MODEL CREATION")
        print("=" * 60)
        
        # Get trained models
        models = []
        for name in model_names:
            if name in self.model_trainer.trained_models:
                models.append(self.model_trainer.trained_models[name])
            else:
                print(f"Warning: Model {name} not found")
        
        if not models:
            print("No models available for ensemble")
            return None
        
        # Create ensemble
        ensemble_model = self.model_factory.create_ensemble_model(models, method)
        
        # Evaluate ensemble
        test_dataset = self.window_generator.get_test_dataset()
        X_test, y_test = self._dataset_to_numpy(test_dataset)
        
        ensemble_metrics = self.model_trainer.evaluate_model(
            ensemble_model, 'ensemble', X_test, y_test,
            self.data_processor.scaler
        )
        
        print(f"Ensemble model created using {method} method.")
        
        return ensemble_model
    
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
    
    def run_complete_pipeline(self, 
                             save_path: Optional[str] = None,
                             train_ensemble: bool = True,
                             **kwargs) -> Dict[str, Any]:
        """
        Run the complete forecasting pipeline.
        
        Args:
            save_path: Path to save results
            train_ensemble: Whether to train ensemble model
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with all results
        """
        print("=" * 80)
        print("STARTING COMPLETE TIME SERIES FORECASTING PIPELINE")
        print("=" * 80)
        
        # Step 1: Load and preprocess data
        self.load_and_preprocess_data(**kwargs)
        
        # Step 2: Create visualizations
        self.create_visualizations(save_path if save_path is not None else '')
        
        # Step 3: Prepare datasets
        train_dataset, val_dataset, test_dataset = self.prepare_datasets(**kwargs)
        
        # Step 4: Train baseline models
        self.train_baseline_models(train_dataset, val_dataset, test_dataset, **kwargs)
        
        # Step 5: Train deep learning models
        self.train_deep_learning_models(train_dataset, val_dataset, test_dataset, **kwargs)
        
        # Step 6: Train transformer model
        self.train_transformer_model(train_dataset, val_dataset, test_dataset, **kwargs)
        
        # Step 7: Create ensemble (optional)
        if train_ensemble:
            deep_models = ['rnn', 'gru', 'lstm', 'cnn_lstm', 'transformer']
            self.create_ensemble_model(deep_models, method='average')
        
        # Step 8: Generate comprehensive report
        report = self.generate_comprehensive_report(save_path)
        
        # Plot training histories
        for model_name in self.model_trainer.training_history.keys():
            self.model_trainer.plot_training_history(model_name, save_path if save_path is not None else '')
            self.model_trainer.plot_predictions(model_name, save_path if save_path is not None else '')
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return {
            'data_processor': self.data_processor,
            'window_generator': self.window_generator,
            'model_trainer': self.model_trainer,
            'report': report
        } 