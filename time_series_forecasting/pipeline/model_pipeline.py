from typing import Dict, Any, Optional, List
import numpy as np
from tensorflow import keras

from .data_pipeline import DataPipeline

class ModelPipeline(DataPipeline):
    """
    Pipeline class that handles model training and evaluation.
    """
    
    def train_baseline_models(self, 
                             train_dataset: Any,
                             val_dataset: Any,
                             test_dataset: Any,
                             **kwargs) -> Dict[str, Any]:
        """
        Train baseline models (Linear Regression, ARIMA, Random Forest).
        
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
        
        # Convert datasets to numpy arrays for sklearn models
        X_train, y_train = self._dataset_to_numpy(train_dataset)
        X_test, y_test = self._dataset_to_numpy(test_dataset)
        
        # Initialize results dictionary
        self.trained_models = {}
        
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
        
        # Store Linear Regression results
        self.trained_models['linear_regression'] = {
            'model': lr_model,
            'metrics': lr_metrics,
            'history': None
        }
        
        # 2. ARIMA (if data is suitable)
        if self.processed_data is not None and len(self.processed_data) > 100:
            print("\nTraining ARIMA...")
            try:
                original_data = self.data_processor.data.iloc[:, 0]
                order = (1, 1, 1)  # Can be optimized
                arima_model = self.model_factory.create_arima(original_data, order=order)
                fitted_arima = arima_model.fit()
                
                self.trained_models['arima'] = {
                    'model': fitted_arima,
                    'metrics': None,
                    'history': None
                }
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
        
        # Store Random Forest results
        self.trained_models['random_forest'] = {
            'model': rf_model,
            'metrics': rf_metrics,
            'history': None
        }
        
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
        example_inputs, _ = next(iter(train_dataset))
        input_shape = example_inputs.shape[1:]
        
        # Convert test dataset to numpy for evaluation
        X_test, y_test = self._dataset_to_numpy(test_dataset)
        
        models_config = {
            'rnn': {
                'create_fn': self.model_factory.create_rnn,
                'kwargs': {'units': 64, 'layers': 2, 'dropout': 0.2}
            },
            'gru': {
                'create_fn': self.model_factory.create_gru,
                'kwargs': {'units': 64, 'layers': 2, 'dropout': 0.2}
            },
            'lstm': {
                'create_fn': self.model_factory.create_lstm,
                'kwargs': {'units': 64, 'layers': 2, 'dropout': 0.2}
            },
            'cnn_lstm': {
                'create_fn': self.model_factory.create_cnn_lstm,
                'kwargs': {
                    'cnn_filters': 64,
                    'cnn_kernel_size': 3,
                    'lstm_units': 64,
                    'dropout': 0.2
                }
            }
        }
        
        for model_name, config in models_config.items():
            print(f"\nTraining {model_name.upper()}...")
            model = config['create_fn'](input_shape=input_shape, **config['kwargs'])
            
            self.model_trainer.train_tensorflow_model(
                model, model_name, train_dataset, val_dataset,
                epochs=epochs, patience=patience
            )
            
            # Evaluate model
            metrics = self.model_trainer.evaluate_model(
                model, model_name, X_test, y_test,
                self.data_processor.scaler
            )
            
            # Store results
            self.trained_models[model_name] = {
                'model': model,
                'metrics': metrics,
                'history': self.model_trainer.training_history[model_name]
            }
        
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
        example_inputs, _ = next(iter(train_dataset))
        input_shape = example_inputs.shape[1:]
        
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
        
        # Store results
        self.trained_models['transformer'] = {
            'model': transformer_model,
            'metrics': transformer_metrics,
            'history': self.model_trainer.training_history['transformer']
        }
        
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
            if name in self.trained_models:
                models.append(self.trained_models[name]['model'])
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
        
        # Store results
        self.trained_models['ensemble'] = {
            'model': ensemble_model,
            'metrics': ensemble_metrics,
            'history': None
        }
        
        print(f"Ensemble model created using {method} method.")
        return ensemble_model 