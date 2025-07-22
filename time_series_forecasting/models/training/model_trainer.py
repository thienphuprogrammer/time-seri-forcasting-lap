import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAXResults as ARIMAResults

class ModelTrainer:
    """
    Class for training, evaluating, and comparing time series forecasting models.
    """
    
    def __init__(self, window_generator=None):
        """
        Initialize ModelTrainer.
        
        Args:
            window_generator: WindowGenerator instance for data handling
        """
        self.window_generator = window_generator
        self.trained_models = {}
        self.training_history = {}
        self.evaluation_results = {}
        
    def train_sklearn_model(self, 
                           model, 
                           model_name: str,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           **kwargs) -> Any:
        """
        Train a scikit-learn model.
        
        Args:
            model: Scikit-learn model instance
            model_name: Name for the model
            X_train: Training features
            y_train: Training targets
            **kwargs: Additional arguments for model fitting
            
        Returns:
            Trained model
        """
        print(f"Training {model_name}...")
        
        # Reshape data if needed
        if len(X_train.shape) == 3:
            X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_reshaped = X_train
            
        if len(y_train.shape) == 3:
            y_train_reshaped = y_train.reshape(y_train.shape[0], -1)
        else:
            y_train_reshaped = y_train
        
        # Train model
        model.fit(X_train_reshaped, y_train_reshaped, **kwargs)
        
        self.trained_models[model_name] = model
        print(f"{model_name} training completed.")
        
        return model
    
    def train_tensorflow_model(self, 
                              model, 
                              model_name: str,
                              train_dataset: tf.data.Dataset,
                              val_dataset: tf.data.Dataset,
                              epochs: int = 100,
                              patience: int = 10,
                              **kwargs) -> tf.keras.Model:
        """
        Train a TensorFlow model with early stopping.
        
        Args:
            model: TensorFlow model instance
            model_name: Name for the model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Maximum number of epochs
            patience: Patience for early stopping
            **kwargs: Additional arguments for model fitting
            
        Returns:
            Trained model
        """
        print(f"Training {model_name}...")
        
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        
        # Learning rate reduction callback
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-6
        )
        
        # Train model
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[early_stopping, lr_scheduler],
            verbose=1,
            **kwargs
        )
        
        self.trained_models[model_name] = model
        self.training_history[model_name] = history.history
        
        print(f"{model_name} training completed. Best epoch: {np.argmin(history.history['val_loss']) + 1}")
        
        return model
    
    def evaluate_model(self, 
                      model, 
                      model_name: str,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      scaler=None) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            scaler: Scaler used for data normalization
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        if isinstance(model, (RandomForestRegressor, LinearRegression, ARIMAResults)):
            # Sklearn or statsmodels model
            # Reshape data if needed
            if len(X_test.shape) == 3:
                X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
            else:
                X_test_reshaped = X_test
                
            predictions = model.predict(X_test_reshaped)
            y_test_reshaped = y_test
            
            # Reshape predictions and test data for scaler
            predictions = predictions.reshape(-1, 1)
            y_test_reshaped = y_test_reshaped.reshape(-1, 1)
        else:
            # TensorFlow model
            predictions = model(X_test).numpy()
            y_test_reshaped = y_test
        
        # Inverse transform if scaler provided
        if scaler is not None:
            predictions = scaler.inverse_transform(predictions)
            y_test_reshaped = scaler.inverse_transform(y_test_reshaped)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_reshaped, predictions)
        mse = mean_squared_error(y_test_reshaped, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_reshaped, predictions)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_test_reshaped - predictions) / y_test_reshaped)) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'predictions': predictions,
            'actual': y_test_reshaped
        }
        
        print(f"{model_name} evaluation completed:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def plot_training_history(self, model_name: str, save_path: str = None):
        """
        Plot training history for a TensorFlow model.
        
        Args:
            model_name: Name of the model
            save_path: Path to save the plot
        """
        if model_name not in self.training_history:
            print(f"No training history found for {model_name}")
            return
        
        history = self.training_history[model_name]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title(f'{model_name} - Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE plot
        if 'mae' in history:
            axes[1].plot(history['mae'], label='Training MAE')
            axes[1].plot(history['val_mae'], label='Validation MAE')
            axes[1].set_title(f'{model_name} - MAE')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_{model_name}_training_history.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, model_name: str, save_path: str = None, max_points: int = 1000):
        """
        Plot actual vs predicted values for a model.
        
        Args:
            model_name: Name of the model
            save_path: Path to save the plot
            max_points: Maximum number of points to plot
        """
        if model_name not in self.evaluation_results:
            print(f"No evaluation results found for {model_name}")
            return
        
        results = self.evaluation_results[model_name]
        actual = results['actual']
        predictions = results['predictions']
        
        # Limit points for plotting
        if len(actual) > max_points:
            indices = np.linspace(0, len(actual)-1, max_points, dtype=int)
            actual = actual[indices]
            predictions = predictions[indices]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Time series plot
        axes[0].plot(actual, label='Actual', alpha=0.7)
        axes[0].plot(predictions, label='Predicted', alpha=0.7)
        axes[0].set_title(f'{model_name} - Predictions vs Actual')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True)
        
        # Scatter plot
        axes[1].scatter(actual, predictions, alpha=0.5)
        axes[1].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        axes[1].set_title(f'{model_name} - Actual vs Predicted')
        axes[1].set_xlabel('Actual')
        axes[1].set_ylabel('Predicted')
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_{model_name}_predictions.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self, model_names: List[str], save_path: str = None) -> pd.DataFrame:
        """
        Compare multiple models using their evaluation metrics.
        
        Args:
            model_names: List of model names to compare
            save_path: Path to save the comparison plot
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for model_name in model_names:
            if model_name in self.evaluation_results:
                metrics = self.evaluation_results[model_name]['metrics']
                metrics['Model'] = model_name
                comparison_data.append(metrics)
        
        if not comparison_data:
            print("No evaluation results found for the specified models")
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('Model')
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # MAE comparison
        axes[0, 0].bar(comparison_df.index, comparison_df['MAE'])
        axes[0, 0].set_title('MAE Comparison')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        axes[0, 1].bar(comparison_df.index, comparison_df['RMSE'])
        axes[0, 1].set_title('RMSE Comparison')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R2 comparison
        axes[1, 0].bar(comparison_df.index, comparison_df['R2'])
        axes[1, 0].set_title('R² Comparison')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MAPE comparison
        axes[1, 1].bar(comparison_df.index, comparison_df['MAPE'])
        axes[1, 1].set_title('MAPE Comparison')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def get_best_model(self, metric: str = 'RMSE') -> Tuple[str, Dict[str, float]]:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric: Metric to use for comparison ('MAE', 'RMSE', 'R2', 'MAPE')
            
        Returns:
            Tuple of (best_model_name, best_model_metrics)
        """
        if not self.evaluation_results:
            print("No evaluation results available")
            return None, {}
        
        best_model = None
        best_score = float('inf') if metric in ['MAE', 'RMSE', 'MAPE'] else float('-inf')
        
        for model_name, results in self.evaluation_results.items():
            score = results['metrics'][metric]
            
            if metric in ['MAE', 'RMSE', 'MAPE']:
                if score < best_score:
                    best_score = score
                    best_model = model_name
            else:  # R2
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        return best_model, self.evaluation_results[best_model]['metrics']
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            filepath: Path where to save the model
        """
        if model_name not in self.trained_models:
            print(f"Model {model_name} not found")
            return
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'save'):
            # TensorFlow model
            model.save(filepath)
        else:
            # Scikit-learn model
            import joblib
            joblib.dump(model, filepath)
        
        print(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name for the loaded model
            filepath: Path to the saved model
        """
        try:
            # Try loading as TensorFlow model
            model = tf.keras.models.load_model(filepath)
        except:
            # Try loading as scikit-learn model
            import joblib
            model = joblib.load(filepath)
        
        self.trained_models[model_name] = model
        print(f"Model {model_name} loaded from {filepath}")
    
    def generate_report(self, save_path: str = None) -> str:
        """
        Generate a comprehensive report of all model performances.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Report text
        """
        report = []
        report.append("=" * 60)
        report.append("TIME SERIES FORECASTING MODEL COMPARISON REPORT")
        report.append("=" * 60)
        report.append("")
        
        if not self.evaluation_results:
            report.append("No evaluation results available.")
            return "\n".join(report)
        
        # Summary table
        report.append("MODEL PERFORMANCE SUMMARY:")
        report.append("-" * 40)
        
        comparison_df = self.compare_models(list(self.evaluation_results.keys()))
        report.append(comparison_df.to_string())
        report.append("")
        
        # Best model by each metric
        report.append("BEST MODELS BY METRIC:")
        report.append("-" * 40)
        
        for metric in ['MAE', 'RMSE', 'R2', 'MAPE']:
            best_model, metrics = self.get_best_model(metric)
            if best_model:
                report.append(f"Best {metric}: {best_model} ({metrics[metric]:.4f})")
        
        report.append("")
        
        # Training information
        if self.training_history:
            report.append("TRAINING INFORMATION:")
            report.append("-" * 40)
            
            for model_name, history in self.training_history.items():
                best_epoch = np.argmin(history['val_loss']) + 1
                final_val_loss = history['val_loss'][-1]
                report.append(f"{model_name}: Best epoch {best_epoch}, Final val_loss: {final_val_loss:.4f}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text 