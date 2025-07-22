from typing import Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt

from .base_pipeline import BasePipeline

class DataPipeline(BasePipeline):
    """
    Pipeline class that handles data loading, preprocessing, and visualization.
    """
    
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
    
    def plot_training_histories(self, save_path: Optional[str] = None):
        """
        Plot training histories for all trained models.
        
        Args:
            save_path: Path to save plots
        """
        for model_name in self.model_trainer.training_history.keys():
            self.model_trainer.plot_training_history(
                model_name, 
                save_path if save_path is not None else ''
            )
            self.model_trainer.plot_predictions(
                model_name, 
                save_path if save_path is not None else ''
            )
    
    def plot_model_comparison(self, save_path: Optional[str] = None):
        """
        Create comparison plots for all models.
        
        Args:
            save_path: Path to save plots
        """
        # Get evaluation metrics
        metrics = {}
        for model_name, results in self.model_trainer.evaluation_results.items():
            if 'metrics' in results:
                metrics[model_name] = results['metrics']
        
        if not metrics:
            print("No metrics available for comparison")
            return
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(metrics).T
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # RMSE comparison
        comparison_df['RMSE'].plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('RMSE Comparison')
        axes[0, 0].set_ylabel('RMSE')
        
        # MAE comparison
        comparison_df['MAE'].plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('MAE Comparison')
        axes[0, 1].set_ylabel('MAE')
        
        # R² comparison
        comparison_df['R2'].plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('R² Score Comparison')
        axes[1, 0].set_ylabel('R²')
        
        # MAPE comparison
        comparison_df['MAPE'].plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('MAPE Comparison')
        axes[1, 1].set_ylabel('MAPE (%)')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(f"{save_path}_model_comparison.png", dpi=300, bbox_inches='tight')
        
        plt.show() 