from typing import Dict, Any, Optional
from .model_pipeline import ModelPipeline

class MainPipeline(ModelPipeline):
    """
    Main pipeline class that orchestrates the entire time series forecasting workflow.
    """
    
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
        
        # Plot training histories and predictions
        self.plot_training_histories(save_path)
        
        # Plot model comparison
        self.plot_model_comparison(save_path)
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return {
            'data_processor': self.data_processor,
            'window_generator': self.window_generator,
            'model_trainer': self.model_trainer,
            'report': report
        } 