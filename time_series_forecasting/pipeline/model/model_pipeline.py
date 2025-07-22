"""
Model Pipeline Module for Time Series Forecasting
"""

from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from datetime import datetime
from ...core import WindowGenerator
from ...models import ModelFactory
from ..base import BasePipeline

class ModelPipeline(BasePipeline):
    """
    Pipeline for model training and evaluation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ModelPipeline.
        
        Args:
            config: Pipeline configuration
        """
        super().__init__(config)
        
        # Initialize components
        self.model_factory = ModelFactory()
        self.models = {}
        self.predictions = {}
        self.metrics = {}
    
    def run(self, data: Union[pd.DataFrame, WindowGenerator], **kwargs) -> Dict[str, Any]:
        """
        Run the model pipeline.
        
        Args:
            data: Input data or window generator
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing model results
        """
        if not self.validate():
            raise ValueError("Invalid pipeline configuration")
        
        self.start_time = datetime.now()
        
        try:
            # Get model configurations
            model_configs = self.config.get('models', [])
            if not model_configs:
                raise ValueError("No models specified in configuration")
            
            # Train and evaluate each model
            for model_config in model_configs:
                model_type = model_config['type']
                model_name = model_config.get('name', model_type)
                
                self.history.append({
                    'step': 'train_model',
                    'timestamp': datetime.now(),
                    'model': model_name
                })
                
                # Create and train model
                model = self.model_factory.create_model(
                    model_type=model_type,
                    config=model_config.get('config')
                )
                
                # Build model
                model.build()
                
                # Train model
                train_history = model.fit(
                    train_data=data,
                    validation_data=kwargs.get('validation_data'),
                    **model_config.get('train_params', {})
                )
                
                # Make predictions
                predictions = model.predict(
                    data=kwargs.get('test_data', data),
                    **model_config.get('predict_params', {})
                )
                
                # Evaluate model
                metrics = model.evaluate(
                    data=kwargs.get('test_data', data),
                    metrics=model_config.get('metrics'),
                    **model_config.get('evaluate_params', {})
                )
                
                # Store results
                self.models[model_name] = model
                self.predictions[model_name] = predictions
                self.metrics[model_name] = metrics
                
                self.history.append({
                    'step': 'evaluate_model',
                    'timestamp': datetime.now(),
                    'model': model_name,
                    'metrics': metrics
                })
            
            # Compare models
            best_model = self._get_best_model()
            
            # Store results
            self.results = {
                'models': {
                    name: {
                        'type': model.__class__.__name__,
                        'config': model.get_params(),
                        'metrics': self.metrics[name]
                    }
                    for name, model in self.models.items()
                },
                'best_model': {
                    'name': best_model,
                    'metrics': self.metrics[best_model] if best_model else None
                }
            }
            
            self.end_time = datetime.now()
            return self.results
            
        except Exception as e:
            self.history.append({
                'step': 'error',
                'timestamp': datetime.now(),
                'error': str(e)
            })
            raise
    
    def validate(self) -> bool:
        """
        Validate pipeline configuration.
        
        Returns:
            True if configuration is valid
        """
        if not self.config:
            print("No configuration provided")
            return False
        
        if 'models' not in self.config:
            print("No models specified in configuration")
            return False
        
        for model_config in self.config['models']:
            if 'type' not in model_config:
                print("Model type not specified")
                return False
            
            if model_config['type'] not in self.model_factory.get_available_models():
                print(f"Unknown model type: {model_config['type']}")
                return False
        
        return True
    
    def get_model(self, name: str):
        """
        Get a trained model by name.
        
        Args:
            name: Model name
            
        Returns:
            Trained model instance
        """
        if name not in self.models:
            raise ValueError(f"Model not found: {name}")
        return self.models[name]
    
    def get_predictions(self, name: str) -> np.ndarray:
        """
        Get predictions for a model.
        
        Args:
            name: Model name
            
        Returns:
            Model predictions
        """
        if name not in self.predictions:
            raise ValueError(f"Predictions not found: {name}")
        return self.predictions[name]
    
    def get_metrics(self, name: str) -> Dict[str, float]:
        """
        Get metrics for a model.
        
        Args:
            name: Model name
            
        Returns:
            Model metrics
        """
        if name not in self.metrics:
            raise ValueError(f"Metrics not found: {name}")
        return self.metrics[name]
    
    def _get_best_model(self) -> Optional[str]:
        """
        Get name of best performing model.
        
        Returns:
            Name of best model
        """
        if not self.metrics:
            return None
        
        # Use first metric for comparison
        metric = list(next(iter(self.metrics.values())).keys())[0]
        
        best_score = float('inf')
        best_model = None
        
        for name, metrics in self.metrics.items():
            score = metrics[metric]
            if score < best_score:
                best_score = score
                best_model = name
        
        return best_model
    
    def __repr__(self) -> str:
        """String representation."""
        status = 'trained' if self.models else 'not trained'
        return f"ModelPipeline(status='{status}', models={list(self.models.keys())})" 