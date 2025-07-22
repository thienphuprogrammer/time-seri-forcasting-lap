"""
Model Factory Module for Time Series Forecasting
"""

from typing import Dict, Any, Optional, Type
from ..base import BaseTimeSeriesModel
from ..traditional import LinearRegressionModel, ARIMAModel
from ..deep_learning import RNNModel, GRUModel, LSTMModel, TransformerModel

class ModelFactory:
    """
    Factory class for creating time series forecasting models.
    """
    
    def __init__(self):
        """Initialize ModelFactory."""
        self._models: Dict[str, Type[BaseTimeSeriesModel]] = {
            # Traditional models
            'linear': LinearRegressionModel,
            'arima': ARIMAModel,
            
            # Deep learning models
            'rnn': RNNModel,
            'gru': GRUModel,
            'lstm': LSTMModel,
            'transformer': TransformerModel
        }
    
    def create_model(self, model_type: str, config: Optional[Dict[str, Any]] = None) -> BaseTimeSeriesModel:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create
            config: Model configuration
            
        Returns:
            Model instance
        """
        if model_type not in self._models:
            raise ValueError(f"Unknown model type: {model_type}. Available models: {list(self._models.keys())}")
        
        model_class = self._models[model_type]
        model = model_class(name=model_type, config=config)
        
        return model
    
    def register_model(self, model_type: str, model_class: Type[BaseTimeSeriesModel]) -> None:
        """
        Register a new model type.
        
        Args:
            model_type: Type name for the model
            model_class: Model class to register
        """
        if not issubclass(model_class, BaseTimeSeriesModel):
            raise ValueError("Model class must inherit from BaseTimeSeriesModel")
        
        self._models[model_type] = model_class
        print(f"Registered model type: {model_type}")
    
    def get_available_models(self) -> list:
        """Get list of available model types."""
        return list(self._models.keys())
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ModelFactory(available_models={self.get_available_models()})" 