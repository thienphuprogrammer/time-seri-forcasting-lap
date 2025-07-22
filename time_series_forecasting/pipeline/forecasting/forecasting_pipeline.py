"""
Forecasting Pipeline Module for Time Series Forecasting
"""

from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from datetime import datetime
from ...core import WindowGenerator
from ..data import DataPipeline
from ..model import ModelPipeline
from ..base import BasePipeline

class ForecastingPipeline(BasePipeline):
    """
    End-to-end pipeline for time series forecasting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ForecastingPipeline.
        
        Args:
            config: Pipeline configuration
        """
        super().__init__(config)
        
        # Initialize sub-pipelines
        self.data_pipeline = DataPipeline(config.get('data_config') if config else None)
        self.model_pipeline = ModelPipeline(config.get('model_config') if config else None)
        
        # Store forecasts
        self.forecasts = {}
        self.forecast_metrics = {}
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the forecasting pipeline.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing forecasting results
        """
        if not self.validate():
            raise ValueError("Invalid pipeline configuration")
        
        self.start_time = datetime.now()
        
        try:
            # Run data pipeline
            self.history.append({
                'step': 'data_pipeline',
                'timestamp': datetime.now()
            })
            
            data_results = self.data_pipeline.run(**kwargs)
            
            # Get processed data
            data = self.data_pipeline.get_data('final')
            window_generator = self.data_pipeline.get_window_generator()
            
            # Run model pipeline
            self.history.append({
                'step': 'model_pipeline',
                'timestamp': datetime.now()
            })
            
            model_results = self.model_pipeline.run(
                data=window_generator if window_generator else data,
                **kwargs
            )
            
            # Generate forecasts
            self.history.append({
                'step': 'generate_forecasts',
                'timestamp': datetime.now()
            })
            
            forecast_horizon = self.config.get('forecast_horizon', 24)
            for model_name, model in self.model_pipeline.models.items():
                forecasts = self._generate_forecasts(
                    model=model,
                    data=data,
                    horizon=forecast_horizon
                )
                self.forecasts[model_name] = forecasts
            
            # Calculate forecast metrics
            if kwargs.get('test_data') is not None:
                for model_name, forecasts in self.forecasts.items():
                    metrics = self._calculate_forecast_metrics(
                        forecasts=forecasts,
                        actual=kwargs['test_data']
                    )
                    self.forecast_metrics[model_name] = metrics
            
            # Store results
            self.results = {
                'data_pipeline': data_results,
                'model_pipeline': model_results,
                'forecasts': {
                    name: forecasts.tolist() if isinstance(forecasts, np.ndarray) else forecasts
                    for name, forecasts in self.forecasts.items()
                },
                'forecast_metrics': self.forecast_metrics
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
        
        # Validate data pipeline config
        if not self.data_pipeline.validate():
            print("Invalid data pipeline configuration")
            return False
        
        # Validate model pipeline config
        if not self.model_pipeline.validate():
            print("Invalid model pipeline configuration")
            return False
        
        return True
    
    def get_forecasts(self, model_name: str) -> np.ndarray:
        """
        Get forecasts for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model forecasts
        """
        if model_name not in self.forecasts:
            raise ValueError(f"Forecasts not found for model: {model_name}")
        return self.forecasts[model_name]
    
    def get_forecast_metrics(self, model_name: str) -> Dict[str, float]:
        """
        Get forecast metrics for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Forecast metrics
        """
        if model_name not in self.forecast_metrics:
            raise ValueError(f"Metrics not found for model: {model_name}")
        return self.forecast_metrics[model_name]
    
    def _generate_forecasts(self,
                          model: Any,
                          data: pd.DataFrame,
                          horizon: int) -> np.ndarray:
        """
        Generate forecasts using a trained model.
        
        Args:
            model: Trained model
            data: Input data
            horizon: Forecast horizon
            
        Returns:
            Model forecasts
        """
        # Get latest data for forecasting
        forecast_input = self._get_forecast_input(data, model)
        
        # Generate forecasts iteratively
        forecasts = []
        current_input = forecast_input
        
        for _ in range(horizon):
            # Make single-step prediction
            prediction = model.predict(current_input)
            forecasts.append(prediction[0])
            
            # Update input for next prediction
            current_input = self._update_forecast_input(current_input, prediction[0])
        
        return np.array(forecasts)
    
    def _get_forecast_input(self, data: pd.DataFrame, model: Any) -> Union[pd.DataFrame, np.ndarray]:
        """
        Prepare input data for forecasting.
        
        Args:
            data: Input data
            model: Model to use for forecasting
            
        Returns:
            Prepared forecast input
        """
        # Get the last window of data
        if hasattr(model, 'config') and 'input_width' in model.config:
            input_width = model.config['input_width']
            return data.iloc[-input_width:].copy()
        
        # Default to last row if no window size specified
        return data.iloc[[-1]].copy()
    
    def _update_forecast_input(self,
                             current_input: Union[pd.DataFrame, np.ndarray],
                             prediction: float) -> Union[pd.DataFrame, np.ndarray]:
        """
        Update input data with new prediction.
        
        Args:
            current_input: Current input data
            prediction: New prediction
            
        Returns:
            Updated input data
        """
        if isinstance(current_input, pd.DataFrame):
            new_input = current_input.copy()
            new_input = new_input.iloc[1:].append(
                pd.DataFrame([prediction], columns=new_input.columns, index=[new_input.index[-1] + pd.Timedelta('1H')])
            )
            return new_input
        
        # For numpy arrays
        new_input = np.roll(current_input, -1, axis=0)
        new_input[-1] = prediction
        return new_input
    
    def _calculate_forecast_metrics(self,
                                 forecasts: np.ndarray,
                                 actual: Union[pd.DataFrame, np.ndarray]) -> Dict[str, float]:
        """
        Calculate forecast accuracy metrics.
        
        Args:
            forecasts: Forecast values
            actual: Actual values
            
        Returns:
            Dictionary of metrics
        """
        if isinstance(actual, pd.DataFrame):
            actual = actual.values
        
        # Calculate metrics
        metrics = {}
        
        # Mean Absolute Error
        metrics['mae'] = np.mean(np.abs(actual - forecasts))
        
        # Mean Squared Error
        metrics['mse'] = np.mean((actual - forecasts) ** 2)
        
        # Root Mean Squared Error
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Mean Absolute Percentage Error
        mask = actual != 0
        metrics['mape'] = np.mean(np.abs((actual[mask] - forecasts[mask]) / actual[mask])) * 100
        
        return metrics
    
    def __repr__(self) -> str:
        """String representation."""
        status = 'completed' if self.results else 'not run'
        return f"ForecastingPipeline(status='{status}', models={list(self.forecasts.keys())})" 