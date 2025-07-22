"""
Task Executor Module for Lab Interface
"""

from typing import Dict, Any, Optional, List
import pandas as pd # type: ignore
import numpy as np # type: ignore
from datetime import datetime
from time_series_forecasting.core import WindowGenerator
from time_series_forecasting.models import ModelFactory

class TaskExecutor:
    """
    Class for executing lab tasks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TaskExecutor.
        
        Args:
            config: Task configuration
        """
        self.config = config or {}
        
        # Initialize components
        self.model_factory = ModelFactory()
        
        # Store results
        self.results: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
    
    def execute_task1(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Execute Task 1: Data preprocessing and analysis.
        
        Args:
            data: Input DataFrame
            **kwargs: Additional arguments
            
        Returns:
            Task results
        """
        window_config = self.config.get('window_config', {})
        window_config.update(kwargs.get('window_config', {}))
        
        # Create window generator
        window_generator = WindowGenerator(**window_config)
        train_data, val_data, test_data = window_generator.split_data(data)
        
        # Store results
        results = {
            'data_shape': data.shape,
            'window_config': window_config,
            'splits': {
                'train': len(train_data),
                'val': len(val_data),
                'test': len(test_data)
            }
        }
        
        self.results['task1'] = results
        self.history.append({
            'task': 'task1',
            'timestamp': datetime.now(),
            'results': results
        })
        
        return results
    
    def execute_task2(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute Task 2: Baseline models."""
        model_configs = self.config.get('baseline_models', [])
        model_configs.extend(kwargs.get('model_configs', []))
        
        # Remove model_configs from kwargs to avoid duplicate argument error
        clean_kwargs = {k: v for k, v in kwargs.items() if k != 'model_configs'}
        return self._execute_model_task('task2', data, model_configs, **clean_kwargs)
    
    def execute_task3(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute Task 3: Deep learning models."""
        model_configs = self.config.get('deep_learning_models', [])
        model_configs.extend(kwargs.get('model_configs', []))
        
        # Remove model_configs from kwargs to avoid duplicate argument error
        clean_kwargs = {k: v for k, v in kwargs.items() if k != 'model_configs'}
        return self._execute_model_task('task3', data, model_configs, **clean_kwargs)
    
    def execute_task4(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute Task 4: Transformer models."""
        model_configs = self.config.get('transformer_models', [])
        model_configs.extend(kwargs.get('model_configs', []))
        
        # Remove model_configs from kwargs to avoid duplicate argument error
        clean_kwargs = {k: v for k, v in kwargs.items() if k != 'model_configs'}
        return self._execute_model_task('task4', data, model_configs, **clean_kwargs)
    
    def _execute_model_task(self, task_name: str, data: pd.DataFrame, 
                           model_configs: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Execute model training task with given configurations.
        
        Args:
            task_name: Name of the task
            data: Input DataFrame
            model_configs: List of model configurations
            **kwargs: Additional arguments
            
        Returns:
            Task results
        """
        # Get configurations
        window_config = self.config.get('window_config', {})
        window_config.update(kwargs.get('window_config', {}))
        
        # Create window generator
        window_generator = WindowGenerator(**window_config)
        train_df, val_df, test_df = window_generator.split_data(data)
        
        # Train and evaluate models
        models = {}
        predictions = {}
        metrics = {}
        
        for model_config in model_configs:
            model_type = model_config['type']
            model_name = model_config.get('name', model_type)
            
            # Create and train model
            model = self.model_factory.create_model(
                model_type=model_type,
                config=model_config.get('config')
            )
            
            # Build model
            # Calculate input shape for all models (optional for traditional models)
            input_width = window_config.get('input_width', 24)
            n_features = len(data.columns)
            input_shape = (input_width, n_features)
            model.build(input_shape)
            
            # Prepare data based on model type
            if model_type in ['rnn', 'gru', 'lstm', 'transformer']:
                # Deep learning models need TensorFlow datasets
                train_data = window_generator.make_dataset(train_df, shuffle=True, batch_size=32)
                val_data = window_generator.make_dataset(val_df, shuffle=False, batch_size=32) if val_df is not None else None
                test_data = window_generator.make_dataset(test_df, shuffle=False, batch_size=32)
            else:
                # Traditional models work with pandas DataFrames
                train_data = train_df
                val_data = val_df
                test_data = test_df
            
            # Train model
            train_history = model.fit(
                train_data=train_data,
                validation_data=val_data,
                **model_config.get('train_params', {})
            )
            
            # Make predictions
            predictions[model_name] = model.predict(
                data=test_data,
                **model_config.get('predict_params', {})
            )
            
            # Evaluate model
            metrics[model_name] = model.evaluate(
                data=test_data,
                metrics=model_config.get('metrics'),
                **model_config.get('evaluate_params', {})
            )
            
            # Store model
            models[model_name] = model
        
        # Store results
        results = {
            'models': {
                name: {
                    'type': model.__class__.__name__,
                    'config': model.get_params(),
                    'metrics': metrics[name]
                }
                for name, model in models.items()
            },
            'predictions': {
                name: pred.tolist() if isinstance(pred, np.ndarray) else pred
                for name, pred in predictions.items()
            }
        }
        
        self.results[task_name] = results
        self.history.append({
            'task': task_name,
            'timestamp': datetime.now(),
            'results': results
        })
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of task execution.
        
        Returns:
            Summary dictionary
        """
        return {
            'completed_tasks': list(self.results.keys()),
            'execution_history': self.history,
            'best_models': {
                task: self._get_best_model(results)
                for task, results in self.results.items()
                if 'models' in results
            }
        }
    
    def _get_best_model(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get best performing model from results.
        
        Args:
            results: Task results
            
        Returns:
            Best model information
        """
        if 'models' not in results:
            return {}
        
        best_score = float('inf')
        best_model = None
        
        for name, model_info in results['models'].items():
            metrics = model_info['metrics']
            # Use first metric for comparison
            metric = list(metrics.keys())[0]
            score = metrics[metric]
            
            if score < best_score:
                best_score = score
                best_model = {
                    'name': name,
                    'type': model_info['type'],
                    'metrics': metrics
                }
        
        return best_model or {} # type: ignore
    
    def __repr__(self) -> str:
        """String representation."""
        return f"TaskExecutor(completed_tasks={list(self.results.keys())})" 