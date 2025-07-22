"""
Data Pipeline Module for Time Series Forecasting
"""

from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
from ...core import (
    DataLoader,
    DataCleaner,
    DataTransformer,
    TimeSeriesFeatureEngineer,
    DataValidator,
    WindowGenerator
)
from ..base import BasePipeline

class DataPipeline(BasePipeline):
    """
    Pipeline for data loading, preprocessing, and feature engineering.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataPipeline.
        
        Args:
            config: Pipeline configuration
        """
        super().__init__(config)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        self.data_transformer = DataTransformer()
        self.feature_engineer = TimeSeriesFeatureEngineer()
        self.data_validator = DataValidator()
        self.window_generator = None
        
        # Store data at each stage
        self.raw_data = None
        self.cleaned_data = None
        self.transformed_data = None
        self.final_data = None
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the data pipeline.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing processed data and metadata
        """
        if not self.validate():
            raise ValueError("Invalid pipeline configuration")
        
        self.start_time = datetime.now()
        
        try:
            # Load data
            self.raw_data = self.data_loader.load_csv(
                data_path=self.config.get('data_path'),
                region=self.config.get('region')
            )
            self.history.append({
                'step': 'load_data',
                'timestamp': datetime.now(),
                'shape': self.raw_data.shape
            })
            
            # Validate raw data
            validation_results = self.data_validator.validate_data(self.raw_data)
            self.history.append({
                'step': 'validate_raw_data',
                'timestamp': datetime.now(),
                'results': validation_results
            })
            
            # Clean data
            self.cleaned_data = self.data_cleaner.clean_data(self.raw_data)
            self.history.append({
                'step': 'clean_data',
                'timestamp': datetime.now(),
                'shape': self.cleaned_data.shape
            })
            
            # Transform data
            self.transformed_data = self.data_transformer.fit_transform(
                self.cleaned_data,
                method=self.config.get('scaling_method', 'minmax')
            )
            self.history.append({
                'step': 'transform_data',
                'timestamp': datetime.now(),
                'shape': self.transformed_data.shape
            })
            
            # Create features
            if self.config.get('features'):
                self.final_data = self.feature_engineer.create_features(
                    self.transformed_data,
                    features=self.config['features']
                )
            else:
                self.final_data = self.transformed_data
            
            self.history.append({
                'step': 'create_features',
                'timestamp': datetime.now(),
                'shape': self.final_data.shape
            })
            
            # Create window generator
            if self.config.get('window_config'):
                self.window_generator = WindowGenerator(**self.config['window_config'])
                train_data, val_data, test_data = self.window_generator.split_data(self.final_data)
                
                self.history.append({
                    'step': 'create_windows',
                    'timestamp': datetime.now(),
                    'splits': {
                        'train': len(train_data),
                        'val': len(val_data),
                        'test': len(test_data)
                    }
                })
            
            # Store results
            self.results = {
                'data_info': self.data_loader.get_data_info(),
                'validation_info': self.data_validator.get_validation_summary(),
                'cleaning_info': self.data_cleaner.get_cleaning_summary(),
                'transform_info': self.data_transformer.get_transform_info(),
                'feature_info': self.feature_engineer.get_feature_info(),
                'window_info': self.window_generator.get_window_info() if self.window_generator else None,
                'final_shape': self.final_data.shape
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
        required_config = ['data_path']
        
        # Check required config
        for key in required_config:
            if key not in self.config:
                print(f"Missing required config: {key}")
                return False
        
        # Validate window config if provided
        if 'window_config' in self.config:
            required_window_config = ['input_width', 'label_width', 'shift']
            for key in required_window_config:
                if key not in self.config['window_config']:
                    print(f"Missing required window config: {key}")
                    return False
        
        return True
    
    def get_data(self, stage: str = 'final') -> pd.DataFrame:
        """
        Get data from a specific pipeline stage.
        
        Args:
            stage: Pipeline stage ('raw', 'cleaned', 'transformed', 'final')
            
        Returns:
            Data from specified stage
        """
        if stage == 'raw':
            return self.raw_data
        elif stage == 'cleaned':
            return self.cleaned_data
        elif stage == 'transformed':
            return self.transformed_data
        elif stage == 'final':
            return self.final_data
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def get_window_generator(self) -> Optional[WindowGenerator]:
        """Get the window generator if created."""
        return self.window_generator
    
    def __repr__(self) -> str:
        """String representation."""
        stages = []
        if self.raw_data is not None:
            stages.append('raw')
        if self.cleaned_data is not None:
            stages.append('cleaned')
        if self.transformed_data is not None:
            stages.append('transformed')
        if self.final_data is not None:
            stages.append('final')
        
        return f"DataPipeline(stages={stages})" 