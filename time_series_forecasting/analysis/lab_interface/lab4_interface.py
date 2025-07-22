"""
Lab4 Interface Module for Time Series Analysis
"""

from typing import Dict, Any, Optional
import pandas as pd # type: ignore
from datetime import datetime
from pathlib import Path
from time_series_forecasting.core import (
    DataLoader,
    DataCleaner,
    DataTransformer,
    TimeSeriesFeatureEngineer,
    DataValidator,
    WindowGenerator
)
from time_series_forecasting.models import ModelFactory
from time_series_forecasting.analysis.data_analysis import (
    get_basic_statistics,
    get_seasonal_statistics,
    get_rolling_statistics,
    analyze_data_distribution,
    detect_anomalies,
    detect_seasonality,
    detect_trends
)
from time_series_forecasting.analysis.visualization import (
    plot_time_series,
    plot_seasonal_patterns,
    plot_distribution,
    plot_trends,
    plot_anomalies,
    plot_correlation,
    create_report_plots
)
from time_series_forecasting.analysis.lab_interface.task_executor import TaskExecutor
from time_series_forecasting.analysis.lab_interface.result_manager import ResultManager

class Lab4Interface:
    """
    Main interface for Lab 4 implementation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Lab4Interface.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        self.data_transformer = DataTransformer()
        self.feature_engineer = TimeSeriesFeatureEngineer()
        self.data_validator = DataValidator()
        self.model_factory = ModelFactory()
        
        # Initialize task components
        self.task_executor = TaskExecutor(self.config.get('task_config'))
        self.result_manager = ResultManager(self.config.get('result_config'))
        
        # Store data
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.window_generator: Optional[WindowGenerator] = None
        
        # Store results
        self.analysis_results: Dict[str, Any] = {}
        self.task_results: Dict[str, Any] = {}
        self.plots: Dict[str, Any] = {}
    
    def load_data(self, data_path: str, region: Optional[str] = None) -> pd.DataFrame:
        """
        Load and prepare data.
        
        Args:
            data_path: Path to data file
            region: Region to filter
            
        Returns:
            Loaded DataFrame
        """
        # Load data
        self.raw_data = self.data_loader.load_csv(data_path, region)
        
        # Validate data
        validation_results = self.data_validator.validate_data(self.raw_data)
        self.analysis_results['validation'] = validation_results
        
        # Clean data
        self.processed_data = self.data_cleaner.clean_data(self.raw_data)
        
        # Transform data
        self.processed_data = self.data_transformer.fit_transform(
            self.processed_data,
            method=self.config.get('scaling_method', 'minmax')
        )
        
        return self.processed_data
    
    def analyze_data(self, target_col: str = 'MW') -> Dict[str, Any]:
        """
        Perform comprehensive data analysis.
        
        Args:
            target_col: Target column name
            
        Returns:
            Analysis results
        """
        if self.processed_data is None:
            raise ValueError("Data must be loaded first")
        
        # Basic statistics
        self.analysis_results['basic_stats'] = get_basic_statistics(
            self.processed_data,
            target_col=target_col
        )
        
        # Seasonal statistics
        self.analysis_results['seasonal_stats'] = get_seasonal_statistics(
            self.processed_data,
            target_col=target_col
        )
        
        # Rolling statistics
        self.analysis_results['rolling_stats'] = get_rolling_statistics(
            self.processed_data,
            target_col=target_col
        )
        
        # Distribution analysis
        self.analysis_results['distribution'] = analyze_data_distribution(
            self.processed_data,
            target_col=target_col
        )
        
        # Anomaly detection
        self.analysis_results['anomalies'] = detect_anomalies(
            self.processed_data,
            target_col=target_col
        )
        
        # Seasonality detection
        self.analysis_results['seasonality'] = detect_seasonality(
            self.processed_data,
            target_col=target_col
        )
        
        # Trend detection
        self.analysis_results['trends'] = detect_trends(
            self.processed_data,
            target_col=target_col
        )
        
        return self.analysis_results
    
    def create_visualizations(self,
                            target_col: str = 'MW',
                            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Create all visualizations.
        
        Args:
            target_col: Target column name
            output_dir: Directory to save plots
            
        Returns:
            Dictionary of plots
        """
        if self.processed_data is None:
            raise ValueError("Data must be loaded first")
        
        self.plots = create_report_plots(
            self.processed_data,
            target_col=target_col,
            output_dir=output_dir
        )
        
        return self.plots
    
    def execute_task1(self, **kwargs) -> Dict[str, Any]:
        """Execute Task 1: Data preprocessing and analysis."""
        return self.task_executor.execute_task1(
            data=self.processed_data,
            **kwargs
        )
    
    def execute_task2(self, **kwargs) -> Dict[str, Any]:
        """Execute Task 2: Baseline models."""
        return self.task_executor.execute_task2(
            data=self.processed_data,
            **kwargs
        )
    
    def execute_task3(self, **kwargs) -> Dict[str, Any]:
        """Execute Task 3: Deep learning models."""
        return self.task_executor.execute_task3(
            data=self.processed_data,
            **kwargs
        )
    
    def execute_task4(self, **kwargs) -> Dict[str, Any]:
        """Execute Task 4: Transformer models."""
        return self.task_executor.execute_task4(
            data=self.processed_data,
            **kwargs
        )
    
    def save_results(self, output_dir: str) -> None:
        """
        Save all results.
        
        Args:
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save analysis results
        self.result_manager.save_analysis_results(
            self.analysis_results,
            output_dir / 'analysis_results.json'
        )
        
        # Save task results
        self.result_manager.save_task_results(
            self.task_results,
            output_dir / 'task_results.json'
        )
        
        # Save plots
        if self.plots:
            plots_dir = output_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            for name, fig in self.plots.items():
                fig.savefig(plots_dir / f'{name}.png', dpi=300, bbox_inches='tight')
    
    def load_results(self, input_dir: str) -> None:
        """
        Load saved results.
        
        Args:
            input_dir: Input directory
        """
        input_dir = Path(input_dir)
        
        # Load analysis results
        self.analysis_results = self.result_manager.load_analysis_results(
            input_dir / 'analysis_results.json'
        )
        
        # Load task results
        self.task_results = self.result_manager.load_task_results(
            input_dir / 'task_results.json'
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all results.
        
        Returns:
            Summary dictionary
        """
        return {
            'data_info': self.data_loader.get_data_info() if self.raw_data is not None else None,
            'validation_summary': self.data_validator.get_validation_summary() if self.raw_data is not None else None,
            'analysis_summary': {
                key: results.get('summary', results) 
                for key, results in self.analysis_results.items()
            } if self.analysis_results else None,
            'task_summary': self.task_executor.get_summary() if self.task_results else None
        }
    
    def __repr__(self) -> str:
        """String representation."""
        status = []
        if self.raw_data is not None:
            status.append('data_loaded')
        if self.analysis_results:
            status.append('analysis_complete')
        if self.task_results:
            status.append('tasks_complete')
        
        return f"Lab4Interface(status={status})" 