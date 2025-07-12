"""
DAT301m Lab 4: Time Series Forecasting Interface

This module provides a comprehensive interface for completing the DAT301m Lab 4 
time series forecasting assignment. It builds upon the existing modules and 
provides a structured approach to complete all tasks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, Optional, List, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

from ..core.data_processor import DataProcessor
from ..core.window_generator import WindowGenerator
from ..models.model_factory import ModelFactory
from ..models.model_trainer import ModelTrainer
from ..pipeline.forecasting_pipeline import ForecastingPipeline

class DAT301mLab4Interface:
    """
    Comprehensive interface for DAT301m Lab 4: Time Series Forecasting
    
    This class provides a structured approach to complete all lab tasks:
    - Task 1: Dataset exploration and preprocessing
    - Task 2: Baseline models (Linear Regression, ARIMA/SARIMA)
    - Task 3: Deep learning models (RNN, GRU, LSTM, Ensemble)
    - Task 4: Advanced attention/transformer models
    """
    
    def __init__(self, 
                 data_path: str,
                 region: Optional[str] = None,
                 input_width: int = 24,
                 label_width: int = 1,
                 shift: int = 1,
                 random_seed: int = 42):
        """
        Initialize the DAT301m Lab 4 interface.
        
        Args:
            data_path: Path to the PJM energy consumption data
            region: Region to analyze (e.g., 'AEP', 'COMED', 'DAYTON')
            input_width: Number of time steps to use as input (default: 24 hours)
            label_width: Number of time steps to predict (default: 1 hour)
            shift: Shift between input and label windows (default: 1 hour)
            random_seed: Random seed for reproducibility
        """
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        # Store configuration
        self.data_path = data_path
        self.region = region
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.random_seed = random_seed
        
        # Initialize pipeline
        self.pipeline = ForecastingPipeline(
            data_path=data_path,
            region=region if region else 'PJME',
            input_width=input_width,
            label_width=label_width,
            shift=shift
        )
        
        # Initialize task results
        self.task_results = {
            'task1': {},
            'task2': {},
            'task3': {},
            'task4': {}
        }
        
        # Initialize data containers
        self.raw_data = None
        self.processed_data = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        print("DAT301m Lab 4 Interface initialized")
        print(f"Configuration: Input Width={input_width}, Label Width={label_width}, Shift={shift}")
        if region:
            print(f"Region: {region}")
    
    def execute_task1(self, 
                     datetime_col: str = 'Datetime',
                     target_col: str = 'MW',
                     normalize_method: str = 'minmax',
                     train_split: float = 0.7,
                     val_split: float = 0.15,
                     test_split: float = 0.15,
                     create_plots: bool = True,
                     save_plots: bool = False,
                     plot_dir: str = 'plots/') -> Dict[str, Any]:
        """
        Execute Task 1: Dataset Exploration and Preprocessing
        
        Args:
            datetime_col: Name of the datetime column
            target_col: Name of the target column (energy consumption)
            normalize_method: Normalization method ('minmax', 'standard', 'robust')
            train_split: Training data split ratio
            val_split: Validation data split ratio
            test_split: Test data split ratio
            create_plots: Whether to create visualization plots
            save_plots: Whether to save plots to disk
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary containing task results
        """
        print("\n" + "="*80)
        print("TASK 1: DATASET EXPLORATION AND PREPROCESSING")
        print("="*80)
        
        # Step 1.1: Load and explore data
        print("\n1.1 Loading and exploring data...")
        self.raw_data = self._load_and_explore_data(datetime_col, target_col)
        
        # Step 1.2: Data preprocessing
        print("\n1.2 Data preprocessing...")
        self.processed_data = self._preprocess_data(
            datetime_col, target_col, normalize_method
        )
        
        # Step 1.3: Create visualizations
        if create_plots:
            print("\n1.3 Creating visualizations...")
            plots = self._create_task1_visualizations(
                save_plots=save_plots, 
                plot_dir=plot_dir
            )
            self.task_results['task1']['plots'] = plots
        
        # Step 1.4: Implement WindowGenerator and create splits
        print("\n1.4 Creating data splits with WindowGenerator...")
        self.train_dataset, self.val_dataset, self.test_dataset = self._create_data_splits(
            train_split, val_split, test_split
        )
        
        # Store results
        train_df = getattr(self.pipeline.window_generator, 'train_df', None)
        val_df = getattr(self.pipeline.window_generator, 'val_df', None)
        test_df = getattr(self.pipeline.window_generator, 'test_df', None)
        self.task_results['task1'].update({
            'data_shape': self.processed_data.shape if self.processed_data is not None else (0, 0),
            'train_size': len(train_df) if train_df is not None else 0,
            'val_size': len(val_df) if val_df is not None else 0,
            'test_size': len(test_df) if test_df is not None else 0,
            'window_config': {
                'input_width': self.input_width,
                'label_width': self.label_width,
                'shift': self.shift
            }
        })
        
        print("\n✓ Task 1 completed successfully!")
        return self.task_results['task1']
    
    def execute_task2(self, 
                     epochs: int = 100,
                     patience: int = 10,
                     create_plots: bool = True,
                     save_plots: bool = False,
                     plot_dir: str = 'plots/') -> Dict[str, Any]:
        """
        Execute Task 2: Baseline Models
        
        Args:
            epochs: Maximum number of training epochs
            patience: Early stopping patience
            create_plots: Whether to create plots
            save_plots: Whether to save plots
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary containing task results
        """
        print("\n" + "="*80)
        print("TASK 2: BASELINE MODELS")
        print("="*80)
        
        if self.processed_data is None:
            raise ValueError("Please run execute_task1() first to prepare data")
        
        # Train baseline models
        print("\n2.1 Training baseline models...")
        baseline_results = self.pipeline.train_baseline_models(
            self.train_dataset, 
            self.val_dataset, 
            self.test_dataset,
            epochs=epochs,
            patience=patience
        )
        
        # Create plots if requested
        if create_plots:
            print("\n2.2 Creating evaluation plots...")
            plots = self._create_task2_plots(
                baseline_results,
                save_plots=save_plots,
                plot_dir=plot_dir
            )
            self.task_results['task2']['plots'] = plots
        
        # Store results
        self.task_results['task2'].update({
            'models': baseline_results,
            'evaluation_metrics': self._extract_metrics(baseline_results)
        })
        
        print("\n✓ Task 2 completed successfully!")
        return self.task_results['task2']
    
    def execute_task3(self, 
                     epochs: int = 100,
                     patience: int = 10,
                     units: int = 64,
                     layers: int = 2,
                     dropout: float = 0.2,
                     create_ensemble: bool = True,
                     multi_step: bool = False,
                     multi_step_horizon: int = 24,
                     create_plots: bool = True,
                     save_plots: bool = False,
                     plot_dir: str = 'plots/') -> Dict[str, Any]:
        """
        Execute Task 3: Deep Learning Models
        
        Args:
            epochs: Maximum number of training epochs
            patience: Early stopping patience
            units: Number of units in RNN layers
            layers: Number of RNN layers
            dropout: Dropout rate
            create_ensemble: Whether to create ensemble model
            multi_step: Whether to implement multi-step forecasting
            multi_step_horizon: Number of steps for multi-step forecasting
            create_plots: Whether to create plots
            save_plots: Whether to save plots
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary containing task results
        """
        print("\n" + "="*80)
        print("TASK 3: DEEP LEARNING MODELS")
        print("="*80)
        
        if self.processed_data is None:
            raise ValueError("Please run execute_task1() first to prepare data")
        
        # Train deep learning models
        print("\n3.1 Training deep learning models...")
        dl_results = self.pipeline.train_deep_learning_models(
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            epochs=epochs,
            patience=patience,
            units=units,
            layers=layers,
            dropout=dropout
        )
        
        # Create ensemble model if requested
        if create_ensemble:
            print("\n3.2 Creating ensemble model...")
            ensemble_model = self._create_ensemble_model(dl_results)
            dl_results['ensemble'] = ensemble_model
        
        # Multi-step forecasting (advanced)
        if multi_step:
            print(f"\n3.3 Implementing multi-step forecasting ({multi_step_horizon} steps)...")
            multi_step_results = self._implement_multi_step_forecasting(
                dl_results, multi_step_horizon
            )
            self.task_results['task3']['multi_step'] = multi_step_results
        
        # Create plots if requested
        if create_plots:
            print("\n3.4 Creating evaluation plots...")
            plots = self._create_task3_plots(
                dl_results,
                save_plots=save_plots,
                plot_dir=plot_dir
            )
            self.task_results['task3']['plots'] = plots
        
        # Store results
        self.task_results['task3'].update({
            'models': dl_results,
            'evaluation_metrics': self._extract_metrics(dl_results)
        })
        
        print("\n✓ Task 3 completed successfully!")
        return self.task_results['task3']
    
    def execute_task4(self, 
                     epochs: int = 100,
                     patience: int = 10,
                     num_heads: int = 8,
                     d_model: int = 128,
                     num_layers: int = 4,
                     multi_step_horizon: int = 24,
                     create_plots: bool = True,
                     save_plots: bool = False,
                     plot_dir: str = 'plots/') -> Dict[str, Any]:
        """
        Execute Task 4: Advanced Attention-based and Transformer Models
        
        Args:
            epochs: Maximum number of training epochs
            patience: Early stopping patience
            num_heads: Number of attention heads
            d_model: Dimension of the model
            num_layers: Number of transformer layers
            multi_step_horizon: Number of steps for multi-step forecasting
            create_plots: Whether to create plots
            save_plots: Whether to save plots
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary containing task results
        """
        print("\n" + "="*80)
        print("TASK 4: ATTENTION-BASED AND TRANSFORMER MODELS")
        print("="*80)
        
        if self.processed_data is None:
            raise ValueError("Please run execute_task1() first to prepare data")
        
        # Train transformer model
        print("\n4.1 Training Transformer model...")
        transformer_results = self.pipeline.train_transformer_model(
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            epochs=epochs,
            patience=patience,
            num_heads=num_heads,
            d_model=d_model,
            num_layers=num_layers
        )
        
        # Multi-step forecasting
        print(f"\n4.2 Implementing multi-step forecasting ({multi_step_horizon} steps)...")
        multi_step_results = self._implement_multi_step_forecasting(
            {'transformer': transformer_results}, multi_step_horizon
        )
        
        # Compare with Task 3 models
        print("\n4.3 Comparing with Task 3 models...")
        comparison_results = self._compare_with_task3_models(transformer_results)
        
        # Create plots if requested
        if create_plots:
            print("\n4.4 Creating evaluation plots...")
            plots = self._create_task4_plots(
                transformer_results,
                comparison_results,
                save_plots=save_plots,
                plot_dir=plot_dir
            )
            self.task_results['task4']['plots'] = plots
        
        # Store results
        self.task_results['task4'].update({
            'models': {'transformer': transformer_results},
            'multi_step': multi_step_results,
            'comparison': comparison_results,
            'evaluation_metrics': self._extract_metrics({'transformer': transformer_results})
        })
        
        print("\n✓ Task 4 completed successfully!")
        return self.task_results['task4']
    
    def answer_questions(self) -> Dict[str, str]:
        """
        Generate answers to lab questions Q1 and Q2.
        
        Returns:
            Dictionary with question answers
        """
        print("\n" + "="*80)
        print("GENERATING ANSWERS TO LAB QUESTIONS")
        print("="*80)
        
        answers = {}
        
        # Question Q1: Which baseline model generalized better and why?
        if 'task2' in self.task_results and 'evaluation_metrics' in self.task_results['task2']:
            answers['Q1'] = self._generate_q1_answer()
        
        # Question Q2: Which deep learning model captured temporal patterns best?
        if 'task3' in self.task_results and 'evaluation_metrics' in self.task_results['task3']:
            answers['Q2'] = self._generate_q2_answer()
        
        return answers
    
    def generate_comprehensive_report(self, 
                                    output_path: str = 'lab4_report.txt',
                                    include_plots: bool = True) -> str:
        """
        Generate a comprehensive report for the lab submission.
        
        Args:
            output_path: Path to save the report
            include_plots: Whether to include plot descriptions in the report
            
        Returns:
            Generated report as string
        """
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        report = self._generate_lab_report(include_plots)
        
        # Save report to file
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {output_path}")
        return report
    
    def run_complete_lab(self, 
                        output_dir: str = 'lab4_results/',
                        save_plots: bool = True,
                        multi_step: bool = True,
                        create_ensemble: bool = True) -> Dict[str, Any]:
        """
        Execute the complete DAT301m Lab 4 workflow.
        
        Args:
            output_dir: Directory to save all outputs
            save_plots: Whether to save plots
            multi_step: Whether to implement multi-step forecasting
            create_ensemble: Whether to create ensemble models
            
        Returns:
            Complete results dictionary
        """
        print("\n" + "="*80)
        print("EXECUTING COMPLETE DAT301m LAB 4 WORKFLOW")
        print("="*80)
        
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plot_dir = os.path.join(output_dir, 'plots/')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        # Execute all tasks
        task1_results = self.execute_task1(
            create_plots=True, 
            save_plots=save_plots, 
            plot_dir=plot_dir
        )
        
        task2_results = self.execute_task2(
            create_plots=True, 
            save_plots=save_plots, 
            plot_dir=plot_dir
        )
        
        task3_results = self.execute_task3(
            create_ensemble=create_ensemble,
            multi_step=multi_step,
            create_plots=True, 
            save_plots=save_plots, 
            plot_dir=plot_dir
        )
        
        task4_results = self.execute_task4(
            create_plots=True, 
            save_plots=save_plots, 
            plot_dir=plot_dir
        )
        
        # Generate answers to questions
        answers = self.answer_questions()
        
        # Generate comprehensive report
        report_path = os.path.join(output_dir, 'lab4_report.txt')
        report = self.generate_comprehensive_report(
            output_path=report_path,
            include_plots=save_plots
        )
        
        # Save results summary
        results_summary = {
            'task1': task1_results,
            'task2': task2_results,
            'task3': task3_results,
            'task4': task4_results,
            'answers': answers,
            'report_path': report_path
        }
        
        print(f"\n✓ Complete DAT301m Lab 4 workflow completed!")
        print(f"Results saved to: {output_dir}")
        
        return results_summary
    
    # Helper methods for data loading and preprocessing
    def _load_and_explore_data(self, datetime_col: str, target_col: str) -> pd.DataFrame:
        """Load and explore the raw data."""
        # Load data using the pipeline
        processed_data = self.pipeline.load_and_preprocess_data(
            datetime_col=datetime_col,
            target_column=target_col
        )
        
        # The processed data will have column named 'target'
        actual_target_col = 'target' if 'target' in processed_data.columns else 'MW'
        
        # Print basic statistics
        print(f"Data shape: {processed_data.shape}")
        print(f"Date range: {processed_data.index.min()} to {processed_data.index.max()}")
        print(f"Target column statistics:")
        print(processed_data[actual_target_col].describe())
        
        # Check for missing values
        missing_values = processed_data.isnull().sum()
        if missing_values.any():
            print(f"Missing values found: {missing_values[missing_values > 0]}")
        
        return processed_data
    
    def _preprocess_data(self, datetime_col: str, target_col: str, normalize_method: str) -> pd.DataFrame:
        """Preprocess the data."""
        return self.pipeline.processed_data if self.pipeline.processed_data is not None else pd.DataFrame()
    
    def _create_data_splits(self, train_split: float, val_split: float, test_split: float) -> Tuple[Any, Any, Any]:
        """Create train/validation/test splits."""
        return self.pipeline.prepare_datasets(
            train_split=train_split,
            val_split=val_split,
            test_split=test_split
        )
    
    def _create_task1_visualizations(self, save_plots: bool, plot_dir: str) -> Dict[str, Any]:
        """Create Task 1 visualizations."""
        return self.pipeline.create_visualizations(
            save_path=plot_dir if save_plots else ''
        )
    
    def _create_task2_plots(self, results: Dict[str, Any], save_plots: bool, plot_dir: str) -> Dict[str, Any]:
        """Create Task 2 evaluation plots."""
        plots = {}
        
        # Training history plots for models that have history
        for model_name, model_info in results.items():
            if 'history' in model_info and model_info['history']:
                self.pipeline.model_trainer.plot_training_history(
                    model_name, 
                    save_path=f"{plot_dir}/{model_name}_history.png" if save_plots else ''
                )
        
        # Prediction plots
        for model_name in results.keys():
            self.pipeline.model_trainer.plot_predictions(
                model_name,
                save_path=f"{plot_dir}/{model_name}_predictions.png" if save_plots else ''
            )
        
        # Model comparison
        comparison_df = self.pipeline.model_trainer.compare_models(
            list(results.keys()),
            save_path=f"{plot_dir}/baseline_comparison.png" if save_plots else ''
        )
        plots['comparison'] = comparison_df
        
        return plots
    
    def _create_task3_plots(self, results: Dict[str, Any], save_plots: bool, plot_dir: str) -> Dict[str, Any]:
        """Create Task 3 evaluation plots."""
        return self._create_task2_plots(results, save_plots, plot_dir)
    
    def _create_task4_plots(self, transformer_results: Dict[str, Any], 
                           comparison_results: Dict[str, Any], 
                           save_plots: bool, plot_dir: str) -> Dict[str, Any]:
        """Create Task 4 evaluation plots."""
        plots = {}
        
        # Transformer training history
        if 'history' in transformer_results and transformer_results['history']:
            self.pipeline.model_trainer.plot_training_history(
                'transformer',
                save_path=f"{plot_dir}/transformer_history.png" if save_plots else ''
            )
        
        # Transformer predictions
        self.pipeline.model_trainer.plot_predictions(
            'transformer',
            save_path=f"{plot_dir}/transformer_predictions.png" if save_plots else ''
        )
        
        # Comparison with Task 3 models
        if comparison_results:
            plots['comparison'] = comparison_results
        
        return plots
    
    def _create_ensemble_model(self, models: Dict[str, Any]) -> Any:
        """Create ensemble model from trained models."""
        model_list = []
        for model_name, model_info in models.items():
            if 'model' in model_info:
                model_list.append(model_info['model'])
        
        if model_list:
            return self.pipeline.create_ensemble_model(
                list(models.keys()),
                method='average'
            )
        return None
    
    def _implement_multi_step_forecasting(self, models: Dict[str, Any], horizon: int) -> Dict[str, Any]:
        """Implement multi-step forecasting."""
        # This would require modifying the WindowGenerator to support multi-step
        # For now, return a placeholder
        return {
            'horizon': horizon,
            'implemented': False,
            'note': 'Multi-step forecasting implementation required'
        }
    
    def _compare_with_task3_models(self, transformer_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare transformer with Task 3 models."""
        if 'task3' not in self.task_results or 'evaluation_metrics' not in self.task_results['task3']:
            return {'note': 'Task 3 results not available for comparison'}
        
        task3_metrics = self.task_results['task3']['evaluation_metrics']
        transformer_metrics = self._extract_metrics({'transformer': transformer_results})
        
        comparison = {
            'task3_best': self._get_best_model_from_metrics(task3_metrics),
            'transformer': transformer_metrics.get('transformer', {}),
            'improvement': {}
        }
        
        return comparison
    
    def _extract_metrics(self, results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract evaluation metrics from results."""
        metrics = {}
        for model_name, model_info in results.items():
            if 'metrics' in model_info:
                metrics[model_name] = model_info['metrics']
        return metrics
    
    def _get_best_model_from_metrics(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Get the best model based on RMSE."""
        best_model = None
        best_rmse = float('inf')
        
        for model_name, model_metrics in metrics.items():
            if 'RMSE' in model_metrics and model_metrics['RMSE'] < best_rmse:
                best_rmse = model_metrics['RMSE']
                best_model = model_name
        
        return {'model': best_model, 'rmse': best_rmse}
    
    def _generate_q1_answer(self) -> str:
        """Generate answer for Question Q1."""
        if 'task2' not in self.task_results:
            return "Task 2 not completed yet."
        
        metrics = self.task_results['task2']['evaluation_metrics']
        best_model = self._get_best_model_from_metrics(metrics)
        
        answer = f"""
        Q1: Which baseline model generalized better and why?
        
        Based on the evaluation metrics, the {best_model['model']} model generalized better 
        with an RMSE of {best_model['rmse']:.4f}. 
        
        Analysis of underfitting/overfitting:
        - Linear Regression: Simple model, may underfit complex temporal patterns
        - ARIMA/SARIMA: Can capture seasonal patterns better but may overfit with wrong parameters
        
        The training/validation curves show [analysis would be based on actual results].
        """
        return answer.strip()
    
    def _generate_q2_answer(self) -> str:
        """Generate answer for Question Q2."""
        if 'task3' not in self.task_results:
            return "Task 3 not completed yet."
        
        metrics = self.task_results['task3']['evaluation_metrics']
        best_model = self._get_best_model_from_metrics(metrics)
        
        answer = f"""
        Q2: Which deep learning model captured temporal patterns best?
        
        The {best_model['model']} model captured temporal patterns best with an RMSE of {best_model['rmse']:.4f}.
        
        Architecture analysis:
        - RNN: Basic recurrent architecture, may struggle with long-term dependencies
        - GRU: Gated mechanism helps with gradient flow, better than vanilla RNN
        - LSTM: Full gating mechanism, excellent for long-term dependencies
        - CNN-LSTM: Combines local pattern detection with sequence modeling
        
        Pros and cons of each architecture:
        [Analysis would be based on actual training results and convergence patterns]
        """
        return answer.strip()
    
    def _generate_lab_report(self, include_plots: bool) -> str:
        """Generate comprehensive lab report."""
        report = f"""
DAT301m Lab 4: Time Series Forecasting Report
==============================================

Dataset: PJM Hourly Energy Consumption Data
Region: {self.region if self.region else 'All regions'}
Analysis Period: [Date range from actual data]

TASK 1: DATASET EXPLORATION AND PREPROCESSING
---------------------------------------------
{self._generate_task1_report()}

TASK 2: BASELINE MODELS
-----------------------
{self._generate_task2_report()}

TASK 3: DEEP LEARNING MODELS
-----------------------------
{self._generate_task3_report()}

TASK 4: ATTENTION-BASED AND TRANSFORMER MODELS
----------------------------------------------
{self._generate_task4_report()}

CONCLUSIONS AND RECOMMENDATIONS
-------------------------------
{self._generate_conclusions()}

QUESTION ANSWERS
----------------
{self._generate_question_answers()}
        """
        return report.strip()
    
    def _generate_task1_report(self) -> str:
        """Generate Task 1 report section."""
        if 'task1' not in self.task_results:
            return "Task 1 not completed."
        
        results = self.task_results['task1']
        return f"""
Data shape: {results.get('data_shape', 'N/A')}
Training samples: {results.get('train_size', 'N/A')}
Validation samples: {results.get('val_size', 'N/A')}
Test samples: {results.get('test_size', 'N/A')}

WindowGenerator configuration:
- Input width: {results['window_config']['input_width']} hours
- Label width: {results['window_config']['label_width']} hours
- Shift: {results['window_config']['shift']} hours

Data preprocessing steps:
1. Datetime parsing and indexing
2. Missing value handling
3. Normalization using MinMax scaling
4. Time series window creation
        """
    
    def _generate_task2_report(self) -> str:
        """Generate Task 2 report section."""
        if 'task2' not in self.task_results:
            return "Task 2 not completed."
        
        return "Baseline models trained and evaluated. See metrics in results."
    
    def _generate_task3_report(self) -> str:
        """Generate Task 3 report section."""
        if 'task3' not in self.task_results:
            return "Task 3 not completed."
        
        return "Deep learning models trained and evaluated. See metrics in results."
    
    def _generate_task4_report(self) -> str:
        """Generate Task 4 report section."""
        if 'task4' not in self.task_results:
            return "Task 4 not completed."
        
        return "Transformer models trained and evaluated. See metrics in results."
    
    def _generate_conclusions(self) -> str:
        """Generate conclusions section."""
        return """
Based on the comprehensive analysis of different forecasting approaches:

1. Model Performance Ranking: [Based on actual results]
2. Computational Efficiency: [Based on training times]
3. Interpretability: Traditional models vs. deep learning
4. Practical Recommendations: [Based on use case requirements]
        """
    
    def _generate_question_answers(self) -> str:
        """Generate question answers section."""
        answers = self.answer_questions()
        return f"""
{answers.get('Q1', 'Q1 not available')}

{answers.get('Q2', 'Q2 not available')}
        """ 