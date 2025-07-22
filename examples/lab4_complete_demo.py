#!/usr/bin/env python3
"""
DAT301m Lab 4: Complete Demo
=============================

Đây là demo hoàn chỉnh cho Lab 4 - Time Series Forecasting sử dụng PJM data.
Demo này bao gồm tất cả 4 tasks được yêu cầu trong lab.

Task 1: Data Exploration and Preprocessing (1.5 điểm)
Task 2: Baseline Models (3 điểm) 
Task 3: Deep Learning Models (4 điểm)
Task 4: Advanced Attention/Transformer Models (1.5 điểm)

Usage:
    python examples/lab4_complete_demo.py
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from time_series_forecasting.analysis.lab_interface.lab4_interface import Lab4Interface
from time_series_forecasting.analysis.lab_interface.task_executor import TaskExecutor
from time_series_forecasting.analysis.lab_interface.result_manager import ResultManager
from time_series_forecasting.core import DataLoader, WindowGenerator
from time_series_forecasting.analysis.visualization import create_report_plots

def main():
    """Main function to run complete Lab 4 demo."""
    
    print("=" * 80)
    print("DAT301m Lab 4: Time Series Forecasting - Complete Demo")
    print("=" * 80)
    
    # Configuration
    config = {
        'data_path': 'data/PJME_hourly.csv',  # Sử dụng PJME data
        'region': 'PJME',
        'target_col': 'PJME_MW',
        'datetime_col': 'Datetime',
        'input_width': 24,  # 24 giờ input
        'label_width': 1,   # Dự báo 1 giờ
        'shift': 1,
        'output_dir': 'lab4_results'
    }
    
    print(f"Sử dụng dữ liệu: {config['data_path']}")
    print(f"Region: {config['region']}")
    print(f"Input width: {config['input_width']} hours")
    print(f"Label width: {config['label_width']} hour")
    print()
    
    # ============================================================================
    # TASK 1: Data Exploration and Preprocessing
    # ============================================================================
    print("TASK 1: Data Exploration and Preprocessing")
    print("-" * 50)
    
    # Initialize Lab4Interface
    lab_interface = Lab4Interface()
    
    # Load and preprocess data
    print("1.1. Loading and preprocessing data...")
    data = lab_interface.load_data(
        config['data_path'], 
        region=config['region']
    )
    print(f"    ✓ Loaded data shape: {data.shape}")
    
    # Analyze data
    print("1.2. Analyzing data...")
    analysis_results = lab_interface.analyze_data(target_col=config['target_col'])
    print(f"    ✓ Basic statistics computed")
    print(f"    ✓ Seasonal patterns detected")
    print(f"    ✓ Anomalies detected: {len(analysis_results.get('anomalies', {}).get('anomaly_indices', []))}")
    
    # Create visualizations
    print("1.3. Creating visualizations...")
    plots = lab_interface.create_visualizations(
        target_col=config['target_col'],
        output_dir=os.path.join(config['output_dir'], 'plots')
    )
    print(f"    ✓ Created {len(plots)} visualization plots")
    
    # Setup WindowGenerator for tasks
    window_config = {
        'input_width': config['input_width'],
        'label_width': config['label_width'], 
        'shift': config['shift']
    }
    
    print("    ✓ Task 1 completed successfully!")
    print()
    
    # ============================================================================
    # TASK 2: Baseline Models
    # ============================================================================
    print("TASK 2: Baseline Models")
    print("-" * 50)
    
    # Define baseline model configurations
    baseline_models = [
        {
            'type': 'linear',
            'name': 'Linear_Regression',
            'config': {},
            'train_params': {'epochs': 100, 'patience': 10},
            'metrics': ['mae', 'rmse']
        },
        {
            'type': 'arima',
            'name': 'ARIMA',
            'config': {'order': (2, 1, 2)},
            'train_params': {},
            'metrics': ['mae', 'rmse']
        }
    ]
    
    print("2.1. Training baseline models...")
    task2_results = lab_interface.execute_task2(
        window_config=window_config,
        model_configs=baseline_models
    )
    
    print(f"    ✓ Trained {len(baseline_models)} baseline models")
    for model_name in task2_results.get('models', {}):
        metrics = task2_results['models'][model_name]['metrics']
        print(f"    ✓ {model_name}: MAE = {metrics.get('mae', 'N/A'):.4f}")
    
    print("    ✓ Task 2 completed successfully!")
    print()
    
    # ============================================================================
    # TASK 3: Deep Learning Models
    # ============================================================================
    print("TASK 3: Deep Learning Models")
    print("-" * 50)
    
    # Define deep learning model configurations
    deep_learning_models = [
        {
            'type': 'rnn',
            'name': 'Simple_RNN',
            'config': {'units': 64, 'layers': 2, 'dropout': 0.2},
            'train_params': {'epochs': 50, 'patience': 10},
            'metrics': ['mae', 'rmse']
        },
        {
            'type': 'gru',
            'name': 'GRU',
            'config': {'units': 64, 'layers': 2, 'dropout': 0.2},
            'train_params': {'epochs': 50, 'patience': 10},
            'metrics': ['mae', 'rmse']
        },
        {
            'type': 'lstm',
            'name': 'LSTM',
            'config': {'units': 64, 'layers': 2, 'dropout': 0.2},
            'train_params': {'epochs': 50, 'patience': 10},
            'metrics': ['mae', 'rmse']
        }
    ]
    
    print("3.1. Training deep learning models...")
    task3_results = lab_interface.execute_task3(
        window_config=window_config,
        model_configs=deep_learning_models
    )
    
    print(f"    ✓ Trained {len(deep_learning_models)} deep learning models")
    for model_name in task3_results.get('models', {}):
        metrics = task3_results['models'][model_name]['metrics']
        print(f"    ✓ {model_name}: MAE = {metrics.get('mae', 'N/A'):.4f}")
    
    print("    ✓ Task 3 completed successfully!")
    print()
    
    # ============================================================================
    # TASK 4: Transformer Models
    # ============================================================================
    print("TASK 4: Advanced Attention/Transformer Models")
    print("-" * 50)
    
    # Define transformer model configurations
    transformer_models = [
        {
            'type': 'transformer',
            'name': 'Transformer',
            'config': {
                'num_heads': 8,
                'd_model': 128,
                'num_layers': 4,
                'dropout': 0.1
            },
            'train_params': {'epochs': 50, 'patience': 10},
            'metrics': ['mae', 'rmse']
        }
    ]
    
    print("4.1. Training transformer models...")
    task4_results = lab_interface.execute_task4(
        window_config=window_config,
        model_configs=transformer_models
    )
    
    print(f"    ✓ Trained {len(transformer_models)} transformer models")
    for model_name in task4_results.get('models', {}):
        metrics = task4_results['models'][model_name]['metrics']
        print(f"    ✓ {model_name}: MAE = {metrics.get('mae', 'N/A'):.4f}")
    
    print("    ✓ Task 4 completed successfully!")
    print()
    
    # ============================================================================
    # RESULTS SUMMARY
    # ============================================================================
    print("RESULTS SUMMARY")
    print("-" * 50)
    
    # Collect all model results
    all_results = {}
    all_results.update(task2_results.get('models', {}))
    all_results.update(task3_results.get('models', {}))
    all_results.update(task4_results.get('models', {}))
    
    print("Model Performance Comparison:")
    print(f"{'Model':<20} {'Type':<15} {'MAE':<10} {'RMSE':<10}")
    print("-" * 60)
    
    for name, model_info in all_results.items():
        model_type = model_info['type']
        metrics = model_info['metrics']
        mae = metrics.get('mae', 'N/A')
        rmse = metrics.get('rmse', 'N/A')
        print(f"{name:<20} {model_type:<15} {mae:<10.4f} {rmse:<10.4f}")
    
    # Save results
    print("\nSaving results...")
    lab_interface.save_results(config['output_dir'])
    print(f"    ✓ Results saved to: {config['output_dir']}")
    
    # ============================================================================
    # LAB QUESTIONS ANSWERS
    # ============================================================================
    print("\nLAB QUESTIONS ANSWERS")
    print("-" * 50)
    
    print("Q1: Mô hình nào khái quát tốt hơn và tại sao?")
    print("    Dựa trên kết quả MAE và RMSE, mô hình với MAE thấp nhất thường khái quát tốt hơn.")
    print("    Linear Regression thường có độ phức tạp thấp, ít overfitting.")
    print("    ARIMA phù hợp với dữ liệu có tính seasonality mạnh.")
    print("    Cần kiểm tra validation curves để xác định overfitting.")
    print()
    
    print("Q2: Mô hình nào nắm bắt mẫu thời gian tốt nhất?")
    print("    LSTM thường tốt nhất cho long-term dependencies.")
    print("    GRU có hiệu suất tương tự LSTM nhưng đơn giản hơn.")
    print("    Transformer có thể tốt hơn với attention mechanism cho long sequences.")
    print("    RNN đơn giản có thể bị vanishing gradient với sequences dài.")
    print()
    
    print("=" * 80)
    print("LAB 4 COMPLETED SUCCESSFULLY!")
    print("Tất cả 4 tasks đã được hoàn thành:")
    print("✓ Task 1: Data Exploration (1.5 điểm)")
    print("✓ Task 2: Baseline Models (3 điểm)")
    print("✓ Task 3: Deep Learning (4 điểm)")
    print("✓ Task 4: Transformers (1.5 điểm)")
    print(f"✓ Results saved to: {config['output_dir']}")
    print("=" * 80)

if __name__ == "__main__":
    main() 