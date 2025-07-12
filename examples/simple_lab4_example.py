"""
Simple LAB 4 Example - Alternative Implementation

This script provides a simplified approach to complete DAT301m Lab 4
using the time series forecasting package components directly.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add the package to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from time_series_forecasting.core.data_processor import DataProcessor
from time_series_forecasting.core.window_generator import WindowGenerator
from time_series_forecasting.models.model_factory import ModelFactory
from time_series_forecasting.models.model_trainer import ModelTrainer

def main():
    """Main function to run the simplified Lab 4 example."""
    
    print("="*80)
    print("DAT301m Lab 4 - Simple Example")
    print("="*80)
    
    # Configuration
    DATA_PATH = 'data/PJME_hourly.csv'
    REGION = 'PJME'
    INPUT_WIDTH = 24
    LABEL_WIDTH = 1
    SHIFT = 1
    
    print(f"Configuration:")
    print(f"  Data: {DATA_PATH}")
    print(f"  Region: {REGION}")
    print(f"  Input Width: {INPUT_WIDTH}")
    print(f"  Label Width: {LABEL_WIDTH}")
    print(f"  Shift: {SHIFT}")
    
    # ========================================
    # TASK 1: Data Exploration and Preprocessing
    # ========================================
    
    print("\n" + "="*80)
    print("TASK 1: DATA EXPLORATION AND PREPROCESSING")
    print("="*80)
    
    # Initialize data processor
    processor = DataProcessor(DATA_PATH, REGION)
    
    # Load data
    print("\n1.1 Loading data...")
    raw_data = processor.load_data()
    print(f"✓ Data loaded: {raw_data.shape}")
    
    # Parse datetime
    print("\n1.2 Parsing datetime...")
    processor.parse_datetime()
    
    # Handle missing data
    print("\n1.3 Handling missing data...")
    processor.handle_missing_data(method='interpolate')
    
    # Normalize data
    print("\n1.4 Normalizing data...")
    processed_data = processor.normalize_data(method='minmax')
    print(f"✓ Data normalized: {processed_data.shape}")
    
    # Create visualizations
    print("\n1.5 Creating visualizations...")
    try:
        plots = processor.create_visualizations(save_path='plots/task1')
        print("✓ Visualizations created")
    except Exception as e:
        print(f"⚠ Visualization error: {e}")
        print("Continuing without visualizations...")
    
    # Prepare data for forecasting
    forecast_data = processor.prepare_for_forecasting()
    print(f"✓ Data prepared for forecasting: {forecast_data.shape}")
    
    # ========================================
    # TASK 2: Baseline Models
    # ========================================
    
    print("\n" + "="*80)
    print("TASK 2: BASELINE MODELS")
    print("="*80)
    
    # Initialize window generator
    window_gen = WindowGenerator(
        input_width=INPUT_WIDTH,
        label_width=LABEL_WIDTH,
        shift=SHIFT
    )
    
    # Split data
    train_df, val_df, test_df = window_gen.split_data(forecast_data)
    
    # Create datasets
    train_dataset = window_gen.get_train_dataset(batch_size=32)
    val_dataset = window_gen.get_val_dataset(batch_size=32)
    test_dataset = window_gen.get_test_dataset(batch_size=32)
    
    print(f"✓ Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize model factory and trainer
    model_factory = ModelFactory()
    trainer = ModelTrainer(window_gen)
    
    # Get example inputs for model shapes
    example_inputs, example_labels = window_gen.get_example_batch()
    input_shape = example_inputs.shape[1:]
    
    print(f"✓ Input shape: {input_shape}")
    
    # Convert datasets to numpy for sklearn models
    def dataset_to_numpy(dataset):
        X, y = [], []
        for batch_x, batch_y in dataset:
            X.append(batch_x.numpy())
            y.append(batch_y.numpy())
        return np.concatenate(X), np.concatenate(y)
    
    X_train, y_train = dataset_to_numpy(train_dataset)
    X_val, y_val = dataset_to_numpy(val_dataset)
    X_test, y_test = dataset_to_numpy(test_dataset)
    
    print(f"✓ Datasets converted to numpy")
    
    # Train Linear Regression
    print("\n2.1 Training Linear Regression...")
    lr_model = model_factory.create_linear_regression()
    trainer.train_sklearn_model(lr_model, 'linear_regression', X_train, y_train)
    lr_metrics = trainer.evaluate_model(lr_model, 'linear_regression', X_test, y_test, processor.scaler)
    print(f"✓ Linear Regression - RMSE: {lr_metrics['RMSE']:.4f}")
    
    # ========================================
    # TASK 3: Deep Learning Models
    # ========================================
    
    print("\n" + "="*80)
    print("TASK 3: DEEP LEARNING MODELS")
    print("="*80)
    
    # Train LSTM
    print("\n3.1 Training LSTM...")
    lstm_model = model_factory.create_lstm(
        input_shape=input_shape,
        units=64,
        layers=2,
        dropout=0.2
    )
    
    trainer.train_tensorflow_model(
        lstm_model, 'lstm',
        train_dataset, val_dataset,
        epochs=50, patience=5
    )
    
    # Evaluate LSTM
    lstm_metrics = trainer.evaluate_model(lstm_model, 'lstm', X_test, y_test, processor.scaler)
    print(f"✓ LSTM - RMSE: {lstm_metrics['RMSE']:.4f}")
    
    # Train GRU
    print("\n3.2 Training GRU...")
    gru_model = model_factory.create_gru(
        input_shape=input_shape,
        units=64,
        layers=2,
        dropout=0.2
    )
    
    trainer.train_tensorflow_model(
        gru_model, 'gru',
        train_dataset, val_dataset,
        epochs=50, patience=5
    )
    
    # Evaluate GRU
    gru_metrics = trainer.evaluate_model(gru_model, 'gru', X_test, y_test, processor.scaler)
    print(f"✓ GRU - RMSE: {gru_metrics['RMSE']:.4f}")
    
    # ========================================
    # TASK 4: Transformer Model
    # ========================================
    
    print("\n" + "="*80)
    print("TASK 4: TRANSFORMER MODEL")
    print("="*80)
    
    # Train Transformer
    print("\n4.1 Training Transformer...")
    transformer_model = model_factory.create_transformer(
        input_shape=input_shape,
        num_heads=8,
        d_model=64,
        num_layers=2,
        dropout=0.1
    )
    
    trainer.train_tensorflow_model(
        transformer_model, 'transformer',
        train_dataset, val_dataset,
        epochs=50, patience=5
    )
    
    # Evaluate Transformer
    transformer_metrics = trainer.evaluate_model(transformer_model, 'transformer', X_test, y_test, processor.scaler)
    print(f"✓ Transformer - RMSE: {transformer_metrics['RMSE']:.4f}")
    
    # ========================================
    # RESULTS SUMMARY
    # ========================================
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    all_metrics = {
        'Linear Regression': lr_metrics,
        'LSTM': lstm_metrics,
        'GRU': gru_metrics,
        'Transformer': transformer_metrics
    }
    
    print("\nModel Performance Comparison:")
    print("-" * 70)
    print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'MAPE':<10}")
    print("-" * 70)
    
    for model_name, metrics in all_metrics.items():
        print(f"{model_name:<20} {metrics['RMSE']:<10.4f} {metrics['MAE']:<10.4f} "
              f"{metrics['R2']:<10.4f} {metrics['MAPE']:<10.2f}")
    
    # Find best model
    best_model = min(all_metrics.items(), key=lambda x: x[1]['RMSE'])
    print(f"\nBest Model: {best_model[0]} (RMSE: {best_model[1]['RMSE']:.4f})")
    
    # ========================================
    # QUESTION ANSWERS
    # ========================================
    
    print("\n" + "="*80)
    print("QUESTION ANSWERS")
    print("="*80)
    
    print("\nQ1: Which generalized model is better and why?")
    print("-" * 50)
    print("Based on the results, the best baseline model is Linear Regression with RMSE of "
          f"{lr_metrics['RMSE']:.4f}. This model performs well for this time series due to "
          "the relatively stable patterns in energy consumption data.")
    
    print("\nQ2: Which model captured temporal patterns best?")
    print("-" * 50)
    print(f"The {best_model[0]} model captured temporal patterns best with an RMSE of "
          f"{best_model[1]['RMSE']:.4f}. This model's architecture is well-suited for "
          "capturing the complex temporal dependencies in energy consumption data.")
    
    print("\n" + "="*80)
    print("LAB 4 COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("✓ All tasks completed")
    print("✓ All models trained and evaluated")
    print("✓ Questions answered")
    print("✓ Results summarized")
    print("\nNext steps:")
    print("1. Review the results above")
    print("2. Create additional visualizations if needed")
    print("3. Write your lab report based on these results")
    print("4. Submit the report and code")

if __name__ == "__main__":
    main() 