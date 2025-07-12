"""
Example usage script for the Time Series Forecasting Pipeline.

This script demonstrates how to use the ForecastingPipeline class to:
1. Load and preprocess energy consumption data
2. Train baseline models (Linear Regression, ARIMA, Random Forest)
3. Train deep learning models (RNN, GRU, LSTM, CNN-LSTM)
4. Train transformer models
5. Create ensemble models
6. Generate comprehensive reports and visualizations

Author: Your Name
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from forecasting_pipeline import ForecastingPipeline
import warnings
warnings.filterwarnings('ignore')

def create_sample_data(n_samples=1000, save_path='sample_energy_data.csv'):
    """
    Create sample energy consumption data for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        save_path: Path to save the sample data
        
    Returns:
        Path to the created data file
    """
    print("Creating sample energy consumption data...")
    
    # Generate datetime index
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='H')
    
    # Create synthetic energy consumption data with trends and seasonality
    np.random.seed(42)
    
    # Base trend (increasing over time)
    trend = np.linspace(100, 150, n_samples)
    
    # Daily seasonality (24-hour cycle)
    daily_pattern = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
    
    # Weekly seasonality (168-hour cycle)
    weekly_pattern = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 168)
    
    # Monthly seasonality (730-hour cycle)
    monthly_pattern = 15 * np.sin(2 * np.pi * np.arange(n_samples) / 730)
    
    # Random noise
    noise = np.random.normal(0, 5, n_samples)
    
    # Combine all components
    energy_consumption = trend + daily_pattern + weekly_pattern + monthly_pattern + noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'Datetime': dates,
        'Energy_MW': energy_consumption,
        'Temperature': np.random.normal(20, 10, n_samples),  # Additional feature
        'Humidity': np.random.uniform(30, 80, n_samples)     # Additional feature
    })
    
    # Save to CSV
    data.to_csv(save_path, index=False)
    print(f"Sample data saved to {save_path}")
    
    return save_path

def example_single_step_forecasting():
    """
    Example of single-step forecasting (predicting next hour).
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: SINGLE-STEP FORECASTING")
    print("="*80)
    
    # Create sample data
    data_path = create_sample_data(n_samples=2000)
    
    # Initialize pipeline for single-step forecasting
    pipeline = ForecastingPipeline(
        data_path=data_path,
        region=None,  # No region filtering for sample data
        input_width=24,   # Use last 24 hours
        label_width=1,    # Predict next 1 hour
        shift=1           # Shift by 1 hour
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        save_path='results_single_step',
        train_ensemble=True,
        epochs=50,  # Reduced for faster execution
        patience=5
    )
    
    print("\nSingle-step forecasting completed!")
    return results

def example_multi_step_forecasting():
    """
    Example of multi-step forecasting (predicting next 24 hours).
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: MULTI-STEP FORECASTING")
    print("="*80)
    
    # Create sample data
    data_path = create_sample_data(n_samples=3000)
    
    # Initialize pipeline for multi-step forecasting
    pipeline = ForecastingPipeline(
        data_path=data_path,
        region=None,
        input_width=48,   # Use last 48 hours
        label_width=24,   # Predict next 24 hours
        shift=1           # Shift by 1 hour
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        save_path='results_multi_step',
        train_ensemble=True,
        epochs=50,
        patience=5
    )
    
    print("\nMulti-step forecasting completed!")
    return results

def example_custom_configuration():
    """
    Example with custom configuration and specific model selection.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: CUSTOM CONFIGURATION")
    print("="*80)
    
    # Create sample data
    data_path = create_sample_data(n_samples=1500)
    
    # Initialize pipeline with custom settings
    pipeline = ForecastingPipeline(
        data_path=data_path,
        region=None,
        input_width=12,   # Use last 12 hours
        label_width=6,    # Predict next 6 hours
        shift=1
    )
    
    # Step 1: Load and preprocess data
    processed_data = pipeline.load_and_preprocess_data(
        datetime_col='Datetime',
        datetime_format='%Y-%m-%d %H:%M:%S',
        missing_method='interpolate',
        normalize_method='standard',  # Use standard scaling
        target_column='Energy_MW'
    )
    
    # Step 2: Create visualizations
    plots = pipeline.create_visualizations(save_path='custom_plots')
    
    # Step 3: Prepare datasets
    train_dataset, val_dataset, test_dataset = pipeline.prepare_datasets(
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        batch_size=64
    )
    
    # Step 4: Train only specific models
    print("\nTraining only LSTM and Transformer models...")
    
    # Train LSTM
    lstm_results = pipeline.train_deep_learning_models(
        train_dataset, val_dataset, test_dataset,
        epochs=30,
        patience=5
    )
    
    # Train Transformer
    transformer_results = pipeline.train_transformer_model(
        train_dataset, val_dataset, test_dataset,
        epochs=30,
        patience=5
    )
    
    # Step 5: Create ensemble of only these two models
    ensemble = pipeline.create_ensemble_model(
        model_names=['lstm', 'transformer'],
        method='average'
    )
    
    # Step 6: Generate report
    report = pipeline.generate_comprehensive_report(save_path='custom_results')
    
    print("\nCustom configuration completed!")
    return {
        'pipeline': pipeline,
        'report': report
    }

def example_with_real_data_structure():
    """
    Example showing how to use the pipeline with real PJM energy data structure.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: REAL DATA STRUCTURE (PJM ENERGY DATA)")
    print("="*80)
    
    # This example shows the expected structure for real PJM data
    # You would replace this with actual data loading
    
    print("""
    For real PJM energy consumption data, the CSV file should have columns like:
    - Datetime: Timestamp of the measurement
    - PJM_Load_MW: Energy consumption in megawatts
    - Region: Region identifier (optional)
    
    Example usage with real data:
    
    # Initialize pipeline
    pipeline = ForecastingPipeline(
        data_path='path/to/your/pjm_data.csv',
        region='PJM_Load_hourly',  # Specific region if needed
        input_width=24,
        label_width=1,
        shift=1
    )
    
    # Run pipeline
    results = pipeline.run_complete_pipeline(
        save_path='pjm_results',
        epochs=100,
        patience=15
    )
    """)
    
    # Create a more realistic sample data structure
    dates = pd.date_range('2020-01-01', periods=1000, freq='H')
    
    # Simulate PJM load data
    base_load = 30000  # Base load in MW
    daily_variation = 5000 * np.sin(2 * np.pi * np.arange(1000) / 24)
    weekly_variation = 2000 * np.sin(2 * np.pi * np.arange(1000) / 168)
    trend = np.linspace(0, 1000, 1000)
    noise = np.random.normal(0, 500, 1000)
    
    pjm_load = base_load + daily_variation + weekly_variation + trend + noise
    
    pjm_data = pd.DataFrame({
        'Datetime': dates,
        'PJM_Load_MW': pjm_load,
        'Region': ['PJM_Load_hourly'] * 1000
    })
    
    pjm_data.to_csv('sample_pjm_data.csv', index=False)
    
    # Initialize pipeline for PJM data
    pipeline = ForecastingPipeline(
        data_path='sample_pjm_data.csv',
        region='PJM_Load_hourly',
        input_width=24,
        label_width=1,
        shift=1
    )
    
    # Run with PJM data
    results = pipeline.run_complete_pipeline(
        save_path='pjm_sample_results',
        epochs=30,  # Reduced for demonstration
        patience=5
    )
    
    print("\nPJM data structure example completed!")
    return results

def main():
    """
    Main function to run all examples.
    """
    print("TIME SERIES FORECASTING PIPELINE - EXAMPLE USAGE")
    print("="*80)
    
    try:
        # Example 1: Single-step forecasting
        results1 = example_single_step_forecasting()
        
        # Example 2: Multi-step forecasting
        results2 = example_multi_step_forecasting()
        
        # Example 3: Custom configuration
        results3 = example_custom_configuration()
        
        # Example 4: Real data structure
        results4 = example_with_real_data_structure()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print("\nGenerated files:")
        print("- sample_energy_data.csv: Sample energy consumption data")
        print("- sample_pjm_data.csv: Sample PJM-style data")
        print("- results_single_step_*: Single-step forecasting results")
        print("- results_multi_step_*: Multi-step forecasting results")
        print("- custom_*: Custom configuration results")
        print("- pjm_sample_results_*: PJM data structure results")
        
        print("\nTo use with your own data:")
        print("1. Prepare your CSV file with datetime and energy consumption columns")
        print("2. Modify the data_path in the pipeline initialization")
        print("3. Adjust input_width, label_width, and shift as needed")
        print("4. Run the pipeline with your desired configuration")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 