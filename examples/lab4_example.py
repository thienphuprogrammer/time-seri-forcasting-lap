"""
Example script demonstrating how to use the DAT301m Lab 4 Interface
to complete the time series forecasting lab requirements.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from time_series_forecasting.analysis.lab4_interface import DAT301mLab4Interface

def main():
    # Initialize the lab interface
    # Replace 'path/to/your/data.csv' with the actual path to your PJM data
    lab = DAT301mLab4Interface(
        data_path='data/PJME_hourly.csv',  # Update this path
        region='PJME',  # Choose your region
        input_width=24,  # Use 24 hours as input
        label_width=1,   # Predict 1 hour ahead
        shift=1,         # 1 hour shift between input and labels
        random_seed=42   # For reproducibility
    )
    
    print("DAT301m Lab 4 Interface Example")
    print("="*50)
    
    # Option 1: Run complete lab workflow
    print("\nOption 1: Running complete lab workflow...")
    results = lab.run_complete_lab(
        output_dir='lab4_results/',
        save_plots=True,
        multi_step=True,
        create_ensemble=True
    )
    
    print("\n✓ Complete lab workflow finished!")
    print(f"Results saved to: lab4_results/")
    
    # Option 2: Run tasks individually (alternative approach)
    """
    print("\nOption 2: Running tasks individually...")
    
    # Task 1: Dataset exploration and preprocessing
    task1_results = lab.execute_task1(
        datetime_col='Datetime',
        target_col='PJME_MW',
        create_plots=True,
        save_plots=True
    )
    
    # Task 2: Baseline models
    task2_results = lab.execute_task2(
        epochs=50,
        patience=10,
        create_plots=True,
        save_plots=True
    )
    
    # Task 3: Deep learning models
    task3_results = lab.execute_task3(
        epochs=50,
        patience=10,
        units=64,
        layers=2,
        dropout=0.2,
        create_ensemble=True,
        multi_step=True,
        multi_step_horizon=24,
        create_plots=True,
        save_plots=True
    )
    
    # Task 4: Advanced attention/transformer models
    task4_results = lab.execute_task4(
        epochs=50,
        patience=10,
        num_heads=8,
        d_model=128,
        num_layers=4,
        multi_step_horizon=24,
        create_plots=True,
        save_plots=True
    )
    
    # Generate answers to questions
    answers = lab.answer_questions()
    print("\nQuestion Answers:")
    for question, answer in answers.items():
        print(f"\n{question}:")
        print(answer)
    
    # Generate comprehensive report
    report = lab.generate_comprehensive_report('lab4_report.txt')
    print("\nReport generated: lab4_report.txt")
    """

if __name__ == "__main__":
    main() 