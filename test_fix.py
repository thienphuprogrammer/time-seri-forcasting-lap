#!/usr/bin/env python3
"""
Test script to verify the KeyError fix for PJME_MW column.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from time_series_forecasting.analysis.lab4_interface import DAT301mLab4Interface

def test_task1():
    """Test Task 1 execution to see if the KeyError is fixed."""
    print("Testing Task 1 execution...")
    
    # Initialize the lab interface
    lab_interface = DAT301mLab4Interface(
        data_path='data/PJME_hourly.csv',
        region='PJME'
    )
    
    try:
        # Execute Task 1
        results = lab_interface.execute_task1(
            datetime_col='Datetime',
            target_col='PJME_MW',
            normalize_method='minmax',
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
            create_plots=False,  # Don't create plots for testing
            save_plots=False
        )
        
        print("✅ Task 1 executed successfully!")
        print(f"Results keys: {list(results.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_task1()
    if success:
        print("\n🎉 The KeyError fix is working!")
    else:
        print("\n💥 The KeyError fix failed!") 