"""
Data Validator Module for Time Series Data
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

class DataValidator:
    """
    Class for validating time series data quality and consistency.
    """
    
    def __init__(self):
        """Initialize DataValidator."""
        self.data = None
        self.validation_results = {}
        self.validation_history = []
    
    def validate_data(self, data: pd.DataFrame, checks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate data using specified checks.
        
        Args:
            data: Input DataFrame
            checks: List of checks to perform (if None, performs all checks)
            
        Returns:
            Dictionary containing validation results
        """
        self.data = data.copy()
        
        if checks is None:
            checks = self._available_checks()
        
        # Run all specified checks
        for check in checks:
            if check in self._available_checks():
                check_func = getattr(self, f"_check_{check}")
                result = check_func()
                self.validation_results[check] = result
                
                self.validation_history.append({
                    'check': check,
                    'timestamp': datetime.now(),
                    'result': result
                })
        
        return self.validation_results
    
    def _check_missing_values(self) -> Dict[str, Any]:
        """Check for missing values."""
        missing = self.data.isnull().sum()
        total_missing = missing.sum()
        
        return {
            'has_missing': total_missing > 0,
            'total_missing': total_missing,
            'missing_by_column': missing.to_dict(),
            'missing_percentage': (total_missing / len(self.data) * 100)
        }
    
    def _check_duplicates(self) -> Dict[str, Any]:
        """Check for duplicate records."""
        duplicates = self.data.index.duplicated()
        total_duplicates = duplicates.sum()
        
        return {
            'has_duplicates': total_duplicates > 0,
            'total_duplicates': total_duplicates,
            'duplicate_indices': self.data.index[duplicates].tolist()
        }
    
    def _check_time_continuity(self) -> Dict[str, Any]:
        """Check for gaps in time series."""
        if not isinstance(self.data.index, pd.DatetimeIndex):
            return {'error': 'Index is not datetime'}
        
        # Calculate expected frequency
        time_diff = self.data.index[1] - self.data.index[0]
        gaps = []
        
        for i in range(1, len(self.data)):
            current_diff = self.data.index[i] - self.data.index[i-1]
            if current_diff > time_diff:
                gaps.append({
                    'start': self.data.index[i-1],
                    'end': self.data.index[i],
                    'duration': current_diff
                })
        
        return {
            'has_gaps': len(gaps) > 0,
            'total_gaps': len(gaps),
            'gaps': gaps,
            'frequency': str(time_diff)
        }
    
    def _check_outliers(self, threshold: float = 3.0) -> Dict[str, Any]:
        """Check for outliers using z-score method."""
        outliers = {}
        total_outliers = 0
        
        for col in self.data.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
            col_outliers = z_scores > threshold
            outliers[col] = {
                'count': col_outliers.sum(),
                'indices': self.data.index[col_outliers].tolist(),
                'values': self.data.loc[col_outliers, col].tolist()
            }
            total_outliers += col_outliers.sum()
        
        return {
            'has_outliers': total_outliers > 0,
            'total_outliers': total_outliers,
            'outliers_by_column': outliers,
            'threshold': threshold
        }
    
    def _check_data_types(self) -> Dict[str, Any]:
        """Check data types of columns."""
        dtypes = self.data.dtypes.to_dict()
        type_counts = self.data.dtypes.value_counts().to_dict()
        
        return {
            'column_types': {col: str(dtype) for col, dtype in dtypes.items()},
            'type_summary': {str(dtype): count for dtype, count in type_counts.items()}
        }
    
    def _check_value_ranges(self) -> Dict[str, Any]:
        """Check value ranges for numeric columns."""
        ranges = {}
        
        for col in self.data.select_dtypes(include=[np.number]).columns:
            ranges[col] = {
                'min': float(self.data[col].min()),
                'max': float(self.data[col].max()),
                'mean': float(self.data[col].mean()),
                'std': float(self.data[col].std())
            }
        
        return {
            'column_ranges': ranges
        }
    
    def _available_checks(self) -> List[str]:
        """Get list of available validation checks."""
        return [
            'missing_values',
            'duplicates',
            'time_continuity',
            'outliers',
            'data_types',
            'value_ranges'
        ]
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of validation results.
        
        Returns:
            Dictionary containing validation summary
        """
        if not self.validation_results:
            return {'error': 'No validation performed yet'}
        
        issues = []
        
        # Check for issues
        if self.validation_results.get('missing_values', {}).get('has_missing', False):
            issues.append('Has missing values')
        if self.validation_results.get('duplicates', {}).get('has_duplicates', False):
            issues.append('Has duplicate records')
        if self.validation_results.get('time_continuity', {}).get('has_gaps', False):
            issues.append('Has time gaps')
        if self.validation_results.get('outliers', {}).get('has_outliers', False):
            issues.append('Has outliers')
        
        return {
            'data_shape': self.data.shape if self.data is not None else None,
            'checks_performed': list(self.validation_results.keys()),
            'issues_found': issues,
            'validation_history': self.validation_history
        }
    
    def __repr__(self) -> str:
        """String representation of DataValidator."""
        status = 'validated' if self.validation_results else 'not validated'
        return f"DataValidator(status='{status}', checks={list(self.validation_results.keys()) if self.validation_results else None})" 