"""
Result Manager Module for Lab Interface
"""

from typing import Dict, Any, Optional, List
import json
from pathlib import Path
from datetime import datetime

class ResultManager:
    """
    Class for managing lab results.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ResultManager.
        
        Args:
            config: Result manager configuration
        """
        self.config = config or {}
        
        # Store results
        self.analysis_results: Dict[str, Any] = {}
        self.task_results: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
    
    def save_analysis_results(self, results: Dict[str, Any], path: str) -> None:
        """
        Save analysis results to file.
        
        Args:
            results: Analysis results
            path: Path to save results
        """
        path = Path(path) # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Store results
        self.analysis_results = results
        self.history.append({
            'action': 'save_analysis',
            'timestamp': datetime.now(),
            'path': str(path)
        })
        
        # Save to file
        with open(path, 'w') as f:
            json.dump(self._make_serializable(results), f, indent=4)
    
    def load_analysis_results(self, path: str) -> Dict[str, Any]:
        """
        Load analysis results from file.
        
        Args:
            path: Path to load results from
            
        Returns:
            Analysis results
        """
        path = Path(path) # type: ignore
        
        # Load from file
        with open(path, 'r') as f:
            results = json.load(f)
        
        # Store results
        self.analysis_results = results
        self.history.append({
            'action': 'load_analysis',
            'timestamp': datetime.now(),
            'path': str(path)
        })
        
        return results
    
    def save_task_results(self, results: Dict[str, Any], path: str) -> None:
        """
        Save task results to file.
        
        Args:
            results: Task results
            path: Path to save results
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Store results
        self.task_results = results
        self.history.append({
            'action': 'save_task',
            'timestamp': datetime.now(),
            'path': str(path)
        })
        
        # Save to file
        with open(path, 'w') as f:
            json.dump(self._make_serializable(results), f, indent=4)
    
    def load_task_results(self, path: str) -> Dict[str, Any]:
        """
        Load task results from file.
        
        Args:
            path: Path to load results from
            
        Returns:
            Task results
        """
        path = Path(path)
        
        # Load from file
        with open(path, 'r') as f:
            results = json.load(f)
        
        # Store results
        self.task_results = results
        self.history.append({
            'action': 'load_task',
            'timestamp': datetime.now(),
            'path': str(path)
        })
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all results.
        
        Returns:
            Summary dictionary
        """
        return {
            'analysis_results': {
                key: self._get_result_summary(results)
                for key, results in self.analysis_results.items()
            } if self.analysis_results else None,
            'task_results': {
                key: self._get_result_summary(results)
                for key, results in self.task_results.items()
            } if self.task_results else None,
            'history': self.history
        }
    
    def _get_result_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary of a result dictionary.
        
        Args:
            results: Result dictionary
            
        Returns:
            Summary dictionary
        """
        if not isinstance(results, dict):
            return {'type': str(type(results)), 'value': str(results)}
        
        summary = {}
        
        # Get basic info
        summary['keys'] = list(results.keys())
        summary['size'] = len(results)
        
        # Get metrics if available
        if 'metrics' in results:
            summary['metrics'] = results['metrics']
        
        # Get shape if available
        if 'shape' in results:
            summary['shape'] = results['shape']
        
        return summary
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON serializable format.
        
        Args:
            obj: Input object
            
        Returns:
            Serializable object
        """
        import numpy as np
        import pandas as pd
        
        if isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        elif isinstance(obj, set):
            return list(self._make_serializable(item) for item in obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ResultManager(analysis_results={bool(self.analysis_results)}, task_results={bool(self.task_results)})" 