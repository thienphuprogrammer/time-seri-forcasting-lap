"""
Base Pipeline Module for Time Series Forecasting
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd # type: ignore
import numpy as np # type: ignore
from datetime import datetime
import json

class BasePipeline(ABC):
    """
    Abstract base class for all pipelines.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or {}
        self.results: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the pipeline.
        
        Args:
            **kwargs: Pipeline arguments
            
        Returns:
            Pipeline results
        """
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """
        Validate pipeline configuration.
        
        Returns:
            True if configuration is valid
        """
        pass
    
    def save_results(self, path: str) -> None:
        """
        Save pipeline results to file.
        
        Args:
            path: Path to save results
        """
        save_dict = {
            'config': self.config,
            'results': self._make_serializable(self.results),
            'history': self.history,
            'execution_time': {
                'start': str(self.start_time) if self.start_time else None,
                'end': str(self.end_time) if self.end_time else None,
                'duration': str(self.end_time - self.start_time) if self.start_time and self.end_time else None
            }
        }
        
        with open(path, 'w') as f:
            json.dump(save_dict, f, indent=4)
        
        print(f"Results saved to {path}")
    
    def load_results(self, path: str) -> None:
        """
        Load pipeline results from file.
        
        Args:
            path: Path to load results from
        """
        with open(path, 'r') as f:
            load_dict = json.load(f)
        
        self.config = load_dict['config']
        self.results = load_dict['results']
        self.history = load_dict['history']
        
        # Parse execution time
        if load_dict['execution_time']['start']:
            self.start_time = datetime.fromisoformat(load_dict['execution_time']['start'])
        if load_dict['execution_time']['end']:
            self.end_time = datetime.fromisoformat(load_dict['execution_time']['end'])
        
        print(f"Results loaded from {path}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return self.config.copy()
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set pipeline configuration.
        
        Args:
            config: New configuration
        """
        self.config.update(config)
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON serializable format.
        
        Args:
            obj: Input object
            
        Returns:
            Serializable object
        """
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
        else:
            return obj
    
    def __repr__(self) -> str:
        """String representation."""
        status = 'completed' if self.results else 'not run'
        return f"{self.__class__.__name__}(status='{status}')" 