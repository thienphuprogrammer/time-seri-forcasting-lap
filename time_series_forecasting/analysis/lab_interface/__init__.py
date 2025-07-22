"""
Lab Interface Module
"""

from .lab4_interface import Lab4Interface
from .task_executor import TaskExecutor
from .result_manager import ResultManager

# Backward compatibility alias
DAT301mLab4Interface = Lab4Interface

__all__ = [
    'Lab4Interface',
    'DAT301mLab4Interface',
    'TaskExecutor',
    'ResultManager'
] 