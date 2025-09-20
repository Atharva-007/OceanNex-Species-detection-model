"""
Evaluation Package for Fish Species Classification
=================================================

This package contains comprehensive model evaluation tools including
metrics calculation, model comparison, and performance analysis.
"""

from .evaluator import ModelEvaluator
from .metrics import MetricsCalculator
from .comparison import ModelComparison
from .performance_analyzer import PerformanceAnalyzer

__all__ = [
    'ModelEvaluator',
    'MetricsCalculator',
    'ModelComparison', 
    'PerformanceAnalyzer'
]

# Version info
__version__ = "1.0.0"
__author__ = "OceanNex Development Team"
__description__ = "Comprehensive evaluation framework for fish species classification"