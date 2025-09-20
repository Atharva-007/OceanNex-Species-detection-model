"""
Data management package for fish species classification.

This package provides comprehensive dataset management functionality including
dataset loading, analysis, validation, and preparation for training.
"""

from .dataset_manager import DatasetManager, DatasetSplit
from .dataset_loader import DatasetLoader
from .dataset_analyzer import DatasetAnalyzer

__all__ = [
    'DatasetManager',
    'DatasetSplit', 
    'DatasetLoader',
    'DatasetAnalyzer'
]