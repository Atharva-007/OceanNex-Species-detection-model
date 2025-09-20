"""
Training Package for Fish Species Classification
===============================================

This package contains all training-related modules including training managers,
experiment tracking, and training utilities for fish species classification.
"""

from .training_manager import TrainingManager
from .experiment_tracker import ExperimentTracker
from .training_config import TrainingConfig

__all__ = [
    'TrainingManager',
    'ExperimentTracker', 
    'TrainingConfig'
]

# Version info
__version__ = "1.0.0"
__author__ = "OceanNex Development Team"
__description__ = "Training pipeline for fish species classification"