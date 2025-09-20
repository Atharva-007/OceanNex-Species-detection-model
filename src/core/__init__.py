"""Core package for fish species classifier."""

from .models import ModelManager, ModelTrainer
from .preprocessing import ImagePreprocessor, DataAugmentation
# from .training import TrainingPipeline, TrainingCallbacks
# from .evaluation import ModelEvaluator, MetricsCalculator
from .prediction import PredictionEngine, BatchPredictor

__all__ = [
    'ModelManager',
    'ModelTrainer',
    'ImagePreprocessor', 
    'DataAugmentation',
    'TrainingPipeline',
    'TrainingCallbacks',
    'ModelEvaluator',
    'MetricsCalculator',
    'PredictionEngine',
    'BatchPredictor'
]