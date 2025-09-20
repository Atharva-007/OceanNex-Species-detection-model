"""Custom exceptions for fish species classifier."""


class FishClassifierError(Exception):
    """Base exception for fish classifier application."""
    pass


class ModelError(FishClassifierError):
    """Exception raised for model-related errors."""
    pass


class DataError(FishClassifierError):
    """Exception raised for data-related errors."""
    pass


class DatasetError(FishClassifierError):
    """Exception raised for dataset-related errors."""
    pass


class ValidationError(FishClassifierError):
    """Exception raised for validation-related errors."""
    pass


class TrainingError(FishClassifierError):
    """Exception raised for training-related errors."""
    pass


class ConfigurationError(FishClassifierError):
    """Exception raised for configuration-related errors."""
    pass


class ConfigError(FishClassifierError):
    """Exception raised for configuration-related errors."""
    pass


class PredictionError(FishClassifierError):
    """Exception raised for prediction-related errors."""
    pass


class ValidationError(FishClassifierError):
    """Exception raised for validation errors."""
    pass


class PreprocessingError(DataError):
    """Exception raised for preprocessing errors."""
    pass


class AugmentationError(DataError):
    """Exception raised for data augmentation errors."""
    pass


class TrainingError(ModelError):
    """Exception raised for training-related errors."""
    pass


class EvaluationError(ModelError):
    """Exception raised for evaluation-related errors."""
    pass


class UIError(FishClassifierError):
    """Exception raised for UI-related errors."""
    pass