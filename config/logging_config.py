"""Logging configuration for the fish species classifier."""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    logs_dir: Optional[Path] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    format_string: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Specific log file name
        logs_dir: Directory for log files
        enable_console: Enable console logging
        enable_file: Enable file logging
        format_string: Custom format string
    """
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logs directory if needed
    if enable_file and logs_dir:
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            # Generate timestamped log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"fish_classifier_{timestamp}.log"
        
        log_file_path = logs_dir / log_file
    else:
        log_file_path = None
    
    # Configure handlers
    handlers = []
    
    # Console handler
    if enable_console:
        console_handler = {
            'class': 'logging.StreamHandler',
            'level': log_level,
            'formatter': 'detailed',
            'stream': 'ext://sys.stdout'
        }
        handlers.append('console')
    
    # File handler
    if enable_file and log_file_path:
        file_handler = {
            'class': 'logging.FileHandler',
            'level': log_level,
            'formatter': 'detailed',
            'filename': str(log_file_path),
            'mode': 'a'
        }
        handlers.append('file')
    
    # Logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': format_string,
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            }
        },
        'handlers': {},
        'loggers': {
            'fish_classifier': {
                'level': log_level,
                'handlers': handlers,
                'propagate': False
            },
            'tensorflow': {
                'level': 'WARNING',  # Reduce TensorFlow verbosity
                'handlers': handlers,
                'propagate': False
            },
            'PIL': {
                'level': 'WARNING',  # Reduce PIL verbosity
                'handlers': handlers,
                'propagate': False
            }
        },
        'root': {
            'level': log_level,
            'handlers': handlers
        }
    }
    
    # Add handlers to config
    if enable_console:
        config['handlers']['console'] = {
            'class': 'logging.StreamHandler',
            'level': log_level,
            'formatter': 'detailed',
            'stream': 'ext://sys.stdout'
        }
    
    if enable_file and log_file_path:
        config['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'level': log_level,
            'formatter': 'detailed',
            'filename': str(log_file_path),
            'mode': 'a'
        }
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Log configuration info
    logger = logging.getLogger('fish_classifier.config')
    logger.info(f"Logging configured - Level: {log_level}")
    if log_file_path:
        logger.info(f"Log file: {log_file_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"fish_classifier.{name}")


def log_exception(logger: logging.Logger, exception: Exception, message: str = "An error occurred") -> None:
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        exception: Exception to log
        message: Additional message
    """
    logger.error(f"{message}: {str(exception)}", exc_info=True)


def log_model_info(logger: logging.Logger, model, model_name: str = "Model") -> None:
    """
    Log model information.
    
    Args:
        logger: Logger instance
        model: Keras model
        model_name: Model name
    """
    try:
        total_params = model.count_params()
        trainable_params = sum([layer.count_params() for layer in model.layers if layer.trainable])
        non_trainable_params = total_params - trainable_params
        
        logger.info(f"{model_name} Information:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Non-trainable parameters: {non_trainable_params:,}")
        logger.info(f"  Input shape: {model.input_shape}")
        logger.info(f"  Output shape: {model.output_shape}")
        
    except Exception as e:
        logger.warning(f"Could not log model info: {e}")


def log_training_config(logger: logging.Logger, config: dict) -> None:
    """
    Log training configuration.
    
    Args:
        logger: Logger instance
        config: Training configuration dictionary
    """
    logger.info("Training Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")


def log_dataset_info(logger: logging.Logger, dataset_info: dict) -> None:
    """
    Log dataset information.
    
    Args:
        logger: Logger instance
        dataset_info: Dataset information dictionary
    """
    logger.info("Dataset Information:")
    for key, value in dataset_info.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")


class PerformanceLogger:
    """Logger for performance metrics and training progress."""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = get_logger(logger_name)
        self.metrics_history = []
    
    def log_epoch(self, epoch: int, metrics: dict) -> None:
        """Log epoch metrics."""
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch}: {metrics_str}")
        
        # Store metrics
        epoch_data = {"epoch": epoch, **metrics}
        self.metrics_history.append(epoch_data)
    
    def log_training_start(self, total_epochs: int, total_samples: int) -> None:
        """Log training start information."""
        self.logger.info(f"Starting training: {total_epochs} epochs, {total_samples} samples")
    
    def log_training_end(self, total_time: float, best_metrics: dict) -> None:
        """Log training completion information."""
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Best metrics: {best_metrics}")
    
    def save_metrics_history(self, file_path: Path) -> None:
        """Save metrics history to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            self.logger.info(f"Metrics history saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics history: {e}")


# Configure TensorFlow logging
def configure_tensorflow_logging():
    """Configure TensorFlow logging to reduce verbosity."""
    import os
    
    # Set TensorFlow log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING
    
    # Configure TensorFlow logger
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.setLevel(logging.ERROR)


# Initialize performance logger
performance_logger = PerformanceLogger()