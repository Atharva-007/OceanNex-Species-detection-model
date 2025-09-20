"""
Training module for fish species classification models.

This module provides comprehensive training functionality including
model training, validation, callbacks, and training history management.
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List, Tuple, Callable
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger, Callback
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.metrics import Metric

from ..utils.logger import get_logger
from ..utils.exceptions import TrainingError, ConfigurationError
from config.settings import get_settings


class TrainingMetrics:
    """Container for training metrics and history."""
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {}
        self.best_val_accuracy: float = 0.0
        self.best_val_loss: float = float('inf')
        self.training_time: float = 0.0
        self.total_epochs: int = 0
        
    def update_from_history(self, history: tf.keras.callbacks.History):
        """Update metrics from Keras training history."""
        self.history = history.history
        if 'val_accuracy' in self.history:
            self.best_val_accuracy = max(self.history['val_accuracy'])
        if 'val_loss' in self.history:
            self.best_val_loss = min(self.history['val_loss'])
        self.total_epochs = len(self.history.get('loss', []))


class ModelTrainer:
    """
    Enhanced model trainer with comprehensive training capabilities.
    
    Features:
    - Multiple optimizer support
    - Custom callbacks
    - Training resumption
    - Automatic model checkpointing
    - Training metrics tracking
    """
    
    def __init__(self, 
                 model: Model,
                 num_classes: int,
                 model_name: str = "fish_classifier"):
        """
        Initialize the model trainer.
        
        Args:
            model: The Keras model to train
            num_classes: Number of classification classes
            model_name: Name for saving models and logs
        """
        self.model = model
        self.num_classes = num_classes
        self.model_name = model_name
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Training state
        self.is_compiled = False
        self.training_metrics = TrainingMetrics()
        self.callbacks_list: List[Callback] = []
        
        # Paths
        self.model_dir = Path(self.settings.MODEL_SAVE_PATH) / model_name
        self.logs_dir = Path(self.settings.LOGS_DIR) / model_name
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Create necessary directories for training."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
    def compile_model(self,
                     optimizer: str = 'adam',
                     learning_rate: float = 0.001,
                     loss: str = 'categorical_crossentropy',
                     metrics: Optional[List[str]] = None) -> None:
        """
        Compile the model with specified configuration.
        
        Args:
            optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')
            learning_rate: Learning rate for optimization
            loss: Loss function name
            metrics: List of metrics to track
        """
        try:
            # Configure optimizer
            if optimizer.lower() == 'adam':
                opt = Adam(learning_rate=learning_rate)
            elif optimizer.lower() == 'sgd':
                opt = SGD(learning_rate=learning_rate, momentum=0.9)
            elif optimizer.lower() == 'rmsprop':
                opt = RMSprop(learning_rate=learning_rate)
            else:
                raise ConfigurationError(f"Unsupported optimizer: {optimizer}")
            
            # Default metrics
            if metrics is None:
                metrics = ['accuracy', 'top_3_accuracy']
            
            # Compile model
            self.model.compile(
                optimizer=opt,
                loss=loss,
                metrics=metrics
            )
            
            self.is_compiled = True
            self.logger.info(f"Model compiled with {optimizer} optimizer, lr={learning_rate}")
            
        except Exception as e:
            raise TrainingError(f"Failed to compile model: {str(e)}")
    
    def setup_callbacks(self,
                       patience: int = 10,
                       monitor: str = 'val_loss',
                       save_best_only: bool = True,
                       restore_best_weights: bool = True,
                       reduce_lr_patience: int = 5,
                       reduce_lr_factor: float = 0.5,
                       custom_callbacks: Optional[List[Callback]] = None) -> None:
        """
        Setup training callbacks.
        
        Args:
            patience: Early stopping patience
            monitor: Metric to monitor for callbacks
            save_best_only: Whether to save only the best model
            restore_best_weights: Whether to restore best weights on early stopping
            reduce_lr_patience: Patience for learning rate reduction
            reduce_lr_factor: Factor for learning rate reduction
            custom_callbacks: Additional custom callbacks
        """
        try:
            self.callbacks_list = []
            
            # Model checkpoint
            checkpoint_path = self.model_dir / f"{self.model_name}_best.keras"
            checkpoint = ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=monitor,
                save_best_only=save_best_only,
                save_weights_only=False,
                mode='auto',
                verbose=1
            )
            self.callbacks_list.append(checkpoint)
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=restore_best_weights,
                verbose=1,
                mode='auto'
            )
            self.callbacks_list.append(early_stopping)
            
            # Reduce learning rate
            reduce_lr = ReduceLROnPlateau(
                monitor=monitor,
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1,
                mode='auto'
            )
            self.callbacks_list.append(reduce_lr)
            
            # TensorBoard logging
            tensorboard_dir = self.logs_dir / "tensorboard"
            tensorboard = TensorBoard(
                log_dir=str(tensorboard_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch'
            )
            self.callbacks_list.append(tensorboard)
            
            # CSV logger
            csv_path = self.logs_dir / f"{self.model_name}_training.csv"
            csv_logger = CSVLogger(str(csv_path), append=True)
            self.callbacks_list.append(csv_logger)
            
            # Add custom callbacks
            if custom_callbacks:
                self.callbacks_list.extend(custom_callbacks)
            
            self.logger.info(f"Setup {len(self.callbacks_list)} callbacks for training")
            
        except Exception as e:
            raise TrainingError(f"Failed to setup callbacks: {str(e)}")
    
    def train(self,
              train_data: tf.data.Dataset,
              validation_data: Optional[tf.data.Dataset] = None,
              epochs: int = 100,
              initial_epoch: int = 0,
              steps_per_epoch: Optional[int] = None,
              validation_steps: Optional[int] = None,
              class_weight: Optional[Dict[int, float]] = None,
              verbose: int = 1) -> TrainingMetrics:
        """
        Train the model with comprehensive logging and error handling.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset
            epochs: Number of epochs to train
            initial_epoch: Initial epoch (for resuming training)
            steps_per_epoch: Steps per epoch (if None, inferred from dataset)
            validation_steps: Validation steps (if None, inferred from dataset)
            class_weight: Class weights for imbalanced datasets
            verbose: Verbosity level
            
        Returns:
            TrainingMetrics object with training results
        """
        if not self.is_compiled:
            raise TrainingError("Model must be compiled before training")
        
        try:
            self.logger.info(f"Starting training for {epochs} epochs")
            start_time = time.time()
            
            # Train the model
            history = self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=epochs,
                initial_epoch=initial_epoch,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=self.callbacks_list,
                class_weight=class_weight,
                verbose=verbose
            )
            
            # Update training metrics
            self.training_metrics.update_from_history(history)
            self.training_metrics.training_time = time.time() - start_time
            
            self.logger.info(f"Training completed in {self.training_metrics.training_time:.2f} seconds")
            self.logger.info(f"Best validation accuracy: {self.training_metrics.best_val_accuracy:.4f}")
            
            return self.training_metrics
            
        except Exception as e:
            raise TrainingError(f"Training failed: {str(e)}")
    
    def resume_training(self,
                       checkpoint_path: str,
                       train_data: tf.data.Dataset,
                       validation_data: Optional[tf.data.Dataset] = None,
                       epochs: int = 100,
                       **kwargs) -> TrainingMetrics:
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            train_data: Training dataset
            validation_data: Validation dataset
            epochs: Total epochs to train for
            **kwargs: Additional arguments for train method
            
        Returns:
            TrainingMetrics object with training results
        """
        try:
            # Load the checkpoint
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            self.model.load_weights(checkpoint_path)
            self.logger.info(f"Resumed training from checkpoint: {checkpoint_path}")
            
            # Determine initial epoch from checkpoint name or logs
            initial_epoch = kwargs.get('initial_epoch', 0)
            
            return self.train(
                train_data=train_data,
                validation_data=validation_data,
                epochs=epochs,
                initial_epoch=initial_epoch,
                **kwargs
            )
            
        except Exception as e:
            raise TrainingError(f"Failed to resume training: {str(e)}")
    
    def save_model(self, 
                   filepath: Optional[str] = None,
                   save_format: str = 'keras') -> str:
        """
        Save the trained model.
        
        Args:
            filepath: Custom filepath (if None, uses default)
            save_format: Save format ('keras', 'tf', 'h5')
            
        Returns:
            Path where the model was saved
        """
        try:
            if filepath is None:
                if save_format == 'keras':
                    filepath = self.model_dir / f"{self.model_name}_final.keras"
                elif save_format == 'h5':
                    filepath = self.model_dir / f"{self.model_name}_final.h5"
                else:
                    filepath = self.model_dir / f"{self.model_name}_final"
            
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            if save_format == 'keras' or save_format == 'h5':
                self.model.save(filepath)
            elif save_format == 'tf':
                tf.saved_model.save(self.model, str(filepath))
            else:
                raise ConfigurationError(f"Unsupported save format: {save_format}")
            
            self.logger.info(f"Model saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            raise TrainingError(f"Failed to save model: {str(e)}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the training process.
        
        Returns:
            Dictionary containing training summary
        """
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'total_epochs': self.training_metrics.total_epochs,
            'training_time': self.training_metrics.training_time,
            'best_val_accuracy': self.training_metrics.best_val_accuracy,
            'best_val_loss': self.training_metrics.best_val_loss,
            'model_parameters': self.model.count_params(),
            'model_dir': str(self.model_dir),
            'logs_dir': str(self.logs_dir)
        }


class TransferLearningTrainer(ModelTrainer):
    """
    Specialized trainer for transfer learning scenarios.
    
    Provides functionality for fine-tuning pre-trained models
    with different learning rates for different layers.
    """
    
    def __init__(self, 
                 base_model: Model,
                 num_classes: int,
                 model_name: str = "transfer_learning_classifier"):
        """
        Initialize transfer learning trainer.
        
        Args:
            base_model: Pre-trained base model
            num_classes: Number of classification classes
            model_name: Name for saving models and logs
        """
        # Build complete model with classification head
        model = self._build_transfer_model(base_model, num_classes)
        super().__init__(model, num_classes, model_name)
        
        self.base_model = base_model
        self.freeze_base = True
        
    def _build_transfer_model(self, base_model: Model, num_classes: int) -> Model:
        """Build transfer learning model with classification head."""
        # Add classification layers
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        return model
    
    def freeze_base_model(self, freeze: bool = True):
        """Freeze or unfreeze the base model layers."""
        self.base_model.trainable = not freeze
        self.freeze_base = freeze
        
        if freeze:
            self.logger.info("Base model layers frozen")
        else:
            self.logger.info("Base model layers unfrozen for fine-tuning")
    
    def fine_tune(self,
                  train_data: tf.data.Dataset,
                  validation_data: Optional[tf.data.Dataset] = None,
                  fine_tune_epochs: int = 50,
                  fine_tune_lr: float = 1e-5,
                  **kwargs) -> TrainingMetrics:
        """
        Perform fine-tuning after initial training.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset
            fine_tune_epochs: Number of fine-tuning epochs
            fine_tune_lr: Learning rate for fine-tuning
            **kwargs: Additional arguments for train method
            
        Returns:
            TrainingMetrics object with fine-tuning results
        """
        try:
            # Unfreeze base model
            self.freeze_base_model(False)
            
            # Recompile with lower learning rate
            self.compile_model(
                optimizer='adam',
                learning_rate=fine_tune_lr,
                loss='categorical_crossentropy',
                metrics=['accuracy', 'top_3_accuracy']
            )
            
            self.logger.info(f"Starting fine-tuning for {fine_tune_epochs} epochs")
            
            # Fine-tune
            return self.train(
                train_data=train_data,
                validation_data=validation_data,
                epochs=fine_tune_epochs,
                **kwargs
            )
            
        except Exception as e:
            raise TrainingError(f"Fine-tuning failed: {str(e)}")