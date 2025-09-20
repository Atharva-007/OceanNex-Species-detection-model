"""Model management and architecture definitions."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model, save_model
    from tensorflow.keras import optimizers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

from ..utils.logger import get_logger
from ..utils.exceptions import ModelError, ConfigError


logger = get_logger(__name__)


class ModelManager:
    """Manages model loading, saving, and configuration."""
    
    def __init__(self, models_path: Union[str, Path]):
        """
        Initialize model manager.
        
        Args:
            models_path: Path to models directory
        """
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(self.__class__.__name__)
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required but not installed")
    
    def save_model(
        self, 
        model: tf.keras.Model, 
        model_name: str,
        class_mapping: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save model and associated metadata.
        
        Args:
            model: Keras model to save
            model_name: Name for the saved model
            class_mapping: Class name to index mapping
            metadata: Additional metadata
            
        Returns:
            Path to saved model file
            
        Raises:
            ModelError: If saving fails
        """
        try:
            # Ensure model name has proper extension
            if not model_name.endswith('.keras'):
                model_name += '.keras'
            
            model_path = self.models_path / model_name
            
            # Save model
            model.save(model_path)
            self.logger.info(f"Model saved to {model_path}")
            
            # Save class mapping if provided
            if class_mapping:
                mapping_path = self.models_path / f"{model_name.replace('.keras', '_class_mapping.json')}"
                with open(mapping_path, 'w') as f:
                    json.dump(class_mapping, f, indent=2)
                self.logger.info(f"Class mapping saved to {mapping_path}")
            
            # Save metadata if provided
            if metadata:
                metadata_path = self.models_path / f"{model_name.replace('.keras', '_metadata.json')}"
                
                # Add model info to metadata
                model_info = {
                    'total_params': model.count_params(),
                    'input_shape': model.input_shape,
                    'output_shape': model.output_shape,
                    'model_name': model_name
                }
                
                full_metadata = {**metadata, 'model_info': model_info}
                
                with open(metadata_path, 'w') as f:
                    json.dump(full_metadata, f, indent=2, default=str)
                self.logger.info(f"Metadata saved to {metadata_path}")
            
            return model_path
            
        except Exception as e:
            raise ModelError(f"Failed to save model {model_name}: {str(e)}") from e
    
    def load_model(
        self, 
        model_name: str,
        custom_objects: Optional[Dict[str, Any]] = None
    ) -> Tuple[tf.keras.Model, Optional[Dict[str, int]], Optional[Dict[str, Any]]]:
        """
        Load model and associated metadata.
        
        Args:
            model_name: Name of model to load
            custom_objects: Custom objects for model loading
            
        Returns:
            Tuple of (model, class_mapping, metadata)
            
        Raises:
            ModelError: If loading fails
        """
        try:
            # Ensure model name has proper extension
            if not model_name.endswith('.keras'):
                model_name += '.keras'
            
            model_path = self.models_path / model_name
            
            if not model_path.exists():
                raise ModelError(f"Model file not found: {model_path}")
            
            # Load model
            model = load_model(str(model_path), custom_objects=custom_objects)
            self.logger.info(f"Model loaded from {model_path}")
            
            # Load class mapping if exists
            mapping_path = self.models_path / f"{model_name.replace('.keras', '_class_mapping.json')}"
            class_mapping = None
            
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    class_mapping = json.load(f)
                self.logger.info(f"Class mapping loaded from {mapping_path}")
            
            # Load metadata if exists
            metadata_path = self.models_path / f"{model_name.replace('.keras', '_metadata.json')}"
            metadata = None
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.logger.info(f"Metadata loaded from {metadata_path}")
            
            return model, class_mapping, metadata
            
        except Exception as e:
            raise ModelError(f"Failed to load model {model_name}: {str(e)}") from e
    
    def list_models(self) -> List[str]:
        """
        List available model files.
        
        Returns:
            List of model names
        """
        try:
            model_files = []
            for model_path in self.models_path.glob('*.keras'):
                model_files.append(model_path.name)
            
            self.logger.info(f"Found {len(model_files)} model files")
            return sorted(model_files)
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []
    
    def delete_model(self, model_name: str, delete_metadata: bool = True) -> bool:
        """
        Delete model and optionally its metadata.
        
        Args:
            model_name: Name of model to delete
            delete_metadata: Whether to delete associated metadata files
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            # Ensure model name has proper extension
            if not model_name.endswith('.keras'):
                model_name += '.keras'
            
            model_path = self.models_path / model_name
            
            if model_path.exists():
                model_path.unlink()
                self.logger.info(f"Deleted model: {model_path}")
            
            if delete_metadata:
                # Delete class mapping
                mapping_path = self.models_path / f"{model_name.replace('.keras', '_class_mapping.json')}"
                if mapping_path.exists():
                    mapping_path.unlink()
                    self.logger.info(f"Deleted class mapping: {mapping_path}")
                
                # Delete metadata
                metadata_path = self.models_path / f"{model_name.replace('.keras', '_metadata.json')}"
                if metadata_path.exists():
                    metadata_path.unlink()
                    self.logger.info(f"Deleted metadata: {metadata_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get model information without loading the full model.
        
        Args:
            model_name: Name of model
            
        Returns:
            Model information dictionary or None if not found
        """
        try:
            # Ensure model name has proper extension
            if not model_name.endswith('.keras'):
                model_name += '.keras'
            
            metadata_path = self.models_path / f"{model_name.replace('.keras', '_metadata.json')}"
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return metadata.get('model_info', {})
            else:
                # Try to get basic info from model file
                model_path = self.models_path / model_name
                if model_path.exists():
                    return {
                        'model_name': model_name,
                        'file_size': model_path.stat().st_size,
                        'file_path': str(model_path)
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get model info for {model_name}: {e}")
            return None


class ModelTrainer:
    """Handles model training with various configurations."""
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize model trainer.
        
        Args:
            model_manager: ModelManager instance
        """
        self.model_manager = model_manager
        self.logger = get_logger(self.__class__.__name__)
        self.model = None
        self.history = None
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required but not installed")
    
    def compile_model(
        self,
        model: tf.keras.Model,
        optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam',
        learning_rate: float = 0.001,
        loss: str = 'categorical_crossentropy',
        metrics: List[str] = None
    ) -> tf.keras.Model:
        """
        Compile model with specified configuration.
        
        Args:
            model: Keras model to compile
            optimizer: Optimizer name or instance
            learning_rate: Learning rate
            loss: Loss function
            metrics: List of metrics to track
            
        Returns:
            Compiled model
        """
        if metrics is None:
            metrics = ['accuracy']
        
        # Setup optimizer
        if isinstance(optimizer, str):
            if optimizer.lower() == 'adam':
                optimizer_instance = optimizers.Adam(learning_rate=learning_rate)
            elif optimizer.lower() == 'sgd':
                optimizer_instance = optimizers.SGD(learning_rate=learning_rate)
            elif optimizer.lower() == 'rmsprop':
                optimizer_instance = optimizers.RMSprop(learning_rate=learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
        else:
            optimizer_instance = optimizer
        
        # Compile model
        model.compile(
            optimizer=optimizer_instance,
            loss=loss,
            metrics=metrics
        )
        
        self.logger.info(f"Model compiled with {optimizer} optimizer, lr={learning_rate}")
        return model
    
    def setup_callbacks(
        self,
        model_name: str,
        monitor: str = 'val_loss',
        patience: int = 15,
        reduce_lr_patience: int = 8,
        reduce_lr_factor: float = 0.5,
        save_best_only: bool = True,
        restore_best_weights: bool = True
    ) -> List[tf.keras.callbacks.Callback]:
        """
        Setup training callbacks.
        
        Args:
            model_name: Name for saved model
            monitor: Metric to monitor
            patience: Early stopping patience
            reduce_lr_patience: Learning rate reduction patience
            reduce_lr_factor: Learning rate reduction factor
            save_best_only: Save only best model
            restore_best_weights: Restore best weights on early stopping
            
        Returns:
            List of callbacks
        """
        callbacks_list = []
        
        # Model checkpoint
        checkpoint_path = self.model_manager.models_path / f"best_{model_name}"
        if not checkpoint_path.name.endswith('.keras'):
            checkpoint_path = checkpoint_path.with_suffix('.keras')
        
        checkpoint = callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=False,
            mode='auto',
            verbose=1
        )
        callbacks_list.append(checkpoint)
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=restore_best_weights,
            verbose=1
        )
        callbacks_list.append(early_stopping)
        
        # Reduce learning rate
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        )
        callbacks_list.append(reduce_lr)
        
        self.logger.info(f"Setup {len(callbacks_list)} callbacks")
        return callbacks_list
    
    def train_model(
        self,
        model: tf.keras.Model,
        train_data,
        validation_data,
        epochs: int = 50,
        callbacks_list: Optional[List[tf.keras.callbacks.Callback]] = None,
        class_weight: Optional[Dict[int, float]] = None,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Train model with specified data and configuration.
        
        Args:
            model: Compiled Keras model
            train_data: Training data generator
            validation_data: Validation data generator
            epochs: Number of training epochs
            callbacks_list: List of callbacks
            class_weight: Class weights for imbalanced datasets
            verbose: Verbosity level
            
        Returns:
            Training history
            
        Raises:
            ModelError: If training fails
        """
        try:
            self.model = model
            
            # Log training start
            total_train_samples = getattr(train_data, 'samples', 0)
            total_val_samples = getattr(validation_data, 'samples', 0)
            
            self.logger.info(f"Starting training:")
            self.logger.info(f"  Epochs: {epochs}")
            self.logger.info(f"  Train samples: {total_train_samples}")
            self.logger.info(f"  Validation samples: {total_val_samples}")
            self.logger.info(f"  Class weights: {'Yes' if class_weight else 'No'}")
            
            # Train model
            self.history = model.fit(
                train_data,
                epochs=epochs,
                validation_data=validation_data,
                callbacks=callbacks_list,
                class_weight=class_weight,
                verbose=verbose
            )
            
            self.logger.info("Training completed successfully")
            return self.history
            
        except Exception as e:
            raise ModelError(f"Training failed: {str(e)}") from e
    
    def get_training_history(self) -> Optional[Dict[str, List[float]]]:
        """
        Get training history.
        
        Returns:
            Training history dictionary or None
        """
        if self.history:
            return self.history.history
        return None
    
    def save_training_results(
        self,
        model_name: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save training results and model.
        
        Args:
            model_name: Name for saved model
            additional_metadata: Additional metadata to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.model is None:
                self.logger.warning("No trained model to save")
                return False
            
            # Prepare metadata
            metadata = {
                'training_completed': True,
                'training_history': self.get_training_history()
            }
            
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Save model with metadata
            self.model_manager.save_model(
                self.model,
                model_name,
                metadata=metadata
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save training results: {e}")
            return False