"""
Training Manager Module
======================

Comprehensive training manager for fish species classification with support for
different architectures, advanced training strategies, and experiment tracking.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from .training_config import TrainingConfig, ModelArchitecture, OptimizerType, SchedulerType
from .experiment_tracker import ExperimentTracker
from config.model_configs import ModelFactory
from ..data.dataset_manager import DatasetManager
from ..utils.logging_utils import get_logger
from ..utils.visualization import create_prediction_chart, create_heatmap_visualization

logger = get_logger(__name__)

class TrainingManager:
    """Comprehensive training manager with advanced features"""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize training manager
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        # Use default dataset path if not specified in config
        dataset_path = getattr(config, 'dataset_path', 'FishImgDataset')
        self.dataset_manager = DatasetManager(dataset_path=dataset_path)
        self.experiment_tracker = ExperimentTracker(config.experiment_name)
        
        # Training data
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.class_names = None
        self.class_weights = None
        
        # Training state
        self.training_history = None
        self.best_model_path = None
        self.training_callbacks = []
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
        # Configure TensorFlow
        self._configure_tensorflow()
        
        logger.info(f"Training manager initialized for experiment: {config.experiment_name}")
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility"""
        np.random.seed(self.config.random_seed)
        tf.random.set_seed(self.config.random_seed)
        
        # Additional seeding for Python's random module
        import random
        random.seed(self.config.random_seed)
    
    def _configure_tensorflow(self):
        """Configure TensorFlow settings"""
        # Mixed precision training
        if self.config.mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision training enabled")
        
        # GPU configuration
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Configured {len(gpus)} GPU(s)")
            except RuntimeError as e:
                logger.warning(f"GPU configuration failed: {e}")
        else:
            logger.info("No GPUs found, using CPU")
    
    def prepare_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Prepare training, validation, and test datasets
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Preparing datasets...")
        
        # Load dataset information
        dataset_info = self.dataset_manager.get_dataset_info()
        self.class_names = self.dataset_manager.get_class_names()
        
        # Log dataset information
        self.experiment_tracker.log_info(f"Dataset: {len(self.class_names)} classes, {dataset_info.get('total_images', 0)} images")
        
        # Create data generators with augmentation
        train_datagen = self._create_train_data_generator()
        val_datagen = self._create_val_data_generator()
        
        # Load data
        train_generator = train_datagen.flow_from_directory(
            self.config.dataset_path / "train",
            target_size=self.config.image_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=self.config.random_seed
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.config.dataset_path / "val",
            target_size=self.config.image_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = val_datagen.flow_from_directory(
            self.config.dataset_path / "test",
            target_size=self.config.image_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Store class names
        self.class_names = list(train_generator.class_indices.keys())
        
        # Calculate class weights
        if self.config.use_class_weights:
            self.class_weights = self._calculate_class_weights(train_generator)
            self.experiment_tracker.log_info(f"Calculated class weights for {len(self.class_weights)} classes")
        
        # Convert to tf.data.Dataset if needed
        self.train_data = train_generator
        self.val_data = val_generator
        self.test_data = test_generator
        
        # Log data preparation completion
        logger.info(f"Data prepared: {train_generator.samples} train, {val_generator.samples} val, {test_generator.samples} test")
        
        return self.train_data, self.val_data, self.test_data
    
    def _create_train_data_generator(self) -> ImageDataGenerator:
        """Create training data generator with augmentation"""
        aug_config = self.config.augmentation
        
        return ImageDataGenerator(
            rescale=1.0/255.0 if self.config.normalize_pixels else 1.0,
            horizontal_flip=aug_config.horizontal_flip,
            vertical_flip=aug_config.vertical_flip,
            rotation_range=aug_config.rotation_range,
            width_shift_range=aug_config.width_shift_range,
            height_shift_range=aug_config.height_shift_range,
            zoom_range=aug_config.zoom_range,
            shear_range=aug_config.shear_range,
            brightness_range=aug_config.brightness_range,
            fill_mode='nearest'
        )
    
    def _create_val_data_generator(self) -> ImageDataGenerator:
        """Create validation data generator (no augmentation)"""
        return ImageDataGenerator(
            rescale=1.0/255.0 if self.config.normalize_pixels else 1.0
        )
    
    def _calculate_class_weights(self, generator) -> Dict[int, float]:
        """Calculate class weights for imbalanced dataset"""
        # Get class counts
        class_counts = np.bincount(generator.classes)
        
        # Calculate weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.arange(len(class_counts)),
            y=generator.classes
        )
        
        return dict(enumerate(class_weights))
    
    def build_model(self) -> keras.Model:
        """
        Build model based on configuration
        
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building model: {self.config.model.architecture.value}")
        
        # Create model using factory
        model_factory = ModelFactory()
        
        if self.config.model.architecture == ModelArchitecture.CUSTOM_CNN:
            self.model = model_factory.create_custom_cnn(
                input_shape=self.config.model.input_shape,
                num_classes=self.config.model.num_classes,
                dropout_rate=self.config.model.dropout_rate
            )
        else:
            self.model = model_factory.create_transfer_learning_model(
                architecture=self.config.model.architecture.value,
                input_shape=self.config.model.input_shape,
                num_classes=self.config.model.num_classes,
                dropout_rate=self.config.model.dropout_rate,
                use_pretrained=self.config.model.use_pretrained,
                freeze_base=self.config.model.freeze_base
            )
        
        # Compile model
        self._compile_model()
        
        # Log model information
        self.experiment_tracker.log_info(f"Model created: {self.model.count_params():,} parameters")
        
        return self.model
    
    def _compile_model(self):
        """Compile the model with optimizer and loss function"""
        # Create optimizer
        optimizer = self._create_optimizer()
        
        # Define loss function
        loss = 'categorical_crossentropy'
        if self.config.label_smoothing > 0:
            loss = keras.losses.CategoricalCrossentropy(label_smoothing=self.config.label_smoothing)
        
        # Define metrics
        metrics = ['accuracy', 'top_k_categorical_accuracy']
        
        # Compile
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with {self.config.optimizer.type.value} optimizer")
    
    def _create_optimizer(self) -> keras.optimizers.Optimizer:
        """Create optimizer based on configuration"""
        opt_config = self.config.optimizer
        
        if opt_config.type == OptimizerType.ADAM:
            return optimizers.Adam(
                learning_rate=opt_config.learning_rate,
                beta_1=opt_config.beta1,
                beta_2=opt_config.beta2,
                epsilon=opt_config.epsilon,
                weight_decay=opt_config.weight_decay,
                clipnorm=opt_config.clipnorm,
                clipvalue=opt_config.clipvalue
            )
        elif opt_config.type == OptimizerType.ADAMW:
            return optimizers.AdamW(
                learning_rate=opt_config.learning_rate,
                beta_1=opt_config.beta1,
                beta_2=opt_config.beta2,
                epsilon=opt_config.epsilon,
                weight_decay=opt_config.weight_decay,
                clipnorm=opt_config.clipnorm,
                clipvalue=opt_config.clipvalue
            )
        elif opt_config.type == OptimizerType.SGD:
            return optimizers.SGD(
                learning_rate=opt_config.learning_rate,
                momentum=opt_config.momentum,
                weight_decay=opt_config.weight_decay,
                clipnorm=opt_config.clipnorm,
                clipvalue=opt_config.clipvalue
            )
        elif opt_config.type == OptimizerType.RMSPROP:
            return optimizers.RMSprop(
                learning_rate=opt_config.learning_rate,
                momentum=opt_config.momentum,
                epsilon=opt_config.epsilon,
                weight_decay=opt_config.weight_decay,
                clipnorm=opt_config.clipnorm,
                clipvalue=opt_config.clipvalue
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_config.type}")
    
    def _create_callbacks(self) -> List[keras.callbacks.Callback]:
        """Create training callbacks"""
        callback_list = []
        cb_config = self.config.callbacks
        
        # Early stopping
        if cb_config.early_stopping:
            early_stopping = callbacks.EarlyStopping(
                monitor=cb_config.monitor_metric,
                patience=cb_config.early_stopping_patience,
                min_delta=cb_config.early_stopping_min_delta,
                restore_best_weights=cb_config.early_stopping_restore_best,
                verbose=1
            )
            callback_list.append(early_stopping)
        
        # Model checkpoint
        if cb_config.model_checkpoint:
            self.best_model_path = self.config.get_experiment_dir() / "checkpoints" / "best_model.keras"
            
            checkpoint = callbacks.ModelCheckpoint(
                filepath=str(self.best_model_path),
                monitor=cb_config.monitor_metric,
                save_best_only=cb_config.save_best_only,
                save_weights_only=cb_config.save_weights_only,
                verbose=1
            )
            callback_list.append(checkpoint)
        
        # Reduce learning rate on plateau
        if cb_config.reduce_lr_on_plateau:
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor=cb_config.monitor_metric,
                factor=cb_config.lr_factor,
                patience=cb_config.lr_patience,
                min_delta=cb_config.lr_min_delta,
                verbose=1
            )
            callback_list.append(reduce_lr)
        
        # TensorBoard
        if cb_config.tensorboard:
            tensorboard_dir = self.config.get_experiment_dir() / "logs" / "tensorboard"
            tensorboard = callbacks.TensorBoard(
                log_dir=str(tensorboard_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
            callback_list.append(tensorboard)
        
        # CSV Logger
        if cb_config.csv_logger:
            csv_log_path = self.config.get_experiment_dir() / "logs" / "training.csv"
            csv_logger = callbacks.CSVLogger(str(csv_log_path))
            callback_list.append(csv_logger)
        
        # Custom callback for experiment tracking
        experiment_callback = ExperimentTrackingCallback(self.experiment_tracker)
        callback_list.append(experiment_callback)
        
        return callback_list
    
    def train(self) -> keras.callbacks.History:
        """
        Train the model
        
        Returns:
            Training history
        """
        # Start experiment tracking
        self.experiment_tracker.start_experiment(self.config.to_dict())
        
        try:
            # Prepare data
            self.prepare_data()
            
            # Build model
            self.build_model()
            
            # Create callbacks
            self.training_callbacks = self._create_callbacks()
            
            # Start training
            logger.info("Starting training...")
            start_time = time.time()
            
            # Train model
            self.training_history = self.model.fit(
                self.train_data,
                epochs=self.config.epochs,
                validation_data=self.val_data,
                callbacks=self.training_callbacks,
                class_weight=self.class_weights,
                verbose=1
            )
            
            # Calculate training time
            training_time = time.time() - start_time
            self.experiment_tracker.log_metric("training_time_seconds", training_time)
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Evaluate model
            self.evaluate_model()
            
            # Save final model
            self.save_model()
            
            # Generate training plots
            self.plot_training_history()
            
            # End experiment tracking
            self.experiment_tracker.end_experiment("completed")
            
            return self.training_history
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.experiment_tracker.log_error(f"Training failed: {e}")
            self.experiment_tracker.end_experiment("failed")
            raise
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model...")
        
        # Load best model if available
        if self.best_model_path and os.path.exists(self.best_model_path):
            self.model = keras.models.load_model(self.best_model_path)
        
        # Evaluate on test data
        test_loss, test_accuracy, test_top_k = self.model.evaluate(
            self.test_data,
            verbose=1
        )
        
        # Log metrics
        metrics = {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_top_k_accuracy": test_top_k
        }
        
        for name, value in metrics.items():
            self.experiment_tracker.log_metric(name, value)
        
        # Generate predictions for detailed analysis
        self.generate_test_predictions()
        
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        return metrics
    
    def generate_test_predictions(self):
        """Generate and analyze test predictions"""
        logger.info("Generating test predictions...")
        
        # Reset test generator
        self.test_data.reset()
        
        # Generate predictions
        predictions = self.model.predict(self.test_data, verbose=1)
        
        # Get true labels
        true_labels = self.test_data.classes
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Generate classification report
        report = classification_report(
            true_labels,
            predicted_labels,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Save classification report
        report_df = pd.DataFrame(report).transpose()
        self.experiment_tracker.log_dataframe(report_df, "classification_report")
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Plot confusion matrix
        fig = self._plot_confusion_matrix(cm)
        self.experiment_tracker.log_plot(fig, "confusion_matrix")
        
        # Log per-class accuracies
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(per_class_accuracy):
            self.experiment_tracker.log_metric(f"accuracy_{self.class_names[i]}", acc)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'true_label': true_labels,
            'predicted_label': predicted_labels,
            'true_class': [self.class_names[i] for i in true_labels],
            'predicted_class': [self.class_names[i] for i in predicted_labels],
            'confidence': np.max(predictions, axis=1)
        })
        
        self.experiment_tracker.log_dataframe(predictions_df, "test_predictions")
        
        logger.info("Test predictions analysis completed")
    
    def _plot_confusion_matrix(self, cm: np.ndarray) -> plt.Figure:
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Labels
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.class_names,
               yticklabels=self.class_names,
               title='Normalized Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        # Rotate the tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm_normalized[i, j], '.2f'),
                       ha="center", va="center",
                       color="white" if cm_normalized[i, j] > thresh else "black")
        
        plt.tight_layout()
        return fig
    
    def plot_training_history(self):
        """Plot and save training history"""
        if self.training_history is None:
            return
        
        # Use experiment tracker's built-in plotting
        self.experiment_tracker.plot_training_curves(save=True)
        
        # Additional custom plots
        history = self.training_history.history
        
        # Learning rate plot (if available)
        if 'lr' in history:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(history['lr'])
            ax.set_title('Learning Rate Schedule')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            self.experiment_tracker.log_plot(fig, "learning_rate_schedule")
    
    def save_model(self, model_name: str = None):
        """Save the trained model"""
        if self.model is None:
            logger.warning("No model to save")
            return
        
        model_name = model_name or self.config.get_model_name()
        model_path = self.config.get_experiment_dir() / "models" / f"{model_name}.keras"
        
        # Save model
        self.model.save(model_path)
        self.experiment_tracker.log_model(self.model, model_name)
        
        logger.info(f"Model saved: {model_path}")
    
    def load_model(self, model_path: str):
        """Load a saved model"""
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from: {model_path}")

class ExperimentTrackingCallback(keras.callbacks.Callback):
    """Custom callback for experiment tracking"""
    
    def __init__(self, experiment_tracker: ExperimentTracker):
        super().__init__()
        self.experiment_tracker = experiment_tracker
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch"""
        if logs:
            self.experiment_tracker.log_metrics(logs, step=epoch)
    
    def on_train_begin(self, logs=None):
        """Log training start"""
        self.experiment_tracker.log_info("Training started")
    
    def on_train_end(self, logs=None):
        """Log training end"""
        self.experiment_tracker.log_info("Training completed")
    
    def validate_training_config(self) -> Dict[str, Any]:
        """
        Validate training configuration and return validation results.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'experiment_name': hasattr(self.config, 'experiment_name'),
            'dataset_path': hasattr(self.config, 'dataset_path'),
            'random_seed': hasattr(self.config, 'random_seed'),
            'mixed_precision': hasattr(self.config, 'mixed_precision'),
            'dataset_manager_initialized': self.dataset_manager is not None,
            'experiment_tracker_initialized': self.experiment_tracker is not None
        }
        
        return validation_results

def run_training_from_config(config_path: str):
    """
    Run training from configuration file
    
    Args:
        config_path: Path to configuration JSON file
    """
    # Load configuration
    config = TrainingConfig.load_from_file(config_path)
    
    # Validate configuration
    issues = config.validate()
    if issues:
        logger.error(f"Configuration validation failed: {issues}")
        return
    
    # Create directories
    config.create_directories()
    
    # Initialize training manager
    trainer = TrainingManager(config)
    
    # Run training
    trainer.train()
    
    logger.info(f"Training completed for experiment: {config.experiment_name}")


if __name__ == "__main__":
    # Example usage
    config = TrainingConfig()
    config.experiment_name = "test_training"
    
    trainer = TrainingManager(config)
    trainer.train()