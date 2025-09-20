"""
Training Pipeline for Fish Species Classification

A comprehensive training pipeline that orchestrates the entire model training process,
from data loading and preprocessing to training, validation, and evaluation.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
import tensorflow as tf
from datetime import datetime

from config.model_configs import ModelConfig
from core.model_manager import ModelManager
from core.dataset_manager import DatasetManager
from evaluation.model_evaluator import ModelEvaluator
from evaluation.metrics_calculator import MetricsCalculator
from evaluation.performance_analyzer import PerformanceAnalyzer
from utils.logging_utils import setup_logging


class TrainingPipeline:
    """
    Comprehensive training pipeline for fish species classification models.
    
    Handles the complete workflow from data loading to model evaluation,
    with experiment tracking and advanced configuration management.
    """
    
    def __init__(self, config: ModelConfig, experiment_name: Optional[str] = None):
        """
        Initialize the training pipeline.
        
        Args:
            config: Model configuration object
            experiment_name: Optional experiment name for tracking
        """
        self.config = config
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize components
        self.model_manager = ModelManager(config)
        self.dataset_manager = DatasetManager(config)
        self.evaluator = ModelEvaluator()
        self.metrics_calculator = MetricsCalculator()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Setup logging
        self.logger = setup_logging(f"training_pipeline_{self.experiment_name}")
        
        # Training state
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.training_history = None
        self.evaluation_results = None
        
        self.logger.info(f"Initialized training pipeline for experiment: {self.experiment_name}")
    
    def prepare_data(self) -> None:
        """Prepare training, validation, and test datasets."""
        self.logger.info("Preparing datasets...")
        
        try:
            # Load and prepare datasets
            self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_manager.create_datasets()
            
            self.logger.info("Datasets prepared successfully")
            self.logger.info(f"Training batches: {len(list(self.train_dataset))}")
            self.logger.info(f"Validation batches: {len(list(self.val_dataset))}")
            self.logger.info(f"Test batches: {len(list(self.test_dataset))}")
            
        except Exception as e:
            self.logger.error(f"Error preparing datasets: {e}")
            raise
    
    def build_model(self) -> None:
        """Build and compile the model."""
        self.logger.info("Building model...")
        
        try:
            # Create model
            self.model = self.model_manager.create_model()
            
            # Compile model
            self.model_manager.compile_model(self.model)
            
            self.logger.info("Model built and compiled successfully")
            self.logger.info(f"Model architecture: {self.config.architecture}")
            self.logger.info(f"Total parameters: {self.model.count_params():,}")
            
        except Exception as e:
            self.logger.error(f"Error building model: {e}")
            raise
    
    def train_model(self) -> None:
        """Train the model with the prepared datasets."""
        self.logger.info("Starting model training...")
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if self.train_dataset is None or self.val_dataset is None:
            raise ValueError("Datasets not prepared. Call prepare_data() first.")
        
        try:
            # Prepare callbacks
            callbacks = self._prepare_callbacks()
            
            # Train model
            self.training_history = self.model.fit(
                self.train_dataset,
                validation_data=self.val_dataset,
                epochs=self.config.epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            self.logger.info("Model training completed successfully")
            
            # Save training history
            self._save_training_history()
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
    
    def evaluate_model(self) -> Dict[str, Any]:
        """Evaluate the trained model on test dataset."""
        self.logger.info("Evaluating model...")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if self.test_dataset is None:
            raise ValueError("Test dataset not prepared. Call prepare_data() first.")
        
        try:
            # Get predictions
            y_true, y_pred, y_prob = self.evaluator.get_predictions(self.model, self.test_dataset)
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(y_true, y_pred, y_prob)
            
            # Analyze performance
            patterns = self.performance_analyzer.analyze_prediction_patterns(y_true, y_pred, y_prob)
            
            # Combine results
            self.evaluation_results = {
                'metrics': metrics,
                'patterns': patterns,
                'experiment_name': self.experiment_name,
                'model_config': self.config.__dict__,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("Model evaluation completed successfully")
            self.logger.info(f"Test accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            
            # Save evaluation results
            self._save_evaluation_results()
            
            return self.evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            raise
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline from start to finish.
        
        Returns:
            Dictionary containing evaluation results and training metadata
        """
        self.logger.info(f"Starting complete training pipeline: {self.experiment_name}")
        
        try:
            # Execute pipeline steps
            self.prepare_data()
            self.build_model()
            self.train_model()
            results = self.evaluate_model()
            
            # Save model
            self.save_model()
            
            self.logger.info("Complete training pipeline finished successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the trained model.
        
        Args:
            filepath: Optional custom filepath for saving
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        if filepath is None:
            filepath = f"models/{self.experiment_name}_final_model.h5"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        self.model.save(filepath)
        self.logger.info(f"Model saved to: {filepath}")
        
        return filepath
    
    def _prepare_callbacks(self) -> list:
        """Prepare training callbacks."""
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = f"models/checkpoints/{self.experiment_name}"
        os.makedirs(checkpoint_path, exist_ok=True)
        
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{checkpoint_path}/best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        )
        
        # Early stopping
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        )
        
        # Learning rate reduction
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        )
        
        # CSV logger
        log_path = f"logs/{self.experiment_name}_training_log.csv"
        os.makedirs("logs", exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.CSVLogger(log_path)
        )
        
        return callbacks
    
    def _save_training_history(self) -> None:
        """Save training history to file."""
        if self.training_history is None:
            return
        
        history_path = f"logs/{self.experiment_name}_history.json"
        os.makedirs("logs", exist_ok=True)
        
        # Convert history to serializable format
        history_dict = {key: [float(val) for val in values] 
                       for key, values in self.training_history.history.items()}
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        self.logger.info(f"Training history saved to: {history_path}")
    
    def _save_evaluation_results(self) -> None:
        """Save evaluation results to file."""
        if self.evaluation_results is None:
            return
        
        results_path = f"results/{self.experiment_name}_evaluation.json"
        os.makedirs("results", exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation results saved to: {results_path}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment."""
        return {
            'experiment_name': self.experiment_name,
            'config': self.config.__dict__,
            'model_parameters': self.model.count_params() if self.model else None,
            'training_completed': self.training_history is not None,
            'evaluation_completed': self.evaluation_results is not None,
            'timestamp': datetime.now().isoformat()
        }