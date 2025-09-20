"""
Model Manager for Fish Species Classification

This module provides comprehensive model management functionality including
model creation, loading, saving, and configuration management.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
import tensorflow as tf
from pathlib import Path

from config.model_configs import ModelConfig
from src.utils.logging_utils import get_logger


class ModelManager:
    """
    Comprehensive model manager for fish species classification models.
    
    Handles model creation, loading, saving, and configuration management.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the model manager.
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.model = None
        
        self.logger.info(f"Initialized ModelManager with architecture: {config.architecture}")
    
    def create_model(self) -> tf.keras.Model:
        """
        Create a model based on the configuration.
        
        Returns:
            Compiled Keras model
        """
        self.logger.info(f"Creating model with architecture: {self.config.architecture}")
        
        if self.config.architecture == "cnn":
            model = self._create_cnn_model()
        elif self.config.architecture == "resnet50":
            model = self._create_resnet50_model()
        elif self.config.architecture == "vgg16":
            model = self._create_vgg16_model()
        else:
            raise ValueError(f"Unsupported architecture: {self.config.architecture}")
        
        self.model = model
        self.logger.info(f"Model created successfully with {model.count_params():,} parameters")
        
        return model
    
    def _create_cnn_model(self) -> tf.keras.Model:
        """Create a basic CNN model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.config.num_classes, activation='softmax')
        ])
        
        return model
    
    def _create_resnet50_model(self) -> tf.keras.Model:
        """Create a ResNet50-based model."""
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.config.num_classes, activation='softmax')
        ])
        
        return model
    
    def _create_vgg16_model(self) -> tf.keras.Model:
        """Create a VGG16-based model."""
        base_model = tf.keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.config.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, model: tf.keras.Model = None) -> None:
        """
        Compile the model with specified optimizer and loss function.
        
        Args:
            model: Model to compile (uses self.model if None)
        """
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model to compile. Create a model first.")
        
        # Get optimizer
        if self.config.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.config.learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.logger.info("Model compiled successfully")
    
    def save_model(self, filepath: str, model: tf.keras.Model = None) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save the model
            model: Model to save (uses self.model if None)
        """
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model to save")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        model.save(filepath)
        self.logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> tf.keras.Model:
        """
        Load model from file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded Keras model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = tf.keras.models.load_model(filepath)
        self.model = model
        self.logger.info(f"Model loaded from: {filepath}")
        
        return model
    
    def get_model_summary(self, model: tf.keras.Model = None) -> str:
        """
        Get model summary as string.
        
        Args:
            model: Model to summarize (uses self.model if None)
            
        Returns:
            Model summary as string
        """
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model available for summary")
        
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {
                'model_created': False,
                'config': self.config.__dict__
            }
        
        return {
            'model_created': True,
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
            'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights]),
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'config': self.config.__dict__
        }