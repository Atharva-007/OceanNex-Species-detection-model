"""
Dataset management module for fish species classification.

This module provides comprehensive dataset management functionality including
dataset loading, splitting, validation, and metadata management.
"""

import os
import json
import shutil
import random
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.utils.logging_utils import get_logger
from src.utils.exceptions import DatasetError, ValidationError
from config.settings import get_settings


class SplitType(Enum):
    """Dataset split types."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass
class DatasetSplit:
    """Container for dataset split information."""
    train_size: float
    validation_size: float
    test_size: float
    random_state: int = 42
    
    def __post_init__(self):
        """Validate split sizes."""
        total = self.train_size + self.validation_size + self.test_size
        if not abs(total - 1.0) < 1e-6:
            raise ValidationError(f"Split sizes must sum to 1.0, got {total}")


class DatasetManager:
    """
    Comprehensive dataset management system.
    
    Features:
    - Dataset loading and validation
    - Train/validation/test splitting
    - Class distribution analysis
    - Dataset statistics
    - TensorFlow dataset creation
    """
    
    def __init__(self,
                 dataset_path: str,
                 class_mapping_file: Optional[str] = None):
        """
        Initialize the dataset manager.
        
        Args:
            dataset_path: Path to the dataset directory
            class_mapping_file: Path to class mapping JSON file
        """
        self.dataset_path = Path(dataset_path)
        self.class_mapping_file = class_mapping_file
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Dataset information
        self.class_names: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.num_classes: int = 0
        
        # Dataset statistics
        self.dataset_stats: Dict[str, Any] = {}
        self.class_distribution: Dict[str, int] = {}
        
        # File paths
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load and validate the dataset."""
        try:
            self.logger.info(f"Loading dataset from: {self.dataset_path}")
            
            if not self.dataset_path.exists():
                raise DatasetError(f"Dataset path does not exist: {self.dataset_path}")
            
            # Load class mapping if provided
            if self.class_mapping_file and os.path.exists(self.class_mapping_file):
                self._load_class_mapping()
            else:
                self._discover_classes()
            
            # Load image paths and labels
            self._load_image_paths()
            
            # Calculate statistics
            self._calculate_statistics()
            
            self.logger.info(f"Dataset loaded successfully: {len(self.image_paths)} images, "
                           f"{self.num_classes} classes")
            
        except Exception as e:
            raise DatasetError(f"Failed to load dataset: {str(e)}")
    
    def _load_class_mapping(self):
        """Load class mapping from JSON file."""
        try:
            with open(self.class_mapping_file, 'r') as f:
                mapping = json.load(f)
            
            # Handle different mapping formats
            if isinstance(mapping, dict):
                if 'class_names' in mapping:
                    self.class_names = mapping['class_names']
                elif 'classes' in mapping:
                    self.class_names = mapping['classes']
                else:
                    # Assume it's a direct class_name -> index mapping
                    self.class_names = list(mapping.keys())
            elif isinstance(mapping, list):
                self.class_names = mapping
            else:
                raise ValidationError("Invalid class mapping format")
            
            # Create mappings
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
            self.num_classes = len(self.class_names)
            
            self.logger.info(f"Loaded class mapping: {self.num_classes} classes")
            
        except Exception as e:
            raise DatasetError(f"Failed to load class mapping: {str(e)}")
    
    def _discover_classes(self):
        """Discover classes from directory structure."""
        try:
            # Look for subdirectories as classes
            class_dirs = [d for d in self.dataset_path.iterdir() 
                         if d.is_dir() and not d.name.startswith('.')]
            
            if not class_dirs:
                raise DatasetError("No class directories found in dataset path")
            
            # Sort class names for consistent ordering
            self.class_names = sorted([d.name for d in class_dirs])
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
            self.num_classes = len(self.class_names)
            
            self.logger.info(f"Discovered {self.num_classes} classes from directory structure")
            
        except Exception as e:
            raise DatasetError(f"Failed to discover classes: {str(e)}")
    
    def _load_image_paths(self):
        """Load all image paths and corresponding labels."""
        try:
            self.image_paths = []
            self.labels = []
            
            # Supported image extensions
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            
            for class_name in self.class_names:
                class_dir = self.dataset_path / class_name
                
                if not class_dir.exists():
                    self.logger.warning(f"Class directory not found: {class_dir}")
                    continue
                
                class_idx = self.class_to_idx[class_name]
                class_images = []
                
                # Find all image files in class directory
                for file_path in class_dir.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                        class_images.append(str(file_path))
                
                # Add to main lists
                self.image_paths.extend(class_images)
                self.labels.extend([class_idx] * len(class_images))
                
                self.logger.debug(f"Class '{class_name}': {len(class_images)} images")
            
            if not self.image_paths:
                raise DatasetError("No valid image files found in dataset")
            
            self.logger.info(f"Loaded {len(self.image_paths)} image paths")
            
        except Exception as e:
            raise DatasetError(f"Failed to load image paths: {str(e)}")
    
    def _calculate_statistics(self):
        """Calculate dataset statistics."""
        try:
            # Class distribution
            self.class_distribution = {}
            for class_name in self.class_names:
                class_idx = self.class_to_idx[class_name]
                count = self.labels.count(class_idx)
                self.class_distribution[class_name] = count
            
            # Overall statistics
            total_images = len(self.image_paths)
            avg_per_class = total_images / self.num_classes
            min_count = min(self.class_distribution.values())
            max_count = max(self.class_distribution.values())
            
            self.dataset_stats = {
                'total_images': total_images,
                'num_classes': self.num_classes,
                'avg_images_per_class': avg_per_class,
                'min_images_per_class': min_count,
                'max_images_per_class': max_count,
                'class_imbalance_ratio': max_count / min_count if min_count > 0 else float('inf'),
                'std_images_per_class': np.std(list(self.class_distribution.values()))
            }
            
            self.logger.info(f"Dataset statistics calculated: {total_images} total images")
            
        except Exception as e:
            raise DatasetError(f"Failed to calculate statistics: {str(e)}")
    
    def split_dataset(self,
                     split: DatasetSplit,
                     stratify: bool = True) -> Dict[str, Tuple[List[str], List[int]]]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            split: DatasetSplit configuration
            stratify: Whether to maintain class distribution in splits
            
        Returns:
            Dictionary with 'train', 'validation', 'test' keys containing
            (image_paths, labels) tuples
        """
        try:
            self.logger.info("Splitting dataset...")
            
            # Prepare data
            X = np.array(self.image_paths)
            y = np.array(self.labels)
            
            if stratify and len(np.unique(y)) > 1:
                stratify_param = y
            else:
                stratify_param = None
            
            # First split: separate test set
            test_size = split.test_size
            train_val_size = split.train_size + split.validation_size
            
            if test_size > 0:
                X_train_val, X_test, y_train_val, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=split.random_state,
                    stratify=stratify_param
                )
            else:
                X_train_val, X_test = X, np.array([])
                y_train_val, y_test = y, np.array([])
            
            # Second split: separate train and validation
            if split.validation_size > 0 and len(X_train_val) > 0:
                val_size_relative = split.validation_size / train_val_size
                
                if stratify and len(np.unique(y_train_val)) > 1:
                    stratify_param_val = y_train_val
                else:
                    stratify_param_val = None
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val, y_train_val,
                    test_size=val_size_relative,
                    random_state=split.random_state,
                    stratify=stratify_param_val
                )
            else:
                X_train, X_val = X_train_val, np.array([])
                y_train, y_val = y_train_val, np.array([])
            
            # Create result dictionary
            splits = {
                'train': (X_train.tolist(), y_train.tolist()),
                'validation': (X_val.tolist(), y_val.tolist()),
                'test': (X_test.tolist(), y_test.tolist())
            }
            
            # Log split information
            for split_name, (paths, labels) in splits.items():
                self.logger.info(f"{split_name.capitalize()} set: {len(paths)} images")
            
            return splits
            
        except Exception as e:
            raise DatasetError(f"Failed to split dataset: {str(e)}")
    
    def create_tf_dataset(self,
                         image_paths: List[str],
                         labels: List[int],
                         batch_size: int = 32,
                         shuffle: bool = True,
                         repeat: bool = False,
                         augment: bool = False) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from image paths and labels.
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            batch_size: Batch size for the dataset
            shuffle: Whether to shuffle the dataset
            repeat: Whether to repeat the dataset infinitely
            augment: Whether to apply data augmentation
            
        Returns:
            TensorFlow dataset
        """
        try:
            if not image_paths:
                raise ValidationError("Empty image paths list")
            
            # Create dataset from paths and labels
            dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
            
            # Shuffle if requested
            if shuffle:
                dataset = dataset.shuffle(buffer_size=len(image_paths))
            
            # Map function to load and preprocess images
            dataset = dataset.map(
                lambda path, label: self._load_and_preprocess_image(path, label),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            # Apply augmentation if requested
            if augment:
                dataset = dataset.map(
                    self._augment_image,
                    num_parallel_calls=tf.data.AUTOTUNE
                )
            
            # Batch the dataset
            dataset = dataset.batch(batch_size)
            
            # Repeat if requested
            if repeat:
                dataset = dataset.repeat()
            
            # Prefetch for better performance
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            self.logger.info(f"Created TensorFlow dataset: {len(image_paths)} samples, "
                           f"batch_size={batch_size}")
            
            return dataset
            
        except Exception as e:
            raise DatasetError(f"Failed to create TensorFlow dataset: {str(e)}")
    
    def _load_and_preprocess_image(self, image_path: tf.Tensor, label: tf.Tensor):
        """Load and preprocess a single image."""
        # Read image file
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        
        # Resize to target size
        target_size = self.settings.IMAGE_SIZE
        image = tf.image.resize(image, [target_size, target_size])
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        # Convert label to one-hot if needed
        if self.settings.USE_ONE_HOT_LABELS:
            label = tf.one_hot(label, self.num_classes)
        
        return image, label
    
    def _augment_image(self, image: tf.Tensor, label: tf.Tensor):
        """Apply data augmentation to image."""
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        
        # Random rotation
        image = tf.image.rot90(image, tf.random.uniform([], 0, 4, dtype=tf.int32))
        
        # Random brightness and contrast
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        
        # Random saturation
        image = tf.image.random_saturation(image, 0.9, 1.1)
        
        # Ensure values stay in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    def get_class_weights(self) -> Dict[int, float]:
        """
        Calculate class weights for handling imbalanced datasets.
        
        Returns:
            Dictionary mapping class indices to weights
        """
        try:
            # Count samples per class
            class_counts = np.bincount(self.labels, minlength=self.num_classes)
            
            # Calculate weights (inverse frequency)
            total_samples = len(self.labels)
            weights = total_samples / (self.num_classes * class_counts + 1e-8)
            
            # Create class weight dictionary
            class_weights = {i: float(weight) for i, weight in enumerate(weights)}
            
            self.logger.info("Calculated class weights for imbalanced dataset")
            return class_weights
            
        except Exception as e:
            raise DatasetError(f"Failed to calculate class weights: {str(e)}")
    
    def export_dataset_info(self, output_file: str):
        """
        Export dataset information to JSON file.
        
        Args:
            output_file: Path to output JSON file
        """
        try:
            dataset_info = {
                'dataset_path': str(self.dataset_path),
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'class_to_idx': self.class_to_idx,
                'dataset_stats': self.dataset_stats,
                'class_distribution': self.class_distribution,
                'total_images': len(self.image_paths)
            }
            
            with open(output_file, 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            self.logger.info(f"Dataset information exported to: {output_file}")
            
        except Exception as e:
            raise DatasetError(f"Failed to export dataset info: {str(e)}")
    
    def validate_dataset(self) -> Dict[str, Any]:
        """
        Validate dataset integrity and return validation report.
        
        Returns:
            Dictionary containing validation results
        """
        try:
            self.logger.info("Validating dataset...")
            
            validation_results = {
                'valid': True,
                'issues': [],
                'warnings': [],
                'statistics': {}
            }
            
            # Check for missing files
            missing_files = []
            for path in self.image_paths:
                if not os.path.exists(path):
                    missing_files.append(path)
            
            if missing_files:
                validation_results['valid'] = False
                validation_results['issues'].append(f"Missing files: {len(missing_files)}")
            
            # Check class distribution
            min_samples = min(self.class_distribution.values())
            max_samples = max(self.class_distribution.values())
            imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
            
            if imbalance_ratio > 10:
                validation_results['warnings'].append(
                    f"High class imbalance detected (ratio: {imbalance_ratio:.2f})"
                )
            
            # Check minimum samples per class
            classes_with_few_samples = [
                name for name, count in self.class_distribution.items() 
                if count < 10
            ]
            
            if classes_with_few_samples:
                validation_results['warnings'].append(
                    f"Classes with few samples (<10): {classes_with_few_samples}"
                )
            
            # Add statistics
            validation_results['statistics'] = self.dataset_stats.copy()
            validation_results['statistics']['missing_files'] = len(missing_files)
            validation_results['statistics']['imbalance_ratio'] = imbalance_ratio
            
            if validation_results['valid']:
                self.logger.info("Dataset validation passed")
            else:
                self.logger.warning(f"Dataset validation failed with {len(validation_results['issues'])} issues")
            
            return validation_results
            
        except Exception as e:
            raise DatasetError(f"Dataset validation failed: {str(e)}")
    
    def get_sample_images(self, 
                         num_samples: int = 10,
                         per_class: bool = True) -> Dict[str, List[str]]:
        """
        Get sample images for visualization or testing.
        
        Args:
            num_samples: Number of samples to return
            per_class: If True, return samples per class; if False, return random samples
            
        Returns:
            Dictionary with class names as keys and lists of image paths as values
        """
        try:
            if per_class:
                samples = {}
                samples_per_class = max(1, num_samples // self.num_classes)
                
                for class_name in self.class_names:
                    class_idx = self.class_to_idx[class_name]
                    class_paths = [
                        path for path, label in zip(self.image_paths, self.labels)
                        if label == class_idx
                    ]
                    
                    # Randomly sample from class
                    num_to_sample = min(samples_per_class, len(class_paths))
                    samples[class_name] = random.sample(class_paths, num_to_sample)
            else:
                # Random samples across all classes
                random_indices = random.sample(range(len(self.image_paths)), 
                                             min(num_samples, len(self.image_paths)))
                
                samples = {'all_classes': [self.image_paths[i] for i in random_indices]}
            
            return samples
            
        except Exception as e:
            raise DatasetError(f"Failed to get sample images: {str(e)}")
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        return {
            'dataset_path': str(self.dataset_path),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'total_images': len(self.image_paths),
            'class_distribution': {name: len([p for p in self.image_paths if name in str(p)]) 
                                 for name in self.class_names} if self.class_names else {},
            'loaded': len(self.image_paths) > 0
        }
    
    def __repr__(self) -> str:
        """String representation of the dataset manager."""
        return (f"DatasetManager(path='{self.dataset_path}', "
                f"classes={self.num_classes}, images={len(self.image_paths)})")