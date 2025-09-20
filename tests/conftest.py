"""
Test Configuration and Utilities
===============================

Common utilities and configurations for all tests.
"""

import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

class TestConfig:
    """Test configuration constants"""
    
    # Sample dimensions
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224
    IMAGE_CHANNELS = 3
    NUM_CLASSES = 5  # Small number for testing
    BATCH_SIZE = 4
    
    # Sample class names
    CLASS_NAMES = ['test_fish_1', 'test_fish_2', 'test_fish_3', 'test_fish_4', 'test_fish_5']
    
    # Test dataset sizes
    TRAIN_SIZE = 20
    VAL_SIZE = 10
    TEST_SIZE = 10

def create_sample_image_data(num_samples: int, height: int = 224, width: int = 224, 
                           channels: int = 3) -> np.ndarray:
    """
    Create sample image data for testing
    
    Args:
        num_samples: Number of sample images
        height: Image height
        width: Image width
        channels: Number of channels
        
    Returns:
        Sample image array
    """
    return np.random.rand(num_samples, height, width, channels).astype(np.float32)

def create_sample_labels(num_samples: int, num_classes: int) -> np.ndarray:
    """
    Create sample labels for testing
    
    Args:
        num_samples: Number of samples
        num_classes: Number of classes
        
    Returns:
        Sample labels array
    """
    return np.random.randint(0, num_classes, size=num_samples)

def create_sample_probabilities(num_samples: int, num_classes: int) -> np.ndarray:
    """
    Create sample probability predictions for testing
    
    Args:
        num_samples: Number of samples
        num_classes: Number of classes
        
    Returns:
        Sample probabilities array
    """
    probs = np.random.rand(num_samples, num_classes)
    # Normalize to make valid probabilities
    return probs / probs.sum(axis=1, keepdims=True)

def create_temp_directory() -> Path:
    """
    Create temporary directory for test files
    
    Returns:
        Path to temporary directory
    """
    return Path(tempfile.mkdtemp())

def cleanup_temp_directory(temp_dir: Path) -> None:
    """
    Clean up temporary directory
    
    Args:
        temp_dir: Path to temporary directory
    """
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

def create_sample_config() -> Dict[str, Any]:
    """
    Create sample configuration for testing
    
    Returns:
        Sample configuration dictionary
    """
    return {
        'model': {
            'architecture': 'simple_cnn',
            'input_shape': [224, 224, 3],
            'num_classes': 5,
            'learning_rate': 0.001
        },
        'training': {
            'batch_size': 4,
            'epochs': 2,
            'validation_split': 0.2
        },
        'data': {
            'augmentation': True,
            'normalize': True
        }
    }

def create_sample_dataset_structure(base_dir: Path, class_names: List[str], 
                                  samples_per_class: int = 5) -> None:
    """
    Create sample dataset directory structure
    
    Args:
        base_dir: Base directory for dataset
        class_names: List of class names
        samples_per_class: Number of samples per class
    """
    for split in ['train', 'val', 'test']:
        split_dir = base_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for class_name in class_names:
            class_dir = split_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Create dummy image files
            for i in range(samples_per_class):
                dummy_file = class_dir / f"{class_name}_{i:03d}.jpg"
                dummy_file.touch()

def create_sample_model_info() -> Dict[str, Any]:
    """
    Create sample model information for testing
    
    Returns:
        Sample model info dictionary
    """
    return {
        'model_name': 'test_model',
        'architecture': 'simple_cnn',
        'parameters': 100000,
        'training_time': 120.5,
        'inference_time': 0.05,
        'model_size_mb': 2.5,
        'version': '1.0.0',
        'created_at': '2024-01-01T00:00:00Z'
    }

class MockDataGenerator:
    """Mock data generator for testing"""
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, batch_size: int = 4):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.steps_per_epoch = len(x_data) // batch_size
    
    def __len__(self):
        return self.steps_per_epoch
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.x_data))
        
        return (
            self.x_data[start_idx:end_idx],
            self.y_data[start_idx:end_idx]
        )

def assert_array_properties(array: np.ndarray, expected_shape: Tuple, 
                          expected_dtype: np.dtype = None, 
                          min_val: float = None, max_val: float = None) -> None:
    """
    Assert array has expected properties
    
    Args:
        array: Array to check
        expected_shape: Expected shape
        expected_dtype: Expected data type
        min_val: Expected minimum value
        max_val: Expected maximum value
    """
    assert array.shape == expected_shape, f"Expected shape {expected_shape}, got {array.shape}"
    
    if expected_dtype is not None:
        assert array.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {array.dtype}"
    
    if min_val is not None:
        assert np.min(array) >= min_val, f"Min value {np.min(array)} below expected {min_val}"
    
    if max_val is not None:
        assert np.max(array) <= max_val, f"Max value {np.max(array)} above expected {max_val}"

def save_test_results(results: Dict[str, Any], filename: str) -> None:
    """
    Save test results to file
    
    Args:
        results: Results dictionary
        filename: Output filename
    """
    from tests import TEST_RESULTS_DIR
    
    output_path = TEST_RESULTS_DIR / filename
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def load_test_fixture(filename: str) -> Any:
    """
    Load test fixture data
    
    Args:
        filename: Fixture filename
        
    Returns:
        Loaded fixture data
    """
    from tests import TEST_DATA_DIR
    
    fixture_path = TEST_DATA_DIR / filename
    if fixture_path.suffix == '.json':
        with open(fixture_path, 'r') as f:
            return json.load(f)
    elif fixture_path.suffix == '.npy':
        return np.load(fixture_path)
    else:
        with open(fixture_path, 'r') as f:
            return f.read()