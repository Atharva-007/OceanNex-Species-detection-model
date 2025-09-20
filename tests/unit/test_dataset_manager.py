"""
Unit Tests for Data Management
=============================

Tests for dataset manager and data processing utilities.
"""

import unittest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset_manager import DatasetManager
from tests.conftest import (
    TestConfig, create_temp_directory, cleanup_temp_directory,
    create_sample_config, create_sample_image_data, create_sample_labels,
    create_sample_dataset_structure, MockDataGenerator
)

class TestDatasetManager(unittest.TestCase):
    """Test DatasetManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = create_temp_directory()
        self.dataset_dir = self.temp_dir / "test_dataset"
        self.dataset_manager = DatasetManager()
        
        # Create sample dataset structure
        create_sample_dataset_structure(
            self.dataset_dir, 
            TestConfig.CLASS_NAMES, 
            samples_per_class=5
        )
        
    def tearDown(self):
        """Clean up test fixtures"""
        cleanup_temp_directory(self.temp_dir)
    
    def test_dataset_manager_initialization(self):
        """Test dataset manager initialization"""
        self.assertIsNotNone(self.dataset_manager)
        self.assertIsNotNone(self.dataset_manager.config_manager)
    
    def test_scan_dataset_directory(self):
        """Test scanning dataset directory structure"""
        info = self.dataset_manager.scan_dataset_directory(str(self.dataset_dir))
        
        self.assertIn('classes', info)
        self.assertIn('splits', info)
        self.assertIn('total_samples', info)
        
        # Check classes detected
        for class_name in TestConfig.CLASS_NAMES:
            self.assertIn(class_name, info['classes'])
        
        # Check splits detected
        self.assertIn('train', info['splits'])
        self.assertIn('val', info['splits'])
        self.assertIn('test', info['splits'])
    
    def test_get_dataset_statistics(self):
        """Test getting dataset statistics"""
        stats = self.dataset_manager.get_dataset_statistics(str(self.dataset_dir))
        
        self.assertIn('class_distribution', stats)
        self.assertIn('split_distribution', stats)
        self.assertIn('total_samples', stats)
        self.assertIn('num_classes', stats)
        
        # Check statistics values
        self.assertEqual(stats['num_classes'], len(TestConfig.CLASS_NAMES))
        self.assertGreater(stats['total_samples'], 0)
    
    @patch('tensorflow.keras.utils.image_dataset_from_directory')
    def test_create_tensorflow_dataset(self, mock_dataset_from_directory):
        """Test creating TensorFlow dataset"""
        mock_dataset = Mock()
        mock_dataset_from_directory.return_value = mock_dataset
        
        config = {
            'batch_size': TestConfig.BATCH_SIZE,
            'image_size': (TestConfig.IMAGE_HEIGHT, TestConfig.IMAGE_WIDTH),
            'validation_split': 0.2
        }
        
        train_ds, val_ds = self.dataset_manager.create_tensorflow_dataset(
            str(self.dataset_dir / 'train'), config
        )
        
        self.assertEqual(train_ds, mock_dataset)
        mock_dataset_from_directory.assert_called()
    
    def test_load_and_preprocess_images(self):
        """Test loading and preprocessing images"""
        # Create sample image files with actual image data
        sample_images = create_sample_image_data(
            TestConfig.TRAIN_SIZE, 
            TestConfig.IMAGE_HEIGHT, 
            TestConfig.IMAGE_WIDTH, 
            TestConfig.IMAGE_CHANNELS
        )
        
        # Mock image loading
        with patch('tensorflow.keras.utils.load_img') as mock_load_img, \
             patch('tensorflow.keras.utils.img_to_array') as mock_img_to_array:
            
            mock_load_img.return_value = Mock()
            mock_img_to_array.return_value = sample_images[0]
            
            image_paths = [f"test_image_{i}.jpg" for i in range(5)]
            
            processed_images = self.dataset_manager.load_and_preprocess_images(
                image_paths, target_size=(224, 224)
            )
            
            self.assertEqual(len(processed_images), 5)
            self.assertEqual(processed_images[0].shape, (224, 224, 3))
    
    def test_apply_data_augmentation(self):
        """Test applying data augmentation"""
        sample_data = create_sample_image_data(5)
        
        # Mock ImageDataGenerator
        with patch('tensorflow.keras.preprocessing.image.ImageDataGenerator') as mock_idg:
            mock_generator = Mock()
            mock_idg.return_value = mock_generator
            mock_generator.flow.return_value = MockDataGenerator(sample_data, np.zeros(5))
            
            augmented_generator = self.dataset_manager.apply_data_augmentation(
                sample_data, np.zeros(5)
            )
            
            self.assertIsNotNone(augmented_generator)
            mock_idg.assert_called_once()
    
    def test_normalize_images(self):
        """Test image normalization"""
        # Create sample images with values 0-255
        sample_images = np.random.randint(0, 256, size=(5, 224, 224, 3)).astype(np.uint8)
        
        normalized = self.dataset_manager.normalize_images(sample_images)
        
        # Should be normalized to 0-1 range
        self.assertGreaterEqual(np.min(normalized), 0.0)
        self.assertLessEqual(np.max(normalized), 1.0)
        self.assertEqual(normalized.dtype, np.float32)
    
    def test_create_class_mapping(self):
        """Test creating class name to index mapping"""
        class_names = TestConfig.CLASS_NAMES
        
        mapping = self.dataset_manager.create_class_mapping(class_names)
        
        self.assertEqual(len(mapping), len(class_names))
        for i, class_name in enumerate(class_names):
            self.assertEqual(mapping[class_name], i)
    
    def test_split_dataset(self):
        """Test splitting dataset into train/validation sets"""
        sample_data = create_sample_image_data(100)
        sample_labels = create_sample_labels(100, TestConfig.NUM_CLASSES)
        
        train_data, val_data, train_labels, val_labels = self.dataset_manager.split_dataset(
            sample_data, sample_labels, validation_split=0.2, random_state=42
        )
        
        # Check split proportions
        self.assertEqual(len(train_data), 80)
        self.assertEqual(len(val_data), 20)
        self.assertEqual(len(train_labels), 80)
        self.assertEqual(len(val_labels), 20)
        
        # Check data shapes
        self.assertEqual(train_data.shape[1:], sample_data.shape[1:])
        self.assertEqual(val_data.shape[1:], sample_data.shape[1:])
    
    def test_balance_dataset(self):
        """Test dataset balancing functionality"""
        # Create imbalanced dataset
        imbalanced_labels = np.concatenate([
            np.zeros(50),  # 50 samples of class 0
            np.ones(20),   # 20 samples of class 1
            np.full(10, 2)  # 10 samples of class 2
        ]).astype(int)
        
        imbalanced_data = create_sample_image_data(len(imbalanced_labels))
        
        balanced_data, balanced_labels = self.dataset_manager.balance_dataset(
            imbalanced_data, imbalanced_labels, strategy='oversample'
        )
        
        # Check that classes are more balanced
        unique, counts = np.unique(balanced_labels, return_counts=True)
        
        # All classes should have the same count (oversampled to majority class)
        max_count = np.max(counts)
        for count in counts:
            self.assertEqual(count, max_count)
    
    def test_get_data_generators(self):
        """Test creating data generators"""
        train_dir = self.dataset_dir / 'train'
        val_dir = self.dataset_dir / 'val'
        
        with patch('tensorflow.keras.preprocessing.image.ImageDataGenerator.flow_from_directory') as mock_flow:
            mock_generator = Mock()
            mock_generator.samples = 50
            mock_generator.batch_size = TestConfig.BATCH_SIZE
            mock_flow.return_value = mock_generator
            
            train_gen, val_gen = self.dataset_manager.get_data_generators(
                str(train_dir), str(val_dir), TestConfig.BATCH_SIZE
            )
            
            self.assertEqual(train_gen, mock_generator)
            self.assertEqual(val_gen, mock_generator)
    
    def test_calculate_dataset_mean_std(self):
        """Test calculating dataset mean and standard deviation"""
        sample_data = create_sample_image_data(20)
        
        mean, std = self.dataset_manager.calculate_dataset_mean_std(sample_data)
        
        self.assertEqual(mean.shape, (TestConfig.IMAGE_CHANNELS,))
        self.assertEqual(std.shape, (TestConfig.IMAGE_CHANNELS,))
        self.assertGreater(np.min(std), 0)  # Standard deviation should be positive
    
    def test_validate_dataset_structure(self):
        """Test dataset structure validation"""
        # Valid structure should pass
        is_valid, issues = self.dataset_manager.validate_dataset_structure(str(self.dataset_dir))
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
        
        # Invalid structure
        invalid_dir = self.temp_dir / "invalid_dataset"
        invalid_dir.mkdir()
        
        is_valid, issues = self.dataset_manager.validate_dataset_structure(str(invalid_dir))
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
    
    def test_get_sample_images(self):
        """Test getting sample images from dataset"""
        samples = self.dataset_manager.get_sample_images(
            str(self.dataset_dir / 'train'), 
            samples_per_class=2
        )
        
        self.assertIsInstance(samples, dict)
        for class_name in TestConfig.CLASS_NAMES:
            if class_name in samples:
                self.assertLessEqual(len(samples[class_name]), 2)
    
    def test_export_dataset_info(self):
        """Test exporting dataset information"""
        output_file = self.temp_dir / "dataset_info.json"
        
        self.dataset_manager.export_dataset_info(
            str(self.dataset_dir), str(output_file)
        )
        
        self.assertTrue(output_file.exists())
        
        # Check content
        import json
        with open(output_file, 'r') as f:
            info = json.load(f)
        
        self.assertIn('classes', info)
        self.assertIn('statistics', info)
        self.assertIn('structure', info)

class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.dataset_manager = DatasetManager()
    
    def test_resize_images(self):
        """Test image resizing"""
        original_images = create_sample_image_data(5, 512, 512, 3)
        
        resized = self.dataset_manager.resize_images(
            original_images, target_size=(224, 224)
        )
        
        self.assertEqual(resized.shape, (5, 224, 224, 3))
    
    def test_convert_labels_to_categorical(self):
        """Test converting labels to categorical format"""
        labels = np.array([0, 1, 2, 1, 0])
        num_classes = 3
        
        categorical = self.dataset_manager.convert_labels_to_categorical(labels, num_classes)
        
        self.assertEqual(categorical.shape, (5, 3))
        self.assertEqual(np.sum(categorical[0]), 1.0)  # One-hot encoded
        self.assertEqual(np.argmax(categorical[0]), 0)  # First label should be class 0
    
    def test_apply_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline"""
        sample_images = np.random.randint(0, 256, size=(10, 256, 256, 3)).astype(np.uint8)
        sample_labels = create_sample_labels(10, TestConfig.NUM_CLASSES)
        
        preprocessed_images, preprocessed_labels = self.dataset_manager.apply_preprocessing_pipeline(
            sample_images, sample_labels, target_size=(224, 224), normalize=True, categorical=True
        )
        
        # Check image preprocessing
        self.assertEqual(preprocessed_images.shape, (10, 224, 224, 3))
        self.assertGreaterEqual(np.min(preprocessed_images), 0.0)
        self.assertLessEqual(np.max(preprocessed_images), 1.0)
        
        # Check label preprocessing
        self.assertEqual(preprocessed_labels.shape, (10, TestConfig.NUM_CLASSES))

if __name__ == '__main__':
    unittest.main()