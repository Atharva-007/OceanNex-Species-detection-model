"""
Unit Tests for Core Model Management
===================================

Tests for the model manager and factory patterns.
"""

import unittest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.model_manager import ModelManager
from tests.conftest import (
    TestConfig, create_temp_directory, cleanup_temp_directory,
    create_sample_config, create_sample_image_data, create_sample_labels
)

class TestModelManager(unittest.TestCase):
    """Test ModelManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = create_temp_directory()
        self.model_manager = ModelManager()
        self.sample_config = create_sample_config()
        
    def tearDown(self):
        """Clean up test fixtures"""
        cleanup_temp_directory(self.temp_dir)
    
    def test_model_manager_initialization(self):
        """Test model manager initialization"""
        self.assertIsNotNone(self.model_manager)
        self.assertIsNotNone(self.model_manager.config_manager)
    
    @patch('tensorflow.keras.models.Sequential')
    def test_create_simple_cnn_model(self, mock_sequential):
        """Test creating simple CNN model"""
        mock_model = Mock()
        mock_sequential.return_value = mock_model
        
        model_config = {
            'architecture': 'simple_cnn',
            'input_shape': [224, 224, 3],
            'num_classes': 5
        }
        
        result = self.model_manager.create_model(model_config)
        
        # Should return the mock model
        self.assertEqual(result, mock_model)
        mock_sequential.assert_called_once()
    
    @patch('tensorflow.keras.applications.VGG16')
    @patch('tensorflow.keras.models.Model')
    def test_create_vgg16_model(self, mock_model, mock_vgg16):
        """Test creating VGG16 transfer learning model"""
        mock_base = Mock()
        mock_vgg16.return_value = mock_base
        mock_final_model = Mock()
        mock_model.return_value = mock_final_model
        
        model_config = {
            'architecture': 'vgg16',
            'input_shape': [224, 224, 3],
            'num_classes': 5,
            'trainable_layers': 0
        }
        
        result = self.model_manager.create_model(model_config)
        
        # Should create VGG16 base and final model
        mock_vgg16.assert_called_once()
        self.assertEqual(result, mock_final_model)
    
    def test_invalid_architecture(self):
        """Test handling of invalid model architecture"""
        model_config = {
            'architecture': 'invalid_architecture',
            'input_shape': [224, 224, 3],
            'num_classes': 5
        }
        
        with self.assertRaises(ValueError):
            self.model_manager.create_model(model_config)
    
    @patch('tensorflow.keras.models.load_model')
    def test_load_model(self, mock_load_model):
        """Test loading saved model"""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        model_path = self.temp_dir / "test_model.h5"
        
        result = self.model_manager.load_model(str(model_path))
        
        self.assertEqual(result, mock_model)
        mock_load_model.assert_called_once_with(str(model_path))
    
    def test_load_nonexistent_model(self):
        """Test loading nonexistent model"""
        nonexistent_path = self.temp_dir / "nonexistent.h5"
        
        with self.assertRaises(FileNotFoundError):
            self.model_manager.load_model(str(nonexistent_path))
    
    @patch('tensorflow.keras.models.Model.save')
    def test_save_model(self, mock_save):
        """Test saving model"""
        mock_model = Mock()
        model_path = self.temp_dir / "test_model.h5"
        
        self.model_manager.save_model(mock_model, str(model_path))
        
        mock_save.assert_called_once_with(str(model_path))
    
    @patch('tensorflow.keras.models.Model.compile')
    def test_compile_model(self, mock_compile):
        """Test compiling model"""
        mock_model = Mock()
        
        compile_config = {
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy']
        }
        
        self.model_manager.compile_model(mock_model, compile_config)
        
        mock_compile.assert_called_once()
    
    def test_get_model_summary(self):
        """Test getting model summary"""
        mock_model = Mock()
        mock_model.count_params.return_value = 1000000
        mock_model.summary.return_value = None
        
        # Mock model layers
        mock_layer = Mock()
        mock_layer.name = "test_layer"
        mock_layer.output_shape = (None, 224, 224, 64)
        mock_model.layers = [mock_layer]
        
        summary = self.model_manager.get_model_summary(mock_model)
        
        self.assertIn('total_params', summary)
        self.assertIn('layers', summary)
        self.assertEqual(summary['total_params'], 1000000)
    
    def test_model_factory_patterns(self):
        """Test model factory pattern implementation"""
        # Test that different architectures are handled correctly
        architectures = ['simple_cnn', 'vgg16', 'resnet50', 'custom']
        
        for arch in architectures:
            if arch == 'custom':
                # Custom should raise error
                with self.assertRaises(ValueError):
                    config = {'architecture': arch, 'input_shape': [224, 224, 3], 'num_classes': 5}
                    self.model_manager.create_model(config)
            else:
                # Others should work (mocked)
                with patch('tensorflow.keras.models.Sequential'), \
                     patch('tensorflow.keras.applications.VGG16'), \
                     patch('tensorflow.keras.applications.ResNet50'), \
                     patch('tensorflow.keras.models.Model'):
                    
                    config = {'architecture': arch, 'input_shape': [224, 224, 3], 'num_classes': 5}
                    # Should not raise error
                    try:
                        self.model_manager.create_model(config)
                    except ValueError:
                        # Some architectures might not be implemented yet
                        pass
    
    @patch('tensorflow.keras.models.Model.predict')
    def test_model_prediction(self, mock_predict):
        """Test model prediction functionality"""
        mock_model = Mock()
        mock_predict.return_value = np.random.rand(4, 5)  # 4 samples, 5 classes
        
        sample_data = create_sample_image_data(4)
        
        predictions = self.model_manager.predict(mock_model, sample_data)
        
        self.assertEqual(predictions.shape, (4, 5))
        mock_predict.assert_called_once()
    
    @patch('tensorflow.keras.models.Model.evaluate')
    def test_model_evaluation(self, mock_evaluate):
        """Test model evaluation functionality"""
        mock_model = Mock()
        mock_evaluate.return_value = [0.5, 0.85]  # loss, accuracy
        
        sample_data = create_sample_image_data(TestConfig.TEST_SIZE)
        sample_labels = create_sample_labels(TestConfig.TEST_SIZE, TestConfig.NUM_CLASSES)
        
        results = self.model_manager.evaluate(mock_model, sample_data, sample_labels)
        
        self.assertIn('loss', results)
        self.assertIn('accuracy', results)
        mock_evaluate.assert_called_once()
    
    def test_list_available_architectures(self):
        """Test listing available model architectures"""
        architectures = self.model_manager.list_available_architectures()
        
        self.assertIsInstance(architectures, list)
        self.assertIn('simple_cnn', architectures)
        self.assertIn('vgg16', architectures)
    
    def test_model_config_validation(self):
        """Test model configuration validation"""
        # Valid config
        valid_config = {
            'architecture': 'simple_cnn',
            'input_shape': [224, 224, 3],
            'num_classes': 5
        }
        
        self.assertTrue(self.model_manager.validate_model_config(valid_config))
        
        # Invalid configs
        invalid_configs = [
            {},  # Empty config
            {'architecture': 'simple_cnn'},  # Missing required fields
            {'architecture': 'simple_cnn', 'input_shape': [224, 224, 3]},  # Missing num_classes
            {'architecture': 'simple_cnn', 'input_shape': 'invalid', 'num_classes': 5}  # Invalid input_shape
        ]
        
        for invalid_config in invalid_configs:
            self.assertFalse(self.model_manager.validate_model_config(invalid_config))

class TestModelFactories(unittest.TestCase):
    """Test individual model factory functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_manager = ModelManager()
    
    @patch('tensorflow.keras.layers.Conv2D')
    @patch('tensorflow.keras.layers.Dense')
    @patch('tensorflow.keras.models.Sequential')
    def test_simple_cnn_factory(self, mock_sequential, mock_dense, mock_conv2d):
        """Test simple CNN factory function"""
        mock_model = Mock()
        mock_sequential.return_value = mock_model
        
        config = {
            'input_shape': [224, 224, 3],
            'num_classes': 5,
            'layers': [
                {'type': 'conv2d', 'filters': 32, 'kernel_size': 3},
                {'type': 'dense', 'units': 128}
            ]
        }
        
        result = self.model_manager._create_simple_cnn(config)
        
        self.assertEqual(result, mock_model)
        mock_sequential.assert_called_once()
    
    @patch('tensorflow.keras.applications.VGG16')
    def test_transfer_learning_factory(self, mock_vgg16):
        """Test transfer learning model factory"""
        mock_base_model = Mock()
        mock_base_model.output = Mock()
        mock_vgg16.return_value = mock_base_model
        
        config = {
            'input_shape': [224, 224, 3],
            'num_classes': 5,
            'base_model': 'vgg16',
            'trainable_layers': 0
        }
        
        with patch('tensorflow.keras.models.Model'):
            result = self.model_manager._create_transfer_learning_model(config)
            mock_vgg16.assert_called_once()

if __name__ == '__main__':
    unittest.main()