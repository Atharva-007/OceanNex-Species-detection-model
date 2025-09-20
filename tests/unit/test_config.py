"""
Unit Tests for Configuration Management
=====================================

Tests for the centralized configuration system.
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, mock_open

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import ConfigManager
from tests.conftest import create_temp_directory, cleanup_temp_directory, create_sample_config

class TestConfigManager(unittest.TestCase):
    """Test ConfigManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = create_temp_directory()
        self.config_file = self.temp_dir / "test_config.json"
        self.sample_config = create_sample_config()
        
    def tearDown(self):
        """Clean up test fixtures"""
        cleanup_temp_directory(self.temp_dir)
    
    def test_config_initialization(self):
        """Test configuration manager initialization"""
        config_manager = ConfigManager()
        
        # Should have default config
        self.assertIsNotNone(config_manager.config)
        self.assertIn('model', config_manager.config)
        self.assertIn('training', config_manager.config)
        self.assertIn('data', config_manager.config)
    
    def test_load_config_from_file(self):
        """Test loading configuration from file"""
        # Write sample config to file
        with open(self.config_file, 'w') as f:
            json.dump(self.sample_config, f)
        
        config_manager = ConfigManager(config_file=str(self.config_file))
        
        # Should load the sample config
        self.assertEqual(config_manager.config['model']['architecture'], 'simple_cnn')
        self.assertEqual(config_manager.config['training']['batch_size'], 4)
    
    def test_save_config(self):
        """Test saving configuration to file"""
        config_manager = ConfigManager()
        config_manager.update_config({'test_key': 'test_value'})
        
        # Save to temp file
        config_manager.save_config(str(self.config_file))
        
        # Verify file was created and contains data
        self.assertTrue(self.config_file.exists())
        
        with open(self.config_file, 'r') as f:
            saved_config = json.load(f)
        
        self.assertEqual(saved_config['test_key'], 'test_value')
    
    def test_get_config_value(self):
        """Test getting configuration values"""
        config_manager = ConfigManager()
        config_manager.config = self.sample_config
        
        # Test nested key access
        self.assertEqual(config_manager.get('model.architecture'), 'simple_cnn')
        self.assertEqual(config_manager.get('training.batch_size'), 4)
        
        # Test default value
        self.assertEqual(config_manager.get('nonexistent.key', 'default'), 'default')
    
    def test_update_config(self):
        """Test updating configuration"""
        config_manager = ConfigManager()
        
        # Update with new values
        updates = {
            'model': {'new_param': 'new_value'},
            'new_section': {'param': 'value'}
        }
        
        config_manager.update_config(updates)
        
        # Check updates were applied
        self.assertEqual(config_manager.get('model.new_param'), 'new_value')
        self.assertEqual(config_manager.get('new_section.param'), 'value')
    
    def test_validate_config(self):
        """Test configuration validation"""
        config_manager = ConfigManager()
        
        # Valid config should pass
        config_manager.config = self.sample_config
        self.assertTrue(config_manager.validate_config())
        
        # Invalid config should fail
        invalid_config = {'invalid': 'structure'}
        config_manager.config = invalid_config
        self.assertFalse(config_manager.validate_config())
    
    def test_get_model_config(self):
        """Test getting model-specific configuration"""
        config_manager = ConfigManager()
        config_manager.config = self.sample_config
        
        model_config = config_manager.get_model_config()
        
        self.assertEqual(model_config['architecture'], 'simple_cnn')
        self.assertEqual(model_config['num_classes'], 5)
    
    def test_get_training_config(self):
        """Test getting training-specific configuration"""
        config_manager = ConfigManager()
        config_manager.config = self.sample_config
        
        training_config = config_manager.get_training_config()
        
        self.assertEqual(training_config['batch_size'], 4)
        self.assertEqual(training_config['epochs'], 2)
    
    def test_get_data_config(self):
        """Test getting data-specific configuration"""
        config_manager = ConfigManager()
        config_manager.config = self.sample_config
        
        data_config = config_manager.get_data_config()
        
        self.assertTrue(data_config['augmentation'])
        self.assertTrue(data_config['normalize'])
    
    def test_config_file_not_found(self):
        """Test handling of missing config file"""
        nonexistent_file = self.temp_dir / "nonexistent.json"
        
        # Should use default config without raising error
        config_manager = ConfigManager(config_file=str(nonexistent_file))
        self.assertIsNotNone(config_manager.config)
    
    def test_invalid_json_file(self):
        """Test handling of invalid JSON file"""
        # Create invalid JSON file
        with open(self.config_file, 'w') as f:
            f.write("invalid json content {")
        
        # Should use default config without raising error
        config_manager = ConfigManager(config_file=str(self.config_file))
        self.assertIsNotNone(config_manager.config)
    
    def test_merge_configs(self):
        """Test merging multiple configurations"""
        config_manager = ConfigManager()
        
        base_config = {
            'section1': {'param1': 'value1', 'param2': 'value2'},
            'section2': {'param3': 'value3'}
        }
        
        update_config = {
            'section1': {'param2': 'new_value2', 'param4': 'value4'},
            'section3': {'param5': 'value5'}
        }
        
        merged = config_manager._merge_configs(base_config, update_config)
        
        # Check merged result
        self.assertEqual(merged['section1']['param1'], 'value1')  # Preserved
        self.assertEqual(merged['section1']['param2'], 'new_value2')  # Updated
        self.assertEqual(merged['section1']['param4'], 'value4')  # Added
        self.assertEqual(merged['section2']['param3'], 'value3')  # Preserved
        self.assertEqual(merged['section3']['param5'], 'value5')  # Added
    
    def test_environment_variable_override(self):
        """Test configuration override from environment variables"""
        with patch.dict('os.environ', {'FISH_MODEL_ARCHITECTURE': 'resnet50'}):
            config_manager = ConfigManager()
            
            # Environment variable should override config
            if hasattr(config_manager, '_load_env_overrides'):
                config_manager._load_env_overrides()
                self.assertEqual(config_manager.get('model.architecture'), 'resnet50')

class TestModelConfigs(unittest.TestCase):
    """Test model configuration utilities"""
    
    def test_model_config_creation(self):
        """Test creating model configurations"""
        from config.model_configs import get_model_config
        
        # Test different model architectures
        simple_config = get_model_config('simple_cnn')
        self.assertIn('layers', simple_config)
        
        vgg_config = get_model_config('vgg16')
        self.assertIn('base_model', vgg_config)
    
    def test_invalid_model_config(self):
        """Test handling of invalid model configuration"""
        from config.model_configs import get_model_config
        
        # Should return None or raise appropriate error for invalid model
        result = get_model_config('invalid_model')
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()