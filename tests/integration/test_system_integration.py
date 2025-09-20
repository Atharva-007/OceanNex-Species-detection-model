"""
Integration Tests for Fish Species Classification System
======================================================

End-to-end integration tests that verify the complete system works together.
"""

import unittest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import ConfigManager
from src.core.model_manager import ModelManager
from src.data.dataset_manager import DatasetManager
from src.training.training_manager import TrainingManager
from src.evaluation.evaluator import ModelEvaluator
from tests.conftest import (
    TestConfig, create_temp_directory, cleanup_temp_directory,
    create_sample_config, create_sample_dataset_structure,
    create_sample_image_data, create_sample_labels
)

class TestSystemIntegration(unittest.TestCase):
    """Test complete system integration"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = create_temp_directory()
        self.dataset_dir = self.temp_dir / "test_dataset"
        self.output_dir = self.temp_dir / "outputs"
        self.output_dir.mkdir()
        
        # Create sample dataset structure
        create_sample_dataset_structure(
            self.dataset_dir, TestConfig.CLASS_NAMES, samples_per_class=10
        )
        
        # Initialize system components
        self.config_manager = ConfigManager()
        self.model_manager = ModelManager()
        self.dataset_manager = DatasetManager()
        
    def tearDown(self):
        """Clean up integration test environment"""
        cleanup_temp_directory(self.temp_dir)
    
    def test_config_to_model_integration(self):
        """Test configuration system integration with model manager"""
        config = create_sample_config()
        self.config_manager.config = config
        
        # Model manager should use config
        model_config = self.config_manager.get_model_config()
        
        with patch('tensorflow.keras.models.Sequential'):
            model = self.model_manager.create_model(model_config)
            self.assertIsNotNone(model)
    
    def test_dataset_to_training_integration(self):
        """Test dataset manager integration with training pipeline"""
        # Scan dataset
        dataset_info = self.dataset_manager.scan_dataset_directory(str(self.dataset_dir))
        
        self.assertIn('classes', dataset_info)
        self.assertEqual(len(dataset_info['classes']), len(TestConfig.CLASS_NAMES))
        
        # Create data generators
        train_dir = self.dataset_dir / 'train'
        val_dir = self.dataset_dir / 'val'
        
        with patch('tensorflow.keras.preprocessing.image.ImageDataGenerator.flow_from_directory') as mock_flow:
            mock_generator = Mock()
            mock_generator.samples = 50
            mock_generator.batch_size = TestConfig.BATCH_SIZE
            mock_generator.class_indices = {name: i for i, name in enumerate(TestConfig.CLASS_NAMES)}
            mock_flow.return_value = mock_generator
            
            train_gen, val_gen = self.dataset_manager.get_data_generators(
                str(train_dir), str(val_dir), TestConfig.BATCH_SIZE
            )
            
            self.assertIsNotNone(train_gen)
            self.assertIsNotNone(val_gen)
    
    @patch('tensorflow.keras.models.Sequential')
    @patch('tensorflow.keras.models.Model.fit')
    def test_model_training_integration(self, mock_fit, mock_sequential):
        """Test complete model training integration"""
        # Setup mocks
        mock_model = Mock()
        mock_sequential.return_value = mock_model
        mock_fit.return_value = Mock()
        mock_fit.return_value.history = {'loss': [0.5, 0.3], 'accuracy': [0.7, 0.9]}
        
        # Create training manager
        training_manager = TrainingManager(
            config_manager=self.config_manager,
            model_manager=self.model_manager,
            dataset_manager=self.dataset_manager,
            output_dir=str(self.output_dir)
        )
        
        # Configure training
        config = create_sample_config()
        training_config = {
            'model': config['model'],
            'training': config['training'],
            'data': {
                'dataset_dir': str(self.dataset_dir),
                'batch_size': TestConfig.BATCH_SIZE
            }
        }
        
        with patch('tensorflow.keras.preprocessing.image.ImageDataGenerator.flow_from_directory') as mock_flow:
            mock_generator = Mock()
            mock_generator.samples = 50
            mock_generator.batch_size = TestConfig.BATCH_SIZE
            mock_generator.class_indices = {name: i for i, name in enumerate(TestConfig.CLASS_NAMES)}
            mock_flow.return_value = mock_generator
            
            # Run training
            results = training_manager.train_model(training_config)
            
            self.assertIsNotNone(results)
            self.assertIn('model', results)
            self.assertIn('history', results)
    
    @patch('tensorflow.keras.models.Model.predict')
    def test_evaluation_integration(self, mock_predict):
        """Test model evaluation integration"""
        # Setup mock model and predictions
        mock_model = Mock()
        mock_predict.return_value = create_sample_probabilities(50, TestConfig.NUM_CLASSES)
        
        # Create test data
        test_data = create_sample_image_data(50)
        test_labels = create_sample_labels(50, TestConfig.NUM_CLASSES)
        
        # Create evaluator
        evaluator = ModelEvaluator(output_dir=str(self.output_dir))
        
        # Run evaluation
        results = evaluator.evaluate_model(
            mock_model, test_data, test_labels, TestConfig.CLASS_NAMES
        )
        
        self.assertIn('metrics', results)
        self.assertIn('predictions', results)
        self.assertIn('basic', results['metrics'])
    
    def test_complete_pipeline_integration(self):
        """Test complete pipeline from data loading to evaluation"""
        config = create_sample_config()
        
        # Step 1: Dataset analysis
        dataset_stats = self.dataset_manager.get_dataset_statistics(str(self.dataset_dir))
        self.assertIn('num_classes', dataset_stats)
        
        # Step 2: Model creation
        with patch('tensorflow.keras.models.Sequential') as mock_sequential:
            mock_model = Mock()
            mock_sequential.return_value = mock_model
            
            model = self.model_manager.create_model(config['model'])
            self.assertIsNotNone(model)
        
        # Step 3: Training setup
        with patch('tensorflow.keras.preprocessing.image.ImageDataGenerator.flow_from_directory') as mock_flow:
            mock_generator = Mock()
            mock_generator.samples = 50
            mock_generator.batch_size = TestConfig.BATCH_SIZE
            mock_flow.return_value = mock_generator
            
            train_gen, val_gen = self.dataset_manager.get_data_generators(
                str(self.dataset_dir / 'train'),
                str(self.dataset_dir / 'val'),
                TestConfig.BATCH_SIZE
            )
        
        # Step 4: Evaluation
        with patch('tensorflow.keras.models.Model.predict') as mock_predict:
            mock_predict.return_value = create_sample_probabilities(20, TestConfig.NUM_CLASSES)
            
            test_data = create_sample_image_data(20)
            test_labels = create_sample_labels(20, TestConfig.NUM_CLASSES)
            
            evaluator = ModelEvaluator(output_dir=str(self.output_dir))
            eval_results = evaluator.evaluate_model(
                mock_model, test_data, test_labels, TestConfig.CLASS_NAMES
            )
            
            self.assertIsNotNone(eval_results)
    
    def test_configuration_propagation(self):
        """Test that configuration changes propagate through system"""
        # Update configuration
        new_config = {
            'model': {'architecture': 'vgg16', 'num_classes': 10},
            'training': {'batch_size': 8, 'epochs': 5}
        }
        
        self.config_manager.update_config(new_config)
        
        # Check propagation to model manager
        model_config = self.config_manager.get_model_config()
        self.assertEqual(model_config['architecture'], 'vgg16')
        self.assertEqual(model_config['num_classes'], 10)
        
        # Check propagation to training config
        training_config = self.config_manager.get_training_config()
        self.assertEqual(training_config['batch_size'], 8)
        self.assertEqual(training_config['epochs'], 5)
    
    def test_error_handling_integration(self):
        """Test error handling across system components"""
        # Test invalid model configuration
        invalid_config = {'architecture': 'invalid_model'}
        
        with self.assertRaises(ValueError):
            self.model_manager.create_model(invalid_config)
        
        # Test invalid dataset path
        with self.assertRaises(Exception):
            self.dataset_manager.scan_dataset_directory("/nonexistent/path")
    
    def test_data_flow_consistency(self):
        """Test data consistency through the pipeline"""
        # Create consistent test data
        num_samples = 50
        test_data = create_sample_image_data(num_samples)
        test_labels = create_sample_labels(num_samples, TestConfig.NUM_CLASSES)
        
        # Process through dataset manager
        processed_data, processed_labels = self.dataset_manager.apply_preprocessing_pipeline(
            test_data, test_labels, target_size=(224, 224), normalize=True, categorical=True
        )
        
        # Check data consistency
        self.assertEqual(len(processed_data), len(processed_labels))
        self.assertEqual(processed_data.shape[0], num_samples)
        self.assertEqual(processed_labels.shape[0], num_samples)
        
        # Mock model prediction
        with patch('tensorflow.keras.models.Model.predict') as mock_predict:
            mock_predict.return_value = create_sample_probabilities(num_samples, TestConfig.NUM_CLASSES)
            
            mock_model = Mock()
            evaluator = ModelEvaluator(output_dir=str(self.output_dir))
            
            # Convert categorical back to labels for evaluation
            label_indices = np.argmax(processed_labels, axis=1)
            
            results = evaluator.evaluate_model(
                mock_model, processed_data, label_indices, TestConfig.CLASS_NAMES
            )
            
            # Check consistency
            self.assertEqual(len(results['predictions']['y_pred']), num_samples)
            self.assertEqual(results['predictions']['y_prob'].shape[0], num_samples)

class TestUIIntegration(unittest.TestCase):
    """Test UI integration with backend systems"""
    
    def setUp(self):
        """Set up UI integration test environment"""
        self.temp_dir = create_temp_directory()
        
    def tearDown(self):
        """Clean up UI integration test environment"""
        cleanup_temp_directory(self.temp_dir)
    
    @patch('streamlit.selectbox')
    @patch('streamlit.file_uploader')
    def test_streamlit_backend_integration(self, mock_uploader, mock_selectbox):
        """Test Streamlit UI integration with backend"""
        # Mock UI inputs
        mock_selectbox.return_value = 'simple_cnn'
        mock_uploader.return_value = None
        
        # Import UI module (would normally be run in Streamlit context)
        try:
            from src.ui.streamlit_app import FishClassifierUI
            
            # Initialize UI with backend systems
            ui = FishClassifierUI()
            
            # Test UI-backend connection
            self.assertIsNotNone(ui.model_manager)
            self.assertIsNotNone(ui.dataset_manager)
            
        except ImportError:
            # Skip if Streamlit not available
            self.skipTest("Streamlit not available for testing")

class TestPerformanceIntegration(unittest.TestCase):
    """Test system performance and resource usage"""
    
    def setUp(self):
        """Set up performance test environment"""
        self.temp_dir = create_temp_directory()
        
    def tearDown(self):
        """Clean up performance test environment"""
        cleanup_temp_directory(self.temp_dir)
    
    def test_memory_usage_integration(self):
        """Test memory usage across system components"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create multiple system components
        config_manager = ConfigManager()
        model_manager = ModelManager()
        dataset_manager = DatasetManager()
        
        # Create large test data
        large_data = create_sample_image_data(1000)
        large_labels = create_sample_labels(1000, TestConfig.NUM_CLASSES)
        
        # Process data
        processed_data, processed_labels = dataset_manager.apply_preprocessing_pipeline(
            large_data, large_labels, target_size=(224, 224)
        )
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 1GB for test data)
        self.assertLess(memory_increase, 1024 * 1024 * 1024)  # 1GB limit
    
    def test_processing_speed_integration(self):
        """Test processing speed across components"""
        import time
        
        dataset_manager = DatasetManager()
        
        # Time data processing
        start_time = time.time()
        
        test_data = create_sample_image_data(100)
        test_labels = create_sample_labels(100, TestConfig.NUM_CLASSES)
        
        processed_data, processed_labels = dataset_manager.apply_preprocessing_pipeline(
            test_data, test_labels, target_size=(224, 224)
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Processing should complete within reasonable time (10 seconds for 100 samples)
        self.assertLess(processing_time, 10.0)

if __name__ == '__main__':
    unittest.main()