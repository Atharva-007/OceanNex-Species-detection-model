"""
End-to-End Validation Tests
=========================

Complete validation tests that verify the entire system works as expected
and maintains backward compatibility with original functionality.
"""

import unittest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from unittest.mock import Mock, patch

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.conftest import (
    TestConfig, create_temp_directory, cleanup_temp_directory,
    create_sample_dataset_structure, create_sample_image_data,
    create_sample_labels, create_sample_probabilities
)

class TestFunctionalityValidation(unittest.TestCase):
    """Validate that all original functionality is preserved"""
    
    def setUp(self):
        """Set up validation test environment"""
        self.temp_dir = create_temp_directory()
        self.dataset_dir = self.temp_dir / "validation_dataset"
        
        # Create comprehensive test dataset
        create_sample_dataset_structure(
            self.dataset_dir, TestConfig.CLASS_NAMES, samples_per_class=20
        )
        
    def tearDown(self):
        """Clean up validation test environment"""
        cleanup_temp_directory(self.temp_dir)
    
    def test_complete_training_pipeline_validation(self):
        """Validate complete training pipeline works end-to-end"""
        from config.settings import ConfigManager
        from src.core.model_manager import ModelManager
        from src.data.dataset_manager import DatasetManager
        from src.training.training_manager import TrainingManager
        
        # Initialize components
        config_manager = ConfigManager()
        model_manager = ModelManager()
        dataset_manager = DatasetManager()
        training_manager = TrainingManager(
            config_manager=config_manager,
            model_manager=model_manager,
            dataset_manager=dataset_manager,
            output_dir=str(self.temp_dir)
        )
        
        # Configure training
        training_config = {
            'model': {
                'architecture': 'simple_cnn',
                'input_shape': [224, 224, 3],
                'num_classes': len(TestConfig.CLASS_NAMES)
            },
            'training': {
                'batch_size': 4,
                'epochs': 1,  # Short training for validation
                'validation_split': 0.2
            },
            'data': {
                'dataset_dir': str(self.dataset_dir),
                'batch_size': 4,
                'augmentation': True
            }
        }
        
        # Mock the actual training to avoid long computation
        with patch('tensorflow.keras.models.Sequential') as mock_sequential, \
             patch('tensorflow.keras.models.Model.fit') as mock_fit, \
             patch('tensorflow.keras.preprocessing.image.ImageDataGenerator.flow_from_directory') as mock_flow:
            
            # Setup mocks
            mock_model = Mock()
            mock_sequential.return_value = mock_model
            mock_fit.return_value = Mock()
            mock_fit.return_value.history = {'loss': [0.5], 'accuracy': [0.8]}
            
            mock_generator = Mock()
            mock_generator.samples = 80
            mock_generator.batch_size = 4
            mock_generator.class_indices = {name: i for i, name in enumerate(TestConfig.CLASS_NAMES)}
            mock_flow.return_value = mock_generator
            
            # Run training pipeline
            results = training_manager.train_model(training_config)
            
            # Validate results
            self.assertIsNotNone(results)
            self.assertIn('model', results)
            self.assertIn('history', results)
            self.assertIn('config', results)
    
    def test_complete_evaluation_pipeline_validation(self):
        """Validate complete evaluation pipeline works end-to-end"""
        from src.evaluation.evaluator import ModelEvaluator
        from src.evaluation.comparison import ModelComparison
        from src.evaluation.performance_analyzer import PerformanceAnalyzer
        
        # Create test data
        test_data = create_sample_image_data(100)
        test_labels = create_sample_labels(100, TestConfig.NUM_CLASSES)
        
        # Mock model predictions
        mock_model_a = Mock()
        mock_model_b = Mock()
        
        with patch('tensorflow.keras.models.Model.predict') as mock_predict:
            # Different prediction patterns for two models
            y_prob_a = create_sample_probabilities(100, TestConfig.NUM_CLASSES)
            y_prob_b = create_sample_probabilities(100, TestConfig.NUM_CLASSES)
            
            mock_predict.side_effect = [y_prob_a, y_prob_b]
            
            # Evaluate first model
            evaluator = ModelEvaluator(output_dir=str(self.temp_dir / "eval_a"))
            results_a = evaluator.evaluate_model(
                mock_model_a, test_data, test_labels, TestConfig.CLASS_NAMES
            )
            
            # Evaluate second model
            evaluator_b = ModelEvaluator(output_dir=str(self.temp_dir / "eval_b"))
            results_b = evaluator_b.evaluate_model(
                mock_model_b, test_data, test_labels, TestConfig.CLASS_NAMES
            )
            
            # Compare models
            comparison = ModelComparison(output_dir=str(self.temp_dir / "comparison"))
            comparison.add_model_results(
                'model_a', test_labels, results_a['predictions']['y_pred'], y_prob_a
            )
            comparison.add_model_results(
                'model_b', test_labels, results_b['predictions']['y_pred'], y_prob_b
            )
            
            # Validate comparison
            ranking = comparison.create_performance_ranking()
            self.assertEqual(len(ranking), 2)
            
            # Performance analysis
            analyzer = PerformanceAnalyzer(output_dir=str(self.temp_dir / "analysis"))
            analysis = analyzer.generate_comprehensive_analysis(
                test_labels, results_a['predictions']['y_pred'], y_prob_a, 
                class_names=TestConfig.CLASS_NAMES
            )
            
            self.assertIn('prediction_patterns', analysis)
            self.assertIn('misclassifications', analysis)
    
    def test_data_processing_validation(self):
        """Validate data processing maintains consistency"""
        from src.data.dataset_manager import DatasetManager
        
        dataset_manager = DatasetManager()
        
        # Test dataset scanning
        dataset_info = dataset_manager.scan_dataset_directory(str(self.dataset_dir))
        
        # Validate dataset info
        self.assertIn('classes', dataset_info)
        self.assertIn('splits', dataset_info)
        self.assertEqual(len(dataset_info['classes']), len(TestConfig.CLASS_NAMES))
        
        # Test statistics calculation
        stats = dataset_manager.get_dataset_statistics(str(self.dataset_dir))
        
        self.assertIn('num_classes', stats)
        self.assertIn('total_samples', stats)
        self.assertEqual(stats['num_classes'], len(TestConfig.CLASS_NAMES))
        
        # Test data preprocessing
        sample_data = create_sample_image_data(50, 256, 256, 3)
        sample_labels = create_sample_labels(50, TestConfig.NUM_CLASSES)
        
        processed_data, processed_labels = dataset_manager.apply_preprocessing_pipeline(
            sample_data, sample_labels, target_size=(224, 224), normalize=True, categorical=True
        )
        
        # Validate preprocessing
        self.assertEqual(processed_data.shape, (50, 224, 224, 3))
        self.assertEqual(processed_labels.shape, (50, TestConfig.NUM_CLASSES))
        self.assertGreaterEqual(np.min(processed_data), 0.0)
        self.assertLessEqual(np.max(processed_data), 1.0)
    
    def test_configuration_system_validation(self):
        """Validate configuration system works correctly"""
        from config.settings import ConfigManager
        
        config_manager = ConfigManager()
        
        # Test default configuration
        self.assertIsNotNone(config_manager.config)
        self.assertIn('model', config_manager.config)
        self.assertIn('training', config_manager.config)
        self.assertIn('data', config_manager.config)
        
        # Test configuration updates
        updates = {
            'model': {'new_param': 'test_value'},
            'training': {'epochs': 100}
        }
        
        config_manager.update_config(updates)
        
        self.assertEqual(config_manager.get('model.new_param'), 'test_value')
        self.assertEqual(config_manager.get('training.epochs'), 100)
        
        # Test configuration saving/loading
        config_file = self.temp_dir / "test_config.json"
        config_manager.save_config(str(config_file))
        
        self.assertTrue(config_file.exists())
        
        # Load and verify
        new_config_manager = ConfigManager(config_file=str(config_file))
        self.assertEqual(new_config_manager.get('model.new_param'), 'test_value')
    
    def test_model_management_validation(self):
        """Validate model management functionality"""
        from src.core.model_manager import ModelManager
        
        model_manager = ModelManager()
        
        # Test model creation
        model_config = {
            'architecture': 'simple_cnn',
            'input_shape': [224, 224, 3],
            'num_classes': TestConfig.NUM_CLASSES
        }
        
        with patch('tensorflow.keras.models.Sequential') as mock_sequential:
            mock_model = Mock()
            mock_sequential.return_value = mock_model
            
            model = model_manager.create_model(model_config)
            self.assertIsNotNone(model)
        
        # Test model validation
        self.assertTrue(model_manager.validate_model_config(model_config))
        
        # Test invalid config
        invalid_config = {'architecture': 'invalid'}
        self.assertFalse(model_manager.validate_model_config(invalid_config))
    
    def test_backward_compatibility_validation(self):
        """Validate backward compatibility with original scripts"""
        # Test that old functionality patterns still work
        
        # Original dataset analysis should work
        from src.data.dataset_manager import DatasetManager
        dataset_manager = DatasetManager()
        
        stats = dataset_manager.get_dataset_statistics(str(self.dataset_dir))
        self.assertIsInstance(stats, dict)
        
        # Original model creation patterns should work
        from src.core.model_manager import ModelManager
        model_manager = ModelManager()
        
        architectures = model_manager.list_available_architectures()
        self.assertIn('simple_cnn', architectures)
    
    def test_error_handling_validation(self):
        """Validate proper error handling throughout system"""
        from src.core.model_manager import ModelManager
        from src.data.dataset_manager import DatasetManager
        
        model_manager = ModelManager()
        dataset_manager = DatasetManager()
        
        # Test invalid model architecture
        with self.assertRaises(ValueError):
            model_manager.create_model({'architecture': 'nonexistent_model'})
        
        # Test invalid dataset path
        with self.assertRaises(Exception):
            dataset_manager.scan_dataset_directory("/completely/invalid/path")
        
        # Test invalid configuration
        from config.settings import ConfigManager
        config_manager = ConfigManager()
        
        # Should handle invalid config gracefully
        invalid_updates = {'invalid': {'structure': {'deeply': {'nested': 'value'}}}}
        config_manager.update_config(invalid_updates)
        # Should not crash
        
    def test_performance_validation(self):
        """Validate system performance meets requirements"""
        import time
        
        from src.data.dataset_manager import DatasetManager
        dataset_manager = DatasetManager()
        
        # Test data processing speed
        large_data = create_sample_image_data(500)  # 500 samples
        large_labels = create_sample_labels(500, TestConfig.NUM_CLASSES)
        
        start_time = time.time()
        processed_data, processed_labels = dataset_manager.apply_preprocessing_pipeline(
            large_data, large_labels, target_size=(224, 224)
        )
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process 500 samples in reasonable time (30 seconds max)
        self.assertLess(processing_time, 30.0)
        
        # Test evaluation speed
        from src.evaluation.evaluator import ModelEvaluator
        evaluator = ModelEvaluator(output_dir=str(self.temp_dir))
        
        with patch('tensorflow.keras.models.Model.predict') as mock_predict:
            mock_predict.return_value = create_sample_probabilities(500, TestConfig.NUM_CLASSES)
            
            mock_model = Mock()
            test_data = create_sample_image_data(500)
            test_labels = create_sample_labels(500, TestConfig.NUM_CLASSES)
            
            start_time = time.time()
            results = evaluator.evaluate_model(
                mock_model, test_data, test_labels, TestConfig.CLASS_NAMES
            )
            end_time = time.time()
            
            evaluation_time = end_time - start_time
            
            # Evaluation should complete in reasonable time (60 seconds max)
            self.assertLess(evaluation_time, 60.0)

class TestSystemRobustness(unittest.TestCase):
    """Test system robustness and edge cases"""
    
    def setUp(self):
        """Set up robustness test environment"""
        self.temp_dir = create_temp_directory()
        
    def tearDown(self):
        """Clean up robustness test environment"""
        cleanup_temp_directory(self.temp_dir)
    
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets"""
        from src.data.dataset_manager import DatasetManager
        
        dataset_manager = DatasetManager()
        empty_dir = self.temp_dir / "empty_dataset"
        empty_dir.mkdir()
        
        # Should handle empty directory gracefully
        stats = dataset_manager.get_dataset_statistics(str(empty_dir))
        self.assertEqual(stats['num_classes'], 0)
        self.assertEqual(stats['total_samples'], 0)
    
    def test_single_class_dataset_handling(self):
        """Test handling of single-class datasets"""
        from src.data.dataset_manager import DatasetManager
        
        dataset_manager = DatasetManager()
        single_class_dir = self.temp_dir / "single_class_dataset"
        
        # Create single-class dataset
        create_sample_dataset_structure(
            single_class_dir, ['single_fish'], samples_per_class=10
        )
        
        stats = dataset_manager.get_dataset_statistics(str(single_class_dir))
        self.assertEqual(stats['num_classes'], 1)
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets (memory efficiency)"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large dataset simulation
        from src.data.dataset_manager import DatasetManager
        dataset_manager = DatasetManager()
        
        # Process large batch of data
        large_data = create_sample_image_data(2000)  # 2000 samples
        large_labels = create_sample_labels(2000, TestConfig.NUM_CLASSES)
        
        # Should handle without excessive memory usage
        processed_data, processed_labels = dataset_manager.apply_preprocessing_pipeline(
            large_data, large_labels, target_size=(224, 224)
        )
        
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory
        
        # Memory increase should be reasonable (less than 2GB)
        self.assertLess(memory_increase, 2 * 1024 * 1024 * 1024)
    
    def test_corrupted_data_handling(self):
        """Test handling of corrupted or invalid data"""
        from src.data.dataset_manager import DatasetManager
        
        dataset_manager = DatasetManager()
        
        # Test with invalid image data
        invalid_data = np.array([])  # Empty array
        
        # Should handle gracefully without crashing
        try:
            dataset_manager.normalize_images(invalid_data)
        except (ValueError, TypeError):
            # Expected to raise error for invalid input
            pass
    
    def test_concurrent_access_handling(self):
        """Test handling of concurrent access to system components"""
        import threading
        import time
        
        from src.core.model_manager import ModelManager
        
        results = []
        errors = []
        
        def create_model_worker():
            try:
                model_manager = ModelManager()
                with patch('tensorflow.keras.models.Sequential'):
                    model_config = {
                        'architecture': 'simple_cnn',
                        'input_shape': [224, 224, 3],
                        'num_classes': TestConfig.NUM_CLASSES
                    }
                    model = model_manager.create_model(model_config)
                    results.append(model)
            except Exception as e:
                errors.append(e)
        
        # Create multiple concurrent workers
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=create_model_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should complete without errors
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 5)

if __name__ == '__main__':
    unittest.main()