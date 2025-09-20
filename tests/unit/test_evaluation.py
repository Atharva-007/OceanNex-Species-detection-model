"""
Unit Tests for Evaluation Framework
==================================

Tests for model evaluation, metrics calculation, and comparison utilities.
"""

import unittest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.comparison import ModelComparison
from src.evaluation.performance_analyzer import PerformanceAnalyzer
from tests.conftest import (
    TestConfig, create_temp_directory, cleanup_temp_directory,
    create_sample_labels, create_sample_probabilities, create_sample_image_data
)

class TestMetricsCalculator(unittest.TestCase):
    """Test MetricsCalculator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.metrics_calculator = MetricsCalculator()
        self.y_true = create_sample_labels(100, TestConfig.NUM_CLASSES)
        self.y_pred = create_sample_labels(100, TestConfig.NUM_CLASSES)
        self.y_prob = create_sample_probabilities(100, TestConfig.NUM_CLASSES)
    
    def test_basic_metrics_calculation(self):
        """Test basic metrics calculation"""
        metrics = self.metrics_calculator.calculate_basic_metrics(self.y_true, self.y_pred)
        
        # Check required metrics
        required_metrics = [
            'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
            'precision_micro', 'recall_micro', 'f1_micro'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)
    
    def test_per_class_metrics_calculation(self):
        """Test per-class metrics calculation"""
        class_names = TestConfig.CLASS_NAMES
        per_class = self.metrics_calculator.calculate_per_class_metrics(
            self.y_true, self.y_pred, class_names
        )
        
        self.assertIsInstance(per_class, dict)
        
        for class_name in class_names:
            if class_name in per_class:
                class_metrics = per_class[class_name]
                self.assertIn('precision', class_metrics)
                self.assertIn('recall', class_metrics)
                self.assertIn('f1_score', class_metrics)
                self.assertIn('support', class_metrics)
    
    def test_probabilistic_metrics_calculation(self):
        """Test probabilistic metrics calculation"""
        metrics = self.metrics_calculator.calculate_probabilistic_metrics(
            self.y_true, self.y_prob, TestConfig.CLASS_NAMES
        )
        
        # Check for key probabilistic metrics
        expected_metrics = ['confidence_mean', 'confidence_std', 'entropy_mean']
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
    
    def test_confusion_matrix_metrics(self):
        """Test confusion matrix derived metrics"""
        metrics = self.metrics_calculator.calculate_confusion_matrix_metrics(
            self.y_true, self.y_pred
        )
        
        self.assertIn('confusion_matrix', metrics)
        self.assertIn('normalized_confusion_matrix', metrics)
        self.assertIn('total_samples', metrics)
        self.assertEqual(metrics['total_samples'], 100)
    
    def test_top_k_accuracy(self):
        """Test top-k accuracy calculation"""
        for k in [1, 3, 5]:
            if k <= TestConfig.NUM_CLASSES:
                top_k_acc = self.metrics_calculator.calculate_top_k_accuracy(
                    self.y_true, self.y_prob, k
                )
                
                self.assertIsInstance(top_k_acc, float)
                self.assertGreaterEqual(top_k_acc, 0.0)
                self.assertLessEqual(top_k_acc, 1.0)
    
    def test_class_balance_metrics(self):
        """Test class balance analysis"""
        metrics = self.metrics_calculator.calculate_class_balance_metrics(
            self.y_true, TestConfig.CLASS_NAMES
        )
        
        self.assertIn('num_classes', metrics)
        self.assertIn('class_counts', metrics)
        self.assertIn('imbalance_ratio', metrics)
        self.assertEqual(metrics['num_classes'], TestConfig.NUM_CLASSES)
    
    def test_comprehensive_metrics(self):
        """Test comprehensive metrics calculation"""
        all_metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            self.y_true, self.y_pred, self.y_prob, TestConfig.CLASS_NAMES
        )
        
        # Check main sections
        expected_sections = ['basic', 'per_class', 'confusion_matrix', 'probabilistic']
        
        for section in expected_sections:
            self.assertIn(section, all_metrics)
    
    def test_metrics_to_dataframe(self):
        """Test converting metrics to DataFrame"""
        metrics = {'accuracy': 0.85, 'precision': 0.80}
        
        df = self.metrics_calculator.metrics_to_dataframe(metrics)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('metric', df.columns)
        self.assertIn('value', df.columns)

class TestModelEvaluator(unittest.TestCase):
    """Test ModelEvaluator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = create_temp_directory()
        self.evaluator = ModelEvaluator(output_dir=str(self.temp_dir))
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.predict.return_value = create_sample_probabilities(50, TestConfig.NUM_CLASSES)
        
        # Create test data
        self.test_data = create_sample_image_data(50)
        self.test_labels = create_sample_labels(50, TestConfig.NUM_CLASSES)
        
    def tearDown(self):
        """Clean up test fixtures"""
        cleanup_temp_directory(self.temp_dir)
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        self.assertIsNotNone(self.evaluator)
        self.assertIsNotNone(self.evaluator.metrics_calculator)
        self.assertTrue(self.temp_dir.exists())
    
    def test_evaluate_model(self):
        """Test complete model evaluation"""
        results = self.evaluator.evaluate_model(
            self.mock_model, 
            self.test_data, 
            self.test_labels,
            class_names=TestConfig.CLASS_NAMES
        )
        
        # Check evaluation results structure
        self.assertIn('predictions', results)
        self.assertIn('metrics', results)
        self.assertIn('model_info', results)
        
        # Check predictions
        self.assertEqual(results['predictions']['y_pred'].shape, (50,))
        self.assertEqual(results['predictions']['y_prob'].shape, (50, TestConfig.NUM_CLASSES))
    
    @patch('matplotlib.pyplot.savefig')
    def test_create_evaluation_visualizations(self, mock_savefig):
        """Test evaluation visualization creation"""
        y_true = self.test_labels
        y_pred = create_sample_labels(50, TestConfig.NUM_CLASSES)
        y_prob = create_sample_probabilities(50, TestConfig.NUM_CLASSES)
        
        figures = self.evaluator.create_evaluation_visualizations(
            y_true, y_pred, y_prob, TestConfig.CLASS_NAMES
        )
        
        self.assertIsInstance(figures, dict)
        # Should have created multiple visualizations
        self.assertGreater(len(figures), 0)
    
    def test_save_evaluation_results(self):
        """Test saving evaluation results"""
        results = {
            'metrics': {'accuracy': 0.85},
            'predictions': {'y_pred': self.test_labels},
            'model_info': {'name': 'test_model'}
        }
        
        self.evaluator.save_evaluation_results(results)
        
        # Check that files were created
        results_files = list(self.temp_dir.glob("*.json"))
        self.assertGreater(len(results_files), 0)

class TestModelComparison(unittest.TestCase):
    """Test ModelComparison functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = create_temp_directory()
        self.comparison = ModelComparison(output_dir=str(self.temp_dir))
        
        # Create sample data for comparison
        self.y_true = create_sample_labels(100, TestConfig.NUM_CLASSES)
        
        # Create results for different models
        self.model_results = {
            'model_a': {
                'y_pred': create_sample_labels(100, TestConfig.NUM_CLASSES),
                'y_prob': create_sample_probabilities(100, TestConfig.NUM_CLASSES)
            },
            'model_b': {
                'y_pred': create_sample_labels(100, TestConfig.NUM_CLASSES),
                'y_prob': create_sample_probabilities(100, TestConfig.NUM_CLASSES)
            }
        }
        
    def tearDown(self):
        """Clean up test fixtures"""
        cleanup_temp_directory(self.temp_dir)
    
    def test_add_model_results(self):
        """Test adding model results for comparison"""
        for model_name, results in self.model_results.items():
            self.comparison.add_model_results(
                model_name, 
                self.y_true, 
                results['y_pred'], 
                results['y_prob']
            )
        
        self.assertEqual(len(self.comparison.model_results), 2)
        self.assertIn('model_a', self.comparison.model_results)
        self.assertIn('model_b', self.comparison.model_results)
    
    def test_compare_basic_metrics(self):
        """Test basic metrics comparison"""
        # Add model results
        for model_name, results in self.model_results.items():
            self.comparison.add_model_results(
                model_name, self.y_true, results['y_pred'], results['y_prob']
            )
        
        comparison_df = self.comparison.compare_basic_metrics()
        
        self.assertIsInstance(comparison_df, pd.DataFrame)
        self.assertEqual(len(comparison_df), 2)  # Two models
        self.assertIn('Model', comparison_df.columns)
        self.assertIn('accuracy', comparison_df.columns)
    
    def test_create_performance_ranking(self):
        """Test performance ranking creation"""
        # Add model results
        for model_name, results in self.model_results.items():
            self.comparison.add_model_results(
                model_name, self.y_true, results['y_pred'], results['y_prob']
            )
        
        ranking = self.comparison.create_performance_ranking()
        
        self.assertIsInstance(ranking, pd.DataFrame)
        self.assertIn('Rank', ranking.columns)
        self.assertIn('Model', ranking.columns)
        self.assertIn('Average_Score', ranking.columns)
    
    def test_statistical_significance_test(self):
        """Test statistical significance testing"""
        # Add model results
        for model_name, results in self.model_results.items():
            self.comparison.add_model_results(
                model_name, self.y_true, results['y_pred'], results['y_prob']
            )
        
        sig_results = self.comparison.statistical_significance_test()
        
        if sig_results:  # If test was performed
            self.assertIn('pairwise_tests', sig_results)
            self.assertIn('metric_tested', sig_results)
    
    def test_get_best_model(self):
        """Test getting best performing model"""
        # Add model results
        for model_name, results in self.model_results.items():
            self.comparison.add_model_results(
                model_name, self.y_true, results['y_pred'], results['y_prob']
            )
        
        best_model, best_score = self.comparison.get_best_model('accuracy')
        
        self.assertIn(best_model, ['model_a', 'model_b'])
        self.assertIsInstance(best_score, float)

class TestPerformanceAnalyzer(unittest.TestCase):
    """Test PerformanceAnalyzer functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = create_temp_directory()
        self.analyzer = PerformanceAnalyzer(output_dir=str(self.temp_dir))
        
        self.y_true = create_sample_labels(100, TestConfig.NUM_CLASSES)
        self.y_pred = create_sample_labels(100, TestConfig.NUM_CLASSES)
        self.y_prob = create_sample_probabilities(100, TestConfig.NUM_CLASSES)
        
    def tearDown(self):
        """Clean up test fixtures"""
        cleanup_temp_directory(self.temp_dir)
    
    def test_analyze_prediction_patterns(self):
        """Test prediction pattern analysis"""
        patterns = self.analyzer.analyze_prediction_patterns(
            self.y_true, self.y_pred, self.y_prob, TestConfig.CLASS_NAMES
        )
        
        self.assertIn('confidence_analysis', patterns)
        self.assertIn('threshold_analysis', patterns)
        self.assertIn('entropy_analysis', patterns)
    
    def test_analyze_misclassifications(self):
        """Test misclassification analysis"""
        misclass = self.analyzer.analyze_misclassifications(
            self.y_true, self.y_pred, self.y_prob, TestConfig.CLASS_NAMES
        )
        
        if misclass:  # If there are misclassifications
            self.assertIn('basic_stats', misclass)
            self.assertIn('confusion_pairs', misclass)
    
    def test_analyze_feature_importance(self):
        """Test feature importance analysis"""
        features = create_sample_image_data(100, 10, 10, 1).reshape(100, -1)  # Flatten for features
        
        feature_analysis = self.analyzer.analyze_feature_importance(
            features, self.y_true, self.y_pred
        )
        
        # Should contain some analysis even if basic
        self.assertIsInstance(feature_analysis, dict)
    
    @patch('matplotlib.pyplot.savefig')
    def test_create_error_analysis_visualizations(self, mock_savefig):
        """Test error analysis visualization creation"""
        figures = self.analyzer.create_error_analysis_visualizations(
            self.y_true, self.y_pred, self.y_prob, TestConfig.CLASS_NAMES
        )
        
        self.assertIsInstance(figures, dict)
        # Should create multiple visualizations
        self.assertGreater(len(figures), 0)
    
    def test_comprehensive_analysis(self):
        """Test comprehensive performance analysis"""
        analysis = self.analyzer.generate_comprehensive_analysis(
            self.y_true, self.y_pred, self.y_prob, class_names=TestConfig.CLASS_NAMES
        )
        
        self.assertIn('prediction_patterns', analysis)
        self.assertIn('misclassifications', analysis)

if __name__ == '__main__':
    unittest.main()