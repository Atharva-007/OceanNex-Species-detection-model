"""
Model Evaluator Module
=====================

Comprehensive model evaluation for fish species classification with
detailed metrics, visualization, and analysis capabilities.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import LabelBinarizer

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from ..utils.logging_utils import get_logger
from ..core.prediction import FishPredictor
from ..data.dataset_manager import DatasetManager
from .metrics import MetricsCalculator

logger = get_logger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation with detailed analysis"""
    
    def __init__(self, model_path: str, dataset_path: str, class_names: List[str] = None):
        """
        Initialize model evaluator
        
        Args:
            model_path: Path to the trained model
            dataset_path: Path to the dataset
            class_names: List of class names (optional)
        """
        self.model_path = model_path
        self.dataset_path = Path(dataset_path)
        self.class_names = class_names
        
        # Load model
        self.model = self._load_model()
        
        # Initialize components
        self.dataset_manager = DatasetManager()
        self.metrics_calculator = MetricsCalculator()
        self.predictor = None
        
        # Evaluation results
        self.evaluation_results = {}
        self.predictions = None
        self.true_labels = None
        self.prediction_probabilities = None
        
        logger.info(f"Model evaluator initialized with model: {model_path}")
    
    def _load_model(self) -> keras.Model:
        """Load the trained model"""
        try:
            model = keras.models.load_model(self.model_path)
            logger.info(f"Model loaded successfully: {model.count_params():,} parameters")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate_on_test_set(self, test_path: str = None) -> Dict[str, Any]:
        """
        Evaluate model on test dataset
        
        Args:
            test_path: Path to test dataset (optional)
            
        Returns:
            Dictionary of evaluation results
        """
        test_path = test_path or self.dataset_path / "test"
        
        logger.info(f"Evaluating model on test set: {test_path}")
        
        # Prepare test data
        test_generator = self._prepare_test_generator(test_path)
        
        # Get class names if not provided
        if self.class_names is None:
            self.class_names = list(test_generator.class_indices.keys())
        
        # Generate predictions
        start_time = time.time()
        
        # Reset generator
        test_generator.reset()
        
        # Make predictions
        self.prediction_probabilities = self.model.predict(test_generator, verbose=1)
        self.predictions = np.argmax(self.prediction_probabilities, axis=1)
        
        # Get true labels
        self.true_labels = test_generator.classes
        
        prediction_time = time.time() - start_time
        
        # Calculate basic metrics
        basic_metrics = self._calculate_basic_metrics()
        
        # Calculate detailed metrics
        detailed_metrics = self._calculate_detailed_metrics()
        
        # Calculate per-class metrics
        per_class_metrics = self._calculate_per_class_metrics()
        
        # Calculate confusion matrix
        confusion_mat = self._calculate_confusion_matrix()
        
        # Combine all results
        self.evaluation_results = {
            'basic_metrics': basic_metrics,
            'detailed_metrics': detailed_metrics,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': confusion_mat,
            'prediction_time': prediction_time,
            'samples_evaluated': len(self.true_labels),
            'class_names': self.class_names
        }
        
        logger.info(f"Evaluation completed in {prediction_time:.2f} seconds")
        logger.info(f"Overall accuracy: {basic_metrics['accuracy']:.4f}")
        
        return self.evaluation_results
    
    def _prepare_test_generator(self, test_path: Path):
        """Prepare test data generator"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        test_datagen = ImageDataGenerator(rescale=1.0/255.0)
        
        test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=(224, 224),  # Standard size, can be made configurable
            batch_size=32,
            class_mode='categorical',
            shuffle=False  # Important for evaluation
        )
        
        return test_generator
    
    def _calculate_basic_metrics(self) -> Dict[str, float]:
        """Calculate basic evaluation metrics"""
        return {
            'accuracy': accuracy_score(self.true_labels, self.predictions),
            'total_samples': len(self.true_labels),
            'num_classes': len(self.class_names),
            'correct_predictions': np.sum(self.true_labels == self.predictions),
            'incorrect_predictions': np.sum(self.true_labels != self.predictions)
        }
    
    def _calculate_detailed_metrics(self) -> Dict[str, Any]:
        """Calculate detailed evaluation metrics"""
        # Precision, recall, F1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            self.true_labels, self.predictions, average=None, zero_division=0
        )
        
        # Macro and micro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            self.true_labels, self.predictions, average='macro', zero_division=0
        )
        
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            self.true_labels, self.predictions, average='micro', zero_division=0
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            self.true_labels, self.predictions, average='weighted', zero_division=0
        )
        
        return {
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }
    
    def _calculate_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate per-class detailed metrics"""
        per_class = {}
        
        for i, class_name in enumerate(self.class_names):
            # Binary classification metrics for this class
            true_binary = (self.true_labels == i).astype(int)
            pred_binary = (self.predictions == i).astype(int)
            
            # Calculate metrics
            true_positives = np.sum((true_binary == 1) & (pred_binary == 1))
            false_positives = np.sum((true_binary == 0) & (pred_binary == 1))
            false_negatives = np.sum((true_binary == 1) & (pred_binary == 0))
            true_negatives = np.sum((true_binary == 0) & (pred_binary == 0))
            
            # Calculate derived metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
            
            per_class[class_name] = {
                'true_positives': int(true_positives),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives),
                'true_negatives': int(true_negatives),
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'support': int(np.sum(true_binary)),
                'predicted_count': int(np.sum(pred_binary))
            }
        
        return per_class
    
    def _calculate_confusion_matrix(self) -> Dict[str, Any]:
        """Calculate confusion matrix and related metrics"""
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        return {
            'matrix': cm.tolist(),
            'normalized_matrix': cm_normalized.tolist(),
            'accuracy_per_class': cm.diagonal() / cm.sum(axis=1),
            'total_correct': int(np.trace(cm)),
            'total_samples': int(np.sum(cm))
        }
    
    def generate_classification_report(self) -> pd.DataFrame:
        """Generate detailed classification report"""
        if self.evaluation_results is None:
            raise ValueError("Run evaluation first")
        
        # Get sklearn classification report
        report_dict = classification_report(
            self.true_labels,
            self.predictions,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(report_dict).transpose()
        
        # Add additional columns
        if 'per_class_metrics' in self.evaluation_results:
            per_class = self.evaluation_results['per_class_metrics']
            
            for class_name in self.class_names:
                if class_name in per_class:
                    metrics = per_class[class_name]
                    idx = df.index == class_name
                    df.loc[idx, 'specificity'] = metrics['specificity']
                    df.loc[idx, 'true_positives'] = metrics['true_positives']
                    df.loc[idx, 'false_positives'] = metrics['false_positives']
                    df.loc[idx, 'false_negatives'] = metrics['false_negatives']
        
        return df
    
    def plot_confusion_matrix(self, normalize: bool = True, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """Plot confusion matrix"""
        if self.evaluation_results is None:
            raise ValueError("Run evaluation first")
        
        cm_data = self.evaluation_results['confusion_matrix']
        cm = np.array(cm_data['normalized_matrix'] if normalize else cm_data['matrix'])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Labels and title
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.class_names,
               yticklabels=self.class_names,
               title='Normalized Confusion Matrix' if normalize else 'Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        return fig
    
    def plot_per_class_metrics(self) -> plt.Figure:
        """Plot per-class metrics"""
        if self.evaluation_results is None:
            raise ValueError("Run evaluation first")
        
        detailed_metrics = self.evaluation_results['detailed_metrics']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision per class
        ax1.bar(self.class_names, detailed_metrics['precision_per_class'])
        ax1.set_title('Precision per Class')
        ax1.set_ylabel('Precision')
        ax1.tick_params(axis='x', rotation=45)
        
        # Recall per class
        ax2.bar(self.class_names, detailed_metrics['recall_per_class'])
        ax2.set_title('Recall per Class')
        ax2.set_ylabel('Recall')
        ax2.tick_params(axis='x', rotation=45)
        
        # F1-score per class
        ax3.bar(self.class_names, detailed_metrics['f1_per_class'])
        ax3.set_title('F1-Score per Class')
        ax3.set_ylabel('F1-Score')
        ax3.tick_params(axis='x', rotation=45)
        
        # Support per class
        ax4.bar(self.class_names, detailed_metrics['support_per_class'])
        ax4.set_title('Support (Sample Count) per Class')
        ax4.set_ylabel('Number of Samples')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(self, num_classes_to_plot: int = 10) -> plt.Figure:
        """Plot ROC curves for top classes"""
        if self.prediction_probabilities is None:
            raise ValueError("Run evaluation first")
        
        # Binarize labels
        lb = LabelBinarizer()
        true_labels_binary = lb.fit_transform(self.true_labels)
        
        # If binary classification, expand dimensions
        if true_labels_binary.shape[1] == 1:
            true_labels_binary = np.hstack([1 - true_labels_binary, true_labels_binary])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve for each class (limit to top N)
        classes_to_plot = min(num_classes_to_plot, len(self.class_names))
        
        for i in range(classes_to_plot):
            if i < true_labels_binary.shape[1] and i < self.prediction_probabilities.shape[1]:
                fpr, tpr, _ = roc_curve(true_labels_binary[:, i], self.prediction_probabilities[:, i])
                auc_score = roc_auc_score(true_labels_binary[:, i], self.prediction_probabilities[:, i])
                
                ax.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {auc_score:.2f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curves(self, num_classes_to_plot: int = 10) -> plt.Figure:
        """Plot Precision-Recall curves for top classes"""
        if self.prediction_probabilities is None:
            raise ValueError("Run evaluation first")
        
        # Binarize labels
        lb = LabelBinarizer()
        true_labels_binary = lb.fit_transform(self.true_labels)
        
        # If binary classification, expand dimensions
        if true_labels_binary.shape[1] == 1:
            true_labels_binary = np.hstack([1 - true_labels_binary, true_labels_binary])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot PR curve for each class (limit to top N)
        classes_to_plot = min(num_classes_to_plot, len(self.class_names))
        
        for i in range(classes_to_plot):
            if i < true_labels_binary.shape[1] and i < self.prediction_probabilities.shape[1]:
                precision, recall, _ = precision_recall_curve(
                    true_labels_binary[:, i], 
                    self.prediction_probabilities[:, i]
                )
                ap_score = average_precision_score(
                    true_labels_binary[:, i], 
                    self.prediction_probabilities[:, i]
                )
                
                ax.plot(recall, precision, label=f'{self.class_names[i]} (AP = {ap_score:.2f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analyze_misclassifications(self, top_k: int = 10) -> pd.DataFrame:
        """Analyze most common misclassifications"""
        if self.predictions is None:
            raise ValueError("Run evaluation first")
        
        # Find misclassified samples
        misclassified_mask = self.true_labels != self.predictions
        misclassified_true = self.true_labels[misclassified_mask]
        misclassified_pred = self.predictions[misclassified_mask]
        
        # Count misclassification pairs
        misclass_pairs = []
        for true_label, pred_label in zip(misclassified_true, misclassified_pred):
            misclass_pairs.append((self.class_names[true_label], self.class_names[pred_label]))
        
        # Count occurrences
        from collections import Counter
        misclass_counts = Counter(misclass_pairs)
        
        # Create DataFrame
        misclass_data = []
        for (true_class, pred_class), count in misclass_counts.most_common(top_k):
            misclass_data.append({
                'true_class': true_class,
                'predicted_class': pred_class,
                'count': count,
                'percentage': count / len(misclassified_true) * 100
            })
        
        return pd.DataFrame(misclass_data)
    
    def generate_evaluation_summary(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation summary"""
        if self.evaluation_results is None:
            raise ValueError("Run evaluation first")
        
        basic = self.evaluation_results['basic_metrics']
        detailed = self.evaluation_results['detailed_metrics']
        
        # Find best and worst performing classes
        f1_scores = detailed['f1_per_class']
        best_classes = sorted(zip(self.class_names, f1_scores), key=lambda x: x[1], reverse=True)[:3]
        worst_classes = sorted(zip(self.class_names, f1_scores), key=lambda x: x[1])[:3]
        
        summary = {
            'overall_performance': {
                'accuracy': basic['accuracy'],
                'macro_f1': detailed['f1_macro'],
                'weighted_f1': detailed['f1_weighted'],
                'total_samples': basic['total_samples'],
                'num_classes': basic['num_classes']
            },
            'best_performing_classes': [{'class': name, 'f1_score': score} for name, score in best_classes],
            'worst_performing_classes': [{'class': name, 'f1_score': score} for name, score in worst_classes],
            'class_balance': {
                'support_std': np.std(detailed['support_per_class']),
                'support_mean': np.mean(detailed['support_per_class']),
                'min_support': min(detailed['support_per_class']),
                'max_support': max(detailed['support_per_class'])
            },
            'evaluation_time': self.evaluation_results['prediction_time'],
            'samples_per_second': basic['total_samples'] / self.evaluation_results['prediction_time']
        }
        
        return summary
    
    def save_evaluation_results(self, output_dir: str):
        """Save all evaluation results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results as JSON
        import json
        results_file = output_path / "evaluation_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_serializable(self.evaluation_results)
            json.dump(serializable_results, f, indent=2)
        
        # Save classification report
        if self.evaluation_results:
            report_df = self.generate_classification_report()
            report_df.to_csv(output_path / "classification_report.csv")
        
        # Save plots
        try:
            # Confusion matrix
            fig_cm = self.plot_confusion_matrix()
            fig_cm.savefig(output_path / "confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close(fig_cm)
            
            # Per-class metrics
            fig_metrics = self.plot_per_class_metrics()
            fig_metrics.savefig(output_path / "per_class_metrics.png", dpi=300, bbox_inches='tight')
            plt.close(fig_metrics)
            
            # ROC curves
            fig_roc = self.plot_roc_curves()
            fig_roc.savefig(output_path / "roc_curves.png", dpi=300, bbox_inches='tight')
            plt.close(fig_roc)
            
            # Precision-Recall curves
            fig_pr = self.plot_precision_recall_curves()
            fig_pr.savefig(output_path / "precision_recall_curves.png", dpi=300, bbox_inches='tight')
            plt.close(fig_pr)
            
        except Exception as e:
            logger.warning(f"Some plots could not be saved: {e}")
        
        # Save misclassification analysis
        try:
            misclass_df = self.analyze_misclassifications()
            misclass_df.to_csv(output_path / "misclassification_analysis.csv", index=False)
        except Exception as e:
            logger.warning(f"Misclassification analysis could not be saved: {e}")
        
        # Save summary
        summary = self.generate_evaluation_summary()
        summary_file = output_path / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Evaluation results saved to: {output_path}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj

def evaluate_model_from_path(model_path: str, dataset_path: str, output_dir: str = None):
    """
    Convenient function to evaluate a model from file paths
    
    Args:
        model_path: Path to the trained model
        dataset_path: Path to the dataset
        output_dir: Directory to save results (optional)
    """
    evaluator = ModelEvaluator(model_path, dataset_path)
    results = evaluator.evaluate_on_test_set()
    
    if output_dir:
        evaluator.save_evaluation_results(output_dir)
    
    return results