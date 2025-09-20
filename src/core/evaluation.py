"""
Evaluation module for fish species classification models.

This module provides comprehensive model evaluation functionality including
performance metrics, confusion matrices, classification reports, and
visualization capabilities.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path

import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, top_k_accuracy_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.logger import get_logger
from ..utils.exceptions import EvaluationError
from ..utils.visualization import save_confusion_matrix, save_classification_report
from config.settings import get_settings


class EvaluationMetrics:
    """Container for comprehensive evaluation metrics."""
    
    def __init__(self):
        self.accuracy: float = 0.0
        self.top_3_accuracy: float = 0.0
        self.top_5_accuracy: float = 0.0
        self.precision: Dict[str, float] = {}
        self.recall: Dict[str, float] = {}
        self.f1_score: Dict[str, float] = {}
        self.confusion_matrix: np.ndarray = np.array([])
        self.classification_report: Dict[str, Any] = {}
        self.per_class_metrics: Dict[str, Dict[str, float]] = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'accuracy': float(self.accuracy),
            'top_3_accuracy': float(self.top_3_accuracy),
            'top_5_accuracy': float(self.top_5_accuracy),
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix.size > 0 else [],
            'classification_report': self.classification_report,
            'per_class_metrics': self.per_class_metrics
        }


class ModelEvaluator:
    """
    Comprehensive model evaluation with detailed metrics and visualizations.
    
    Features:
    - Multiple evaluation metrics
    - Confusion matrix generation
    - Per-class performance analysis
    - Visualization capabilities
    - Report generation
    """
    
    def __init__(self, 
                 model: tf.keras.Model,
                 class_names: List[str],
                 model_name: str = "fish_classifier"):
        """
        Initialize the model evaluator.
        
        Args:
            model: Trained Keras model
            class_names: List of class names
            model_name: Name for saving evaluation results
        """
        self.model = model
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model_name = model_name
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Results storage
        self.evaluation_metrics = EvaluationMetrics()
        self.results_dir = Path(self.settings.RESULTS_DIR) / model_name
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Create necessary directories for evaluation results."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(self,
                test_data: tf.data.Dataset,
                batch_size: int = 32,
                verbose: int = 1) -> EvaluationMetrics:
        """
        Perform comprehensive model evaluation.
        
        Args:
            test_data: Test dataset
            batch_size: Batch size for evaluation
            verbose: Verbosity level
            
        Returns:
            EvaluationMetrics object with all evaluation results
        """
        try:
            self.logger.info("Starting model evaluation...")
            
            # Get predictions and true labels
            y_true, y_pred, y_pred_proba = self._get_predictions(
                test_data, batch_size, verbose
            )
            
            # Calculate all metrics
            self._calculate_metrics(y_true, y_pred, y_pred_proba)
            
            # Generate visualizations
            self._generate_visualizations(y_true, y_pred)
            
            # Save results
            self._save_evaluation_results()
            
            self.logger.info("Model evaluation completed successfully")
            return self.evaluation_metrics
            
        except Exception as e:
            raise EvaluationError(f"Evaluation failed: {str(e)}")
    
    def _get_predictions(self,
                        test_data: tf.data.Dataset,
                        batch_size: int,
                        verbose: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get model predictions for the test dataset.
        
        Returns:
            Tuple of (true_labels, predicted_labels, predicted_probabilities)
        """
        try:
            # Collect all true labels and predictions
            y_true_list = []
            y_pred_proba_list = []
            
            # Process in batches
            for batch_x, batch_y in test_data:
                # Get true labels
                if len(batch_y.shape) > 1:  # One-hot encoded
                    y_true_batch = np.argmax(batch_y.numpy(), axis=1)
                else:  # Integer labels
                    y_true_batch = batch_y.numpy()
                y_true_list.append(y_true_batch)
                
                # Get predictions
                y_pred_proba_batch = self.model.predict(batch_x, verbose=0)
                y_pred_proba_list.append(y_pred_proba_batch)
            
            # Concatenate all batches
            y_true = np.concatenate(y_true_list)
            y_pred_proba = np.concatenate(y_pred_proba_list)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            self.logger.info(f"Generated predictions for {len(y_true)} samples")
            return y_true, y_pred, y_pred_proba
            
        except Exception as e:
            raise EvaluationError(f"Failed to get predictions: {str(e)}")
    
    def _calculate_metrics(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_pred_proba: np.ndarray):
        """Calculate all evaluation metrics."""
        try:
            # Basic accuracy
            self.evaluation_metrics.accuracy = accuracy_score(y_true, y_pred)
            
            # Top-k accuracy
            if self.num_classes >= 3:
                self.evaluation_metrics.top_3_accuracy = top_k_accuracy_score(
                    y_true, y_pred_proba, k=3
                )
            if self.num_classes >= 5:
                self.evaluation_metrics.top_5_accuracy = top_k_accuracy_score(
                    y_true, y_pred_proba, k=5
                )
            
            # Precision, recall, f1-score
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            
            # Store per-class metrics
            for i, class_name in enumerate(self.class_names):
                self.evaluation_metrics.per_class_metrics[class_name] = {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i])
                }
            
            # Average metrics
            self.evaluation_metrics.precision = {
                'macro': float(np.mean(precision)),
                'weighted': float(np.average(precision, weights=support))
            }
            self.evaluation_metrics.recall = {
                'macro': float(np.mean(recall)),
                'weighted': float(np.average(recall, weights=support))
            }
            self.evaluation_metrics.f1_score = {
                'macro': float(np.mean(f1)),
                'weighted': float(np.average(f1, weights=support))
            }
            
            # Confusion matrix
            self.evaluation_metrics.confusion_matrix = confusion_matrix(y_true, y_pred)
            
            # Classification report
            self.evaluation_metrics.classification_report = classification_report(
                y_true, y_pred, target_names=self.class_names, output_dict=True
            )
            
            self.logger.info(f"Calculated metrics - Accuracy: {self.evaluation_metrics.accuracy:.4f}")
            
        except Exception as e:
            raise EvaluationError(f"Failed to calculate metrics: {str(e)}")
    
    def _generate_visualizations(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray):
        """Generate evaluation visualizations."""
        try:
            # Confusion matrix heatmap
            self._plot_confusion_matrix(y_true, y_pred)
            
            # Per-class performance chart
            self._plot_per_class_metrics()
            
            # Accuracy distribution
            self._plot_accuracy_distribution()
            
            self.logger.info("Generated evaluation visualizations")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate some visualizations: {str(e)}")
    
    def _plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             normalize: bool = True):
        """Plot and save confusion matrix."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            if normalize:
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero
            else:
                cm_norm = cm
            
            # Create figure
            plt.figure(figsize=(max(12, self.num_classes * 0.8), 
                               max(10, self.num_classes * 0.8)))
            
            # Plot heatmap
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt='.2f' if normalize else 'd',
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
            )
            
            plt.title(f'Confusion Matrix - {self.model_name}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save plot
            save_path = self.results_dir / 'confusion_matrix.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Confusion matrix saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to plot confusion matrix: {str(e)}")
    
    def _plot_per_class_metrics(self):
        """Plot per-class performance metrics."""
        try:
            metrics_data = []
            for class_name, metrics in self.evaluation_metrics.per_class_metrics.items():
                metrics_data.append({
                    'Class': class_name,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score']
                })
            
            df = pd.DataFrame(metrics_data)
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot each metric
            metrics = ['Precision', 'Recall', 'F1-Score']
            for i, metric in enumerate(metrics):
                df_sorted = df.sort_values(metric, ascending=False)
                
                bars = axes[i].bar(range(len(df_sorted)), df_sorted[metric], 
                                  color=plt.cm.viridis(i/3))
                axes[i].set_title(f'{metric} by Class')
                axes[i].set_xlabel('Class')
                axes[i].set_ylabel(metric)
                axes[i].set_xticks(range(len(df_sorted)))
                axes[i].set_xticklabels(df_sorted['Class'], rotation=45, ha='right')
                axes[i].set_ylim(0, 1)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            # Save plot
            save_path = self.results_dir / 'per_class_metrics.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Per-class metrics plot saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to plot per-class metrics: {str(e)}")
    
    def _plot_accuracy_distribution(self):
        """Plot accuracy distribution and summary statistics."""
        try:
            # Extract per-class accuracies (diagonal of normalized confusion matrix)
            cm = self.evaluation_metrics.confusion_matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)
            per_class_accuracy = np.diag(cm_norm)
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Histogram
            plt.subplot(2, 1, 1)
            plt.hist(per_class_accuracy, bins=min(20, self.num_classes), 
                    alpha=0.7, edgecolor='black')
            plt.title('Distribution of Per-Class Accuracies')
            plt.xlabel('Accuracy')
            plt.ylabel('Number of Classes')
            plt.axvline(np.mean(per_class_accuracy), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(per_class_accuracy):.3f}')
            plt.legend()
            
            # Box plot
            plt.subplot(2, 1, 2)
            plt.boxplot(per_class_accuracy, vert=False)
            plt.xlabel('Accuracy')
            plt.title('Per-Class Accuracy Distribution')
            
            plt.tight_layout()
            
            # Save plot
            save_path = self.results_dir / 'accuracy_distribution.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Accuracy distribution plot saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to plot accuracy distribution: {str(e)}")
    
    def _save_evaluation_results(self):
        """Save evaluation results to files."""
        try:
            # Save metrics as JSON
            metrics_path = self.results_dir / 'evaluation_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(self.evaluation_metrics.to_dict(), f, indent=2)
            
            # Save classification report as CSV
            report_df = pd.DataFrame(self.evaluation_metrics.classification_report).transpose()
            report_path = self.results_dir / 'classification_report.csv'
            report_df.to_csv(report_path)
            
            # Save confusion matrix as CSV
            cm_df = pd.DataFrame(
                self.evaluation_metrics.confusion_matrix,
                index=self.class_names,
                columns=self.class_names
            )
            cm_path = self.results_dir / 'confusion_matrix.csv'
            cm_df.to_csv(cm_path)
            
            # Save summary report
            self._save_summary_report()
            
            self.logger.info(f"Evaluation results saved to: {self.results_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save evaluation results: {str(e)}")
    
    def _save_summary_report(self):
        """Save a comprehensive summary report."""
        try:
            summary = {
                'Model Information': {
                    'Model Name': self.model_name,
                    'Number of Classes': self.num_classes,
                    'Total Parameters': self.model.count_params()
                },
                'Overall Performance': {
                    'Accuracy': f"{self.evaluation_metrics.accuracy:.4f}",
                    'Top-3 Accuracy': f"{self.evaluation_metrics.top_3_accuracy:.4f}",
                    'Top-5 Accuracy': f"{self.evaluation_metrics.top_5_accuracy:.4f}",
                    'Macro Precision': f"{self.evaluation_metrics.precision.get('macro', 0):.4f}",
                    'Macro Recall': f"{self.evaluation_metrics.recall.get('macro', 0):.4f}",
                    'Macro F1-Score': f"{self.evaluation_metrics.f1_score.get('macro', 0):.4f}"
                },
                'Best Performing Classes': self._get_top_classes('f1_score', top_k=5),
                'Worst Performing Classes': self._get_bottom_classes('f1_score', bottom_k=5),
                'Class Distribution': self._get_class_distribution()
            }
            
            # Save as formatted text
            summary_path = self.results_dir / 'evaluation_summary.txt'
            with open(summary_path, 'w') as f:
                f.write(f"Evaluation Summary - {self.model_name}\n")
                f.write("=" * 50 + "\n\n")
                
                for section, content in summary.items():
                    f.write(f"{section}:\n")
                    f.write("-" * len(section) + "\n")
                    
                    if isinstance(content, dict):
                        for key, value in content.items():
                            f.write(f"  {key}: {value}\n")
                    elif isinstance(content, list):
                        for item in content:
                            f.write(f"  {item}\n")
                    else:
                        f.write(f"  {content}\n")
                    f.write("\n")
            
            self.logger.info(f"Summary report saved to: {summary_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save summary report: {str(e)}")
    
    def _get_top_classes(self, metric: str, top_k: int = 5) -> List[str]:
        """Get top performing classes for a given metric."""
        try:
            class_scores = []
            for class_name, metrics in self.evaluation_metrics.per_class_metrics.items():
                class_scores.append((class_name, metrics[metric]))
            
            class_scores.sort(key=lambda x: x[1], reverse=True)
            return [f"{name}: {score:.4f}" for name, score in class_scores[:top_k]]
        except:
            return []
    
    def _get_bottom_classes(self, metric: str, bottom_k: int = 5) -> List[str]:
        """Get worst performing classes for a given metric."""
        try:
            class_scores = []
            for class_name, metrics in self.evaluation_metrics.per_class_metrics.items():
                class_scores.append((class_name, metrics[metric]))
            
            class_scores.sort(key=lambda x: x[1])
            return [f"{name}: {score:.4f}" for name, score in class_scores[:bottom_k]]
        except:
            return []
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution from confusion matrix."""
        try:
            cm = self.evaluation_metrics.confusion_matrix
            class_counts = np.sum(cm, axis=1)
            return {self.class_names[i]: int(count) for i, count in enumerate(class_counts)}
        except:
            return {}
    
    def compare_models(self, 
                      other_evaluator: 'ModelEvaluator',
                      save_comparison: bool = True) -> Dict[str, Any]:
        """
        Compare this model's performance with another model.
        
        Args:
            other_evaluator: Another ModelEvaluator instance
            save_comparison: Whether to save comparison results
            
        Returns:
            Comparison results dictionary
        """
        try:
            comparison = {
                'models': [self.model_name, other_evaluator.model_name],
                'accuracy': [
                    self.evaluation_metrics.accuracy,
                    other_evaluator.evaluation_metrics.accuracy
                ],
                'top_3_accuracy': [
                    self.evaluation_metrics.top_3_accuracy,
                    other_evaluator.evaluation_metrics.top_3_accuracy
                ],
                'macro_f1': [
                    self.evaluation_metrics.f1_score.get('macro', 0),
                    other_evaluator.evaluation_metrics.f1_score.get('macro', 0)
                ],
                'winner': {
                    'accuracy': self.model_name if self.evaluation_metrics.accuracy > 
                               other_evaluator.evaluation_metrics.accuracy else other_evaluator.model_name,
                    'top_3_accuracy': self.model_name if self.evaluation_metrics.top_3_accuracy > 
                                     other_evaluator.evaluation_metrics.top_3_accuracy else other_evaluator.model_name,
                    'macro_f1': self.model_name if self.evaluation_metrics.f1_score.get('macro', 0) > 
                               other_evaluator.evaluation_metrics.f1_score.get('macro', 0) else other_evaluator.model_name
                }
            }
            
            if save_comparison:
                comparison_path = self.results_dir / 'model_comparison.json'
                with open(comparison_path, 'w') as f:
                    json.dump(comparison, f, indent=2)
                
                self.logger.info(f"Model comparison saved to: {comparison_path}")
            
            return comparison
            
        except Exception as e:
            raise EvaluationError(f"Model comparison failed: {str(e)}")


def evaluate_multiple_models(models_info: List[Dict[str, Any]],
                           test_data: tf.data.Dataset,
                           class_names: List[str]) -> pd.DataFrame:
    """
    Evaluate multiple models and return comparison results.
    
    Args:
        models_info: List of dictionaries with 'model', 'name' keys
        test_data: Test dataset
        class_names: List of class names
        
    Returns:
        DataFrame with comparison results
    """
    try:
        results = []
        
        for model_info in models_info:
            model = model_info['model']
            name = model_info['name']
            
            evaluator = ModelEvaluator(model, class_names, name)
            metrics = evaluator.evaluate(test_data)
            
            results.append({
                'Model': name,
                'Accuracy': metrics.accuracy,
                'Top-3 Accuracy': metrics.top_3_accuracy,
                'Top-5 Accuracy': metrics.top_5_accuracy,
                'Macro Precision': metrics.precision.get('macro', 0),
                'Macro Recall': metrics.recall.get('macro', 0),
                'Macro F1-Score': metrics.f1_score.get('macro', 0),
                'Parameters': model.count_params()
            })
        
        df = pd.DataFrame(results)
        return df.sort_values('Accuracy', ascending=False)
        
    except Exception as e:
        raise EvaluationError(f"Multiple model evaluation failed: {str(e)}")