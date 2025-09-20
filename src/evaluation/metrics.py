"""
Metrics Calculator Module
========================

Comprehensive metrics calculation for fish species classification
including standard ML metrics and domain-specific evaluation measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, log_loss, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.preprocessing import LabelBinarizer
import warnings

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class MetricsCalculator:
    """Comprehensive metrics calculator for classification tasks"""
    
    def __init__(self):
        """Initialize metrics calculator"""
        self.metrics_cache = {}
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of basic metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred)
        }
    
    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  class_names: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names (optional)
            
        Returns:
            Dictionary of per-class metrics
        """
        # Get unique classes
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in unique_classes]
        
        # Calculate per-class precision, recall, f1
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Calculate confusion matrix for additional metrics
        cm = confusion_matrix(y_true, y_pred)
        
        per_class_metrics = {}
        
        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                # Basic metrics
                metrics = {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i]),
                    'support': int(np.sum(y_true == i))
                }
                
                # Additional metrics from confusion matrix
                if i < cm.shape[0]:
                    tp = cm[i, i]
                    fp = cm[:, i].sum() - tp
                    fn = cm[i, :].sum() - tp
                    tn = cm.sum() - tp - fp - fn
                    
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                    metrics.update({
                        'true_positives': int(tp),
                        'false_positives': int(fp),
                        'false_negatives': int(fn),
                        'true_negatives': int(tn),
                        'specificity': float(specificity),
                        'sensitivity': float(sensitivity)
                    })
                
                per_class_metrics[class_name] = metrics
        
        return per_class_metrics
    
    def calculate_probabilistic_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                       class_names: List[str] = None) -> Dict[str, Any]:
        """
        Calculate metrics that require probability predictions
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            class_names: List of class names (optional)
            
        Returns:
            Dictionary of probabilistic metrics
        """
        try:
            # Binarize labels for multi-class ROC-AUC
            lb = LabelBinarizer()
            y_true_binary = lb.fit_transform(y_true)
            
            # Handle binary classification case
            if y_true_binary.shape[1] == 1:
                y_true_binary = np.hstack([1 - y_true_binary, y_true_binary])
                y_prob_binary = np.hstack([1 - y_prob, y_prob])
            else:
                y_prob_binary = y_prob
            
            metrics = {}
            
            # Log loss
            try:
                metrics['log_loss'] = log_loss(y_true, y_prob)
            except Exception as e:
                logger.warning(f"Could not calculate log loss: {e}")
                metrics['log_loss'] = None
            
            # ROC-AUC scores
            try:
                # Macro-averaged ROC-AUC
                metrics['roc_auc_macro'] = roc_auc_score(y_true_binary, y_prob_binary, average='macro')
                
                # Weighted ROC-AUC
                metrics['roc_auc_weighted'] = roc_auc_score(y_true_binary, y_prob_binary, average='weighted')
                
                # Per-class ROC-AUC
                roc_auc_per_class = []
                for i in range(y_true_binary.shape[1]):
                    if len(np.unique(y_true_binary[:, i])) > 1:  # Check if class appears in both positive and negative
                        auc = roc_auc_score(y_true_binary[:, i], y_prob_binary[:, i])
                        roc_auc_per_class.append(auc)
                    else:
                        roc_auc_per_class.append(np.nan)
                
                metrics['roc_auc_per_class'] = roc_auc_per_class
                
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                metrics['roc_auc_macro'] = None
                metrics['roc_auc_weighted'] = None
                metrics['roc_auc_per_class'] = None
            
            # Average Precision scores
            try:
                # Macro-averaged AP
                ap_scores = []
                for i in range(y_true_binary.shape[1]):
                    if len(np.unique(y_true_binary[:, i])) > 1:
                        ap = average_precision_score(y_true_binary[:, i], y_prob_binary[:, i])
                        ap_scores.append(ap)
                
                metrics['average_precision_macro'] = np.mean(ap_scores) if ap_scores else None
                metrics['average_precision_per_class'] = ap_scores
                
            except Exception as e:
                logger.warning(f"Could not calculate Average Precision: {e}")
                metrics['average_precision_macro'] = None
                metrics['average_precision_per_class'] = None
            
            # Confidence statistics
            max_probs = np.max(y_prob, axis=1)
            metrics['confidence_mean'] = float(np.mean(max_probs))
            metrics['confidence_std'] = float(np.std(max_probs))
            metrics['confidence_min'] = float(np.min(max_probs))
            metrics['confidence_max'] = float(np.max(max_probs))
            
            # Prediction entropy
            epsilon = 1e-15
            y_prob_safe = np.clip(y_prob, epsilon, 1 - epsilon)
            entropy = -np.sum(y_prob_safe * np.log(y_prob_safe), axis=1)
            metrics['entropy_mean'] = float(np.mean(entropy))
            metrics['entropy_std'] = float(np.std(entropy))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating probabilistic metrics: {e}")
            return {}
    
    def calculate_confusion_matrix_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate metrics derived from confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of confusion matrix metrics
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Calculate per-class accuracy from diagonal
        per_class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
        
        # Overall metrics
        total_samples = np.sum(cm)
        correct_predictions = np.trace(cm)
        
        return {
            'confusion_matrix': cm.tolist(),
            'normalized_confusion_matrix': cm_normalized.tolist(),
            'per_class_accuracy': per_class_accuracy.tolist(),
            'total_samples': int(total_samples),
            'correct_predictions': int(correct_predictions),
            'misclassifications': int(total_samples - correct_predictions),
            'error_rate': float(1 - correct_predictions / total_samples)
        }
    
    def calculate_top_k_accuracy(self, y_true: np.ndarray, y_prob: np.ndarray, k: int = 5) -> float:
        """
        Calculate top-k accuracy
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy
        """
        # Get top-k predictions
        top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
        
        # Check if true label is in top-k predictions
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        
        return correct / len(y_true)
    
    def calculate_class_balance_metrics(self, y_true: np.ndarray, class_names: List[str] = None) -> Dict[str, Any]:
        """
        Calculate metrics related to class balance
        
        Args:
            y_true: True labels
            class_names: List of class names (optional)
            
        Returns:
            Dictionary of class balance metrics
        """
        unique_classes, counts = np.unique(y_true, return_counts=True)
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in unique_classes]
        
        # Calculate balance metrics
        total_samples = len(y_true)
        class_proportions = counts / total_samples
        
        # Gini coefficient for class imbalance
        def gini_coefficient(proportions):
            sorted_props = np.sort(proportions)
            n = len(sorted_props)
            cumulative = np.cumsum(sorted_props)
            return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
        
        gini = gini_coefficient(class_proportions)
        
        # Shannon entropy
        shannon_entropy = -np.sum(class_proportions * np.log2(class_proportions + 1e-15))
        max_entropy = np.log2(len(unique_classes))
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
        
        return {
            'num_classes': len(unique_classes),
            'total_samples': int(total_samples),
            'class_counts': {class_names[i]: int(counts[i]) for i in range(len(unique_classes))},
            'class_proportions': {class_names[i]: float(class_proportions[i]) for i in range(len(unique_classes))},
            'min_class_size': int(np.min(counts)),
            'max_class_size': int(np.max(counts)),
            'mean_class_size': float(np.mean(counts)),
            'std_class_size': float(np.std(counts)),
            'imbalance_ratio': float(np.max(counts) / np.min(counts)),
            'gini_coefficient': float(gini),
            'shannon_entropy': float(shannon_entropy),
            'normalized_entropy': float(normalized_entropy)
        }
    
    def calculate_hierarchical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     class_hierarchy: Dict[str, List[str]] = None) -> Dict[str, float]:
        """
        Calculate hierarchical classification metrics (if applicable)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_hierarchy: Dictionary defining class hierarchy
            
        Returns:
            Dictionary of hierarchical metrics
        """
        if class_hierarchy is None:
            logger.info("No class hierarchy provided, skipping hierarchical metrics")
            return {}
        
        # This is a placeholder for hierarchical metrics
        # Implementation would depend on specific hierarchy structure
        hierarchical_metrics = {
            'hierarchical_precision': 0.0,
            'hierarchical_recall': 0.0,
            'hierarchical_f1': 0.0
        }
        
        return hierarchical_metrics
    
    def calculate_domain_specific_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                        y_prob: np.ndarray = None, 
                                        class_names: List[str] = None) -> Dict[str, Any]:
        """
        Calculate domain-specific metrics for fish species classification
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            class_names: List of class names (optional)
            
        Returns:
            Dictionary of domain-specific metrics
        """
        metrics = {}
        
        # Calculate misidentification patterns
        misclass_matrix = confusion_matrix(y_true, y_pred)
        
        # Find most commonly confused species pairs
        confused_pairs = []
        for i in range(misclass_matrix.shape[0]):
            for j in range(misclass_matrix.shape[1]):
                if i != j and misclass_matrix[i, j] > 0:
                    if class_names:
                        pair = (class_names[i], class_names[j], int(misclass_matrix[i, j]))
                    else:
                        pair = (f"Class_{i}", f"Class_{j}", int(misclass_matrix[i, j]))
                    confused_pairs.append(pair)
        
        # Sort by confusion count
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        metrics['most_confused_pairs'] = confused_pairs[:10]  # Top 10
        
        # Confidence-based metrics
        if y_prob is not None:
            # High confidence misclassifications (concerning)
            max_probs = np.max(y_prob, axis=1)
            high_conf_threshold = 0.8
            
            high_conf_mask = max_probs > high_conf_threshold
            high_conf_correct = np.sum((y_true == y_pred) & high_conf_mask)
            high_conf_incorrect = np.sum((y_true != y_pred) & high_conf_mask)
            
            metrics['high_confidence_correct'] = int(high_conf_correct)
            metrics['high_confidence_incorrect'] = int(high_conf_incorrect)
            metrics['high_confidence_error_rate'] = float(
                high_conf_incorrect / (high_conf_correct + high_conf_incorrect)
                if (high_conf_correct + high_conf_incorrect) > 0 else 0
            )
            
            # Low confidence correct predictions (good uncertainty)
            low_conf_threshold = 0.5
            low_conf_mask = max_probs < low_conf_threshold
            low_conf_correct = np.sum((y_true == y_pred) & low_conf_mask)
            
            metrics['low_confidence_correct'] = int(low_conf_correct)
        
        # Rare species performance
        class_counts = np.bincount(y_true)
        rare_threshold = np.percentile(class_counts, 25)  # Bottom 25% are considered rare
        
        rare_classes = np.where(class_counts <= rare_threshold)[0]
        rare_mask = np.isin(y_true, rare_classes)
        
        if np.sum(rare_mask) > 0:
            rare_accuracy = accuracy_score(y_true[rare_mask], y_pred[rare_mask])
            metrics['rare_species_accuracy'] = float(rare_accuracy)
            metrics['rare_species_count'] = len(rare_classes)
            metrics['rare_species_samples'] = int(np.sum(rare_mask))
        
        return metrics
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_prob: np.ndarray = None, 
                                      class_names: List[str] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics suite
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            class_names: List of class names (optional)
            
        Returns:
            Dictionary containing all calculated metrics
        """
        logger.info("Calculating comprehensive metrics...")
        
        all_metrics = {}
        
        try:
            # Basic metrics
            all_metrics['basic'] = self.calculate_basic_metrics(y_true, y_pred)
            
            # Per-class metrics
            all_metrics['per_class'] = self.calculate_per_class_metrics(y_true, y_pred, class_names)
            
            # Confusion matrix metrics
            all_metrics['confusion_matrix'] = self.calculate_confusion_matrix_metrics(y_true, y_pred)
            
            # Class balance metrics
            all_metrics['class_balance'] = self.calculate_class_balance_metrics(y_true, class_names)
            
            # Probabilistic metrics (if probabilities provided)
            if y_prob is not None:
                all_metrics['probabilistic'] = self.calculate_probabilistic_metrics(y_true, y_prob, class_names)
                
                # Top-k accuracies
                all_metrics['top_k'] = {}
                for k in [3, 5, 10]:
                    if k <= y_prob.shape[1]:
                        all_metrics['top_k'][f'top_{k}_accuracy'] = self.calculate_top_k_accuracy(y_true, y_prob, k)
            
            # Domain-specific metrics
            all_metrics['domain_specific'] = self.calculate_domain_specific_metrics(
                y_true, y_pred, y_prob, class_names
            )
            
            logger.info("Comprehensive metrics calculation completed")
            
        except Exception as e:
            logger.error(f"Error in comprehensive metrics calculation: {e}")
            
        return all_metrics
    
    def metrics_to_dataframe(self, metrics: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert metrics dictionary to pandas DataFrame for easy viewing
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            DataFrame with metrics
        """
        rows = []
        
        def flatten_dict(d, prefix=''):
            for key, value in d.items():
                new_key = f"{prefix}_{key}" if prefix else key
                
                if isinstance(value, dict):
                    rows.extend(flatten_dict(value, new_key))
                elif isinstance(value, (list, np.ndarray)):
                    if len(value) <= 10:  # Only include short lists
                        rows.append({'metric': new_key, 'value': str(value)})
                else:
                    rows.append({'metric': new_key, 'value': value})
            
            return rows
        
        flatten_dict(metrics)
        return pd.DataFrame(rows)