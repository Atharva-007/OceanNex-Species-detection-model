"""
Performance Analyzer Module
===========================

Advanced performance analysis tools for fish species classification models
including error analysis, feature importance, and prediction insights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class PerformanceAnalyzer:
    """Advanced performance analysis and insights"""
    
    def __init__(self, output_dir: str = "performance_analysis"):
        """
        Initialize performance analyzer
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.analysis_results = {}
    
    def analyze_prediction_patterns(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_prob: np.ndarray, class_names: List[str] = None) -> Dict[str, Any]:
        """
        Analyze prediction patterns and confidence distributions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            class_names: List of class names
            
        Returns:
            Dictionary with pattern analysis results
        """
        logger.info("Analyzing prediction patterns...")
        
        results = {}
        
        # Prediction confidence analysis
        max_probs = np.max(y_prob, axis=1)
        predicted_classes = np.argmax(y_prob, axis=1)
        correct_mask = (y_true == y_pred)
        
        # Confidence distribution for correct vs incorrect predictions
        correct_confidence = max_probs[correct_mask]
        incorrect_confidence = max_probs[~correct_mask]
        
        results['confidence_analysis'] = {
            'correct_predictions': {
                'mean_confidence': float(np.mean(correct_confidence)),
                'std_confidence': float(np.std(correct_confidence)),
                'median_confidence': float(np.median(correct_confidence)),
                'min_confidence': float(np.min(correct_confidence)),
                'max_confidence': float(np.max(correct_confidence))
            },
            'incorrect_predictions': {
                'mean_confidence': float(np.mean(incorrect_confidence)),
                'std_confidence': float(np.std(incorrect_confidence)),
                'median_confidence': float(np.median(incorrect_confidence)),
                'min_confidence': float(np.min(incorrect_confidence)),
                'max_confidence': float(np.max(incorrect_confidence))
            }
        }
        
        # Confidence thresholds analysis
        thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_analysis = []
        
        for threshold in thresholds:
            high_conf_mask = max_probs >= threshold
            if np.sum(high_conf_mask) > 0:
                high_conf_accuracy = np.mean(correct_mask[high_conf_mask])
                coverage = np.sum(high_conf_mask) / len(y_true)
                
                threshold_analysis.append({
                    'threshold': float(threshold),
                    'accuracy': float(high_conf_accuracy),
                    'coverage': float(coverage),
                    'samples': int(np.sum(high_conf_mask))
                })
        
        results['threshold_analysis'] = threshold_analysis
        
        # Prediction entropy analysis
        epsilon = 1e-15
        y_prob_safe = np.clip(y_prob, epsilon, 1 - epsilon)
        entropy = -np.sum(y_prob_safe * np.log(y_prob_safe), axis=1)
        
        results['entropy_analysis'] = {
            'mean_entropy': float(np.mean(entropy)),
            'std_entropy': float(np.std(entropy)),
            'correct_predictions_entropy': float(np.mean(entropy[correct_mask])),
            'incorrect_predictions_entropy': float(np.mean(entropy[~correct_mask]))
        }
        
        # Class-wise confidence analysis
        if class_names is not None:
            class_confidence = {}
            for i, class_name in enumerate(class_names):
                class_mask = (y_true == i)
                if np.sum(class_mask) > 0:
                    class_predictions = y_pred[class_mask]
                    class_probabilities = y_prob[class_mask]
                    class_max_probs = np.max(class_probabilities, axis=1)
                    
                    class_confidence[class_name] = {
                        'mean_confidence': float(np.mean(class_max_probs)),
                        'std_confidence': float(np.std(class_max_probs)),
                        'accuracy': float(np.mean(class_predictions == i)),
                        'samples': int(np.sum(class_mask))
                    }
            
            results['class_confidence'] = class_confidence
        
        return results
    
    def analyze_misclassifications(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_prob: np.ndarray, class_names: List[str] = None,
                                 sample_indices: np.ndarray = None) -> Dict[str, Any]:
        """
        Detailed analysis of misclassifications
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            class_names: List of class names
            sample_indices: Original sample indices (optional)
            
        Returns:
            Dictionary with misclassification analysis
        """
        logger.info("Analyzing misclassifications...")
        
        results = {}
        
        # Find misclassified samples
        misclassified_mask = (y_true != y_pred)
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) == 0:
            logger.info("No misclassifications found!")
            return results
        
        # Basic misclassification statistics
        results['basic_stats'] = {
            'total_misclassifications': int(len(misclassified_indices)),
            'misclassification_rate': float(len(misclassified_indices) / len(y_true)),
            'total_samples': int(len(y_true))
        }
        
        # Confusion pairs analysis
        confusion_pairs = []
        cm = confusion_matrix(y_true, y_pred)
        
        for true_class in range(cm.shape[0]):
            for pred_class in range(cm.shape[1]):
                if true_class != pred_class and cm[true_class, pred_class] > 0:
                    true_name = class_names[true_class] if class_names else f"Class_{true_class}"
                    pred_name = class_names[pred_class] if class_names else f"Class_{pred_class}"
                    
                    confusion_pairs.append({
                        'true_class': true_name,
                        'predicted_class': pred_name,
                        'count': int(cm[true_class, pred_class]),
                        'percentage': float(cm[true_class, pred_class] / np.sum(cm[true_class]) * 100)
                    })
        
        # Sort by frequency
        confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
        results['confusion_pairs'] = confusion_pairs[:20]  # Top 20
        
        # Confidence analysis for misclassifications
        misclass_probs = y_prob[misclassified_mask]
        misclass_confidence = np.max(misclass_probs, axis=1)
        
        results['misclassification_confidence'] = {
            'mean_confidence': float(np.mean(misclass_confidence)),
            'std_confidence': float(np.std(misclass_confidence)),
            'median_confidence': float(np.median(misclass_confidence)),
            'high_confidence_errors': int(np.sum(misclass_confidence > 0.8)),
            'low_confidence_errors': int(np.sum(misclass_confidence < 0.5))
        }
        
        # Most confident misclassifications (concerning errors)
        confident_errors_mask = misclass_confidence > 0.8
        confident_error_indices = misclassified_indices[confident_errors_mask]
        
        if len(confident_error_indices) > 0:
            confident_errors = []
            for idx in confident_error_indices[:10]:  # Top 10
                error_info = {
                    'sample_index': int(sample_indices[idx] if sample_indices is not None else idx),
                    'true_class': class_names[y_true[idx]] if class_names else int(y_true[idx]),
                    'predicted_class': class_names[y_pred[idx]] if class_names else int(y_pred[idx]),
                    'confidence': float(misclass_confidence[confident_errors_mask][list(confident_error_indices).index(idx)]),
                    'probabilities': y_prob[idx].tolist()
                }
                confident_errors.append(error_info)
            
            results['high_confidence_errors'] = confident_errors
        
        # Class-wise error analysis
        if class_names is not None:
            class_errors = {}
            for i, class_name in enumerate(class_names):
                true_class_mask = (y_true == i)
                pred_class_mask = (y_pred == i)
                
                # Errors where this class was the true class
                true_class_errors = misclassified_mask & true_class_mask
                # Errors where this class was predicted
                pred_class_errors = misclassified_mask & pred_class_mask
                
                class_errors[class_name] = {
                    'missed_detections': int(np.sum(true_class_errors)),
                    'false_positives': int(np.sum(pred_class_errors)),
                    'total_true_samples': int(np.sum(true_class_mask)),
                    'total_predicted_samples': int(np.sum(pred_class_mask)),
                    'miss_rate': float(np.sum(true_class_errors) / np.sum(true_class_mask)) if np.sum(true_class_mask) > 0 else 0.0,
                    'false_positive_rate': float(np.sum(pred_class_errors) / np.sum(pred_class_mask)) if np.sum(pred_class_mask) > 0 else 0.0
                }
            
            results['class_wise_errors'] = class_errors
        
        return results
    
    def analyze_feature_importance(self, features: np.ndarray, y_true: np.ndarray, 
                                 y_pred: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Analyze feature importance and dimensionality reduction
        
        Args:
            features: Feature vectors
            y_true: True labels
            y_pred: Predicted labels
            feature_names: Names of features (optional)
            
        Returns:
            Dictionary with feature analysis results
        """
        logger.info("Analyzing feature importance...")
        
        results = {}
        
        try:
            # PCA analysis
            pca = PCA(n_components=min(50, features.shape[1]))
            features_pca = pca.fit_transform(features)
            
            results['pca_analysis'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'n_components_95_variance': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1),
                'n_components_99_variance': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.99) + 1)
            }
            
            # t-SNE analysis (on subset for performance)
            if features.shape[0] > 1000:
                sample_indices = np.random.choice(features.shape[0], 1000, replace=False)
                features_subset = features[sample_indices]
                labels_subset = y_true[sample_indices]
            else:
                features_subset = features
                labels_subset = y_true
            
            # Use PCA features for t-SNE
            pca_subset = PCA(n_components=min(50, features_subset.shape[1]))
            features_pca_subset = pca_subset.fit_transform(features_subset)
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_subset)//4))
            features_tsne = tsne.fit_transform(features_pca_subset)
            
            results['tsne_coordinates'] = {
                'coordinates': features_tsne.tolist(),
                'labels': labels_subset.tolist(),
                'sample_indices': sample_indices.tolist() if features.shape[0] > 1000 else list(range(len(features)))
            }
            
        except Exception as e:
            logger.warning(f"Error in dimensionality reduction analysis: {e}")
            results['dimensionality_reduction_error'] = str(e)
        
        return results
    
    def analyze_class_separability(self, features: np.ndarray, y_true: np.ndarray, 
                                 class_names: List[str] = None) -> Dict[str, Any]:
        """
        Analyze class separability in feature space
        
        Args:
            features: Feature vectors
            y_true: True labels
            class_names: List of class names
            
        Returns:
            Dictionary with separability analysis
        """
        logger.info("Analyzing class separability...")
        
        results = {}
        
        try:
            unique_classes = np.unique(y_true)
            n_classes = len(unique_classes)
            
            # Calculate class centroids
            class_centroids = {}
            class_statistics = {}
            
            for class_idx in unique_classes:
                class_mask = (y_true == class_idx)
                class_features = features[class_mask]
                
                class_name = class_names[class_idx] if class_names else f"Class_{class_idx}"
                
                centroid = np.mean(class_features, axis=0)
                class_centroids[class_name] = centroid.tolist()
                
                # Class statistics
                class_statistics[class_name] = {
                    'mean_distance_to_centroid': float(np.mean(np.linalg.norm(class_features - centroid, axis=1))),
                    'std_distance_to_centroid': float(np.std(np.linalg.norm(class_features - centroid, axis=1))),
                    'samples_count': int(np.sum(class_mask))
                }
            
            results['class_centroids'] = class_centroids
            results['class_statistics'] = class_statistics
            
            # Pairwise class distances
            pairwise_distances = {}
            class_names_list = list(class_centroids.keys())
            
            for i, class1 in enumerate(class_names_list):
                for j, class2 in enumerate(class_names_list[i+1:], i+1):
                    centroid1 = np.array(class_centroids[class1])
                    centroid2 = np.array(class_centroids[class2])
                    distance = float(np.linalg.norm(centroid1 - centroid2))
                    
                    pairwise_distances[f"{class1}_vs_{class2}"] = distance
            
            results['pairwise_centroid_distances'] = pairwise_distances
            
            # Find most and least separable class pairs
            sorted_distances = sorted(pairwise_distances.items(), key=lambda x: x[1])
            results['least_separable_pairs'] = sorted_distances[:5]  # 5 closest pairs
            results['most_separable_pairs'] = sorted_distances[-5:]  # 5 most distant pairs
            
            # Overall separability metrics
            all_distances = list(pairwise_distances.values())
            results['separability_metrics'] = {
                'mean_pairwise_distance': float(np.mean(all_distances)),
                'std_pairwise_distance': float(np.std(all_distances)),
                'min_pairwise_distance': float(np.min(all_distances)),
                'max_pairwise_distance': float(np.max(all_distances))
            }
            
        except Exception as e:
            logger.error(f"Error in class separability analysis: {e}")
            results['separability_error'] = str(e)
        
        return results
    
    def create_error_analysis_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                           y_prob: np.ndarray, class_names: List[str] = None,
                                           save_plots: bool = True) -> Dict[str, plt.Figure]:
        """
        Create visualizations for error analysis
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            class_names: List of class names
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary of matplotlib figures
        """
        figures = {}
        
        # 1. Confidence distribution for correct vs incorrect predictions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        max_probs = np.max(y_prob, axis=1)
        correct_mask = (y_true == y_pred)
        
        correct_confidence = max_probs[correct_mask]
        incorrect_confidence = max_probs[~correct_mask]
        
        ax1.hist(correct_confidence, bins=30, alpha=0.7, label='Correct', color='green')
        ax1.hist(incorrect_confidence, bins=30, alpha=0.7, label='Incorrect', color='red')
        ax1.set_xlabel('Prediction Confidence')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Confidence Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Confidence vs Accuracy
        confidence_bins = np.arange(0, 1.1, 0.1)
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(confidence_bins)-1):
            mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i+1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(correct_mask[mask])
                bin_accuracies.append(bin_accuracy)
                bin_counts.append(np.sum(mask))
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        ax2.plot(bin_centers, bin_accuracies, 'b-o', label='Accuracy')
        ax2.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        ax2.set_xlabel('Confidence Bin')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Calibration Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['confidence_analysis'] = fig
        
        if save_plots:
            fig.savefig(self.output_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        
        # 2. Confusion matrix with error highlighting
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts
        im1 = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax1.set_title('Confusion Matrix (Counts)')
        plt.colorbar(im1, ax=ax1)
        
        # Normalized
        im2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax2.set_title('Confusion Matrix (Normalized)')
        plt.colorbar(im2, ax=ax2)
        
        # Add text annotations for significant errors
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > np.max(cm) * 0.05:  # Significant errors
                    ax1.text(j, i, cm[i, j], ha="center", va="center", color="red", fontweight='bold')
                    ax2.text(j, i, f'{cm_normalized[i, j]:.2f}', ha="center", va="center", 
                            color="red", fontweight='bold')
        
        if class_names and len(class_names) <= 20:  # Only show labels for reasonable number of classes
            ax1.set_xticks(range(len(class_names)))
            ax1.set_yticks(range(len(class_names)))
            ax1.set_xticklabels(class_names, rotation=45)
            ax1.set_yticklabels(class_names)
            
            ax2.set_xticks(range(len(class_names)))
            ax2.set_yticks(range(len(class_names)))
            ax2.set_xticklabels(class_names, rotation=45)
            ax2.set_yticklabels(class_names)
        
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        figures['confusion_matrices'] = fig
        
        if save_plots:
            fig.savefig(self.output_dir / 'detailed_confusion_matrices.png', dpi=300, bbox_inches='tight')
        
        # 3. Per-class error analysis
        if class_names is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # False positive and false negative rates
            class_fp_rates = []
            class_fn_rates = []
            
            for i, class_name in enumerate(class_names):
                true_positives = cm[i, i]
                false_positives = cm[:, i].sum() - true_positives
                false_negatives = cm[i, :].sum() - true_positives
                true_negatives = cm.sum() - true_positives - false_positives - false_negatives
                
                fp_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
                fn_rate = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0
                
                class_fp_rates.append(fp_rate)
                class_fn_rates.append(fn_rate)
            
            x_pos = np.arange(len(class_names))
            
            ax1.bar(x_pos, class_fp_rates, alpha=0.7, color='red', label='False Positive Rate')
            ax1.set_xlabel('Class')
            ax1.set_ylabel('False Positive Rate')
            ax1.set_title('Per-Class False Positive Rates')
            ax1.set_xticks(x_pos)
            if len(class_names) <= 20:
                ax1.set_xticklabels(class_names, rotation=45)
            ax1.grid(True, alpha=0.3)
            
            ax2.bar(x_pos, class_fn_rates, alpha=0.7, color='orange', label='False Negative Rate')
            ax2.set_xlabel('Class')
            ax2.set_ylabel('False Negative Rate')
            ax2.set_title('Per-Class False Negative Rates')
            ax2.set_xticks(x_pos)
            if len(class_names) <= 20:
                ax2.set_xticklabels(class_names, rotation=45)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            figures['per_class_errors'] = fig
            
            if save_plots:
                fig.savefig(self.output_dir / 'per_class_error_rates.png', dpi=300, bbox_inches='tight')
        
        logger.info(f"Created {len(figures)} error analysis visualizations")
        return figures
    
    def generate_comprehensive_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_prob: np.ndarray, features: np.ndarray = None,
                                      class_names: List[str] = None,
                                      sample_indices: np.ndarray = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance analysis
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            features: Feature vectors (optional)
            class_names: List of class names
            sample_indices: Original sample indices (optional)
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Generating comprehensive performance analysis...")
        
        analysis = {}
        
        # Prediction patterns analysis
        analysis['prediction_patterns'] = self.analyze_prediction_patterns(
            y_true, y_pred, y_prob, class_names
        )
        
        # Misclassification analysis
        analysis['misclassifications'] = self.analyze_misclassifications(
            y_true, y_pred, y_prob, class_names, sample_indices
        )
        
        # Feature analysis (if features provided)
        if features is not None:
            analysis['feature_importance'] = self.analyze_feature_importance(
                features, y_true, y_pred
            )
            analysis['class_separability'] = self.analyze_class_separability(
                features, y_true, class_names
            )
        
        # Create visualizations
        analysis['visualizations'] = self.create_error_analysis_visualizations(
            y_true, y_pred, y_prob, class_names, save_plots=True
        )
        
        # Save analysis results
        self.analysis_results = analysis
        self.save_analysis_results()
        
        logger.info("Comprehensive performance analysis completed")
        
        return analysis
    
    def save_analysis_results(self) -> None:
        """Save analysis results to files"""
        logger.info("Saving performance analysis results...")
        
        # Save main analysis (excluding visualizations)
        analysis_to_save = {k: v for k, v in self.analysis_results.items() if k != 'visualizations'}
        
        with open(self.output_dir / 'performance_analysis.json', 'w') as f:
            json.dump(analysis_to_save, f, indent=2)
        
        # Save specific analysis components as CSV files
        if 'prediction_patterns' in self.analysis_results:
            patterns = self.analysis_results['prediction_patterns']
            
            # Threshold analysis
            if 'threshold_analysis' in patterns:
                threshold_df = pd.DataFrame(patterns['threshold_analysis'])
                threshold_df.to_csv(self.output_dir / 'threshold_analysis.csv', index=False)
        
        if 'misclassifications' in self.analysis_results:
            misclass = self.analysis_results['misclassifications']
            
            # Confusion pairs
            if 'confusion_pairs' in misclass:
                confusion_df = pd.DataFrame(misclass['confusion_pairs'])
                confusion_df.to_csv(self.output_dir / 'confusion_pairs.csv', index=False)
        
        logger.info(f"Performance analysis results saved to {self.output_dir}")