"""
Model Comparison Module
======================

Comprehensive model comparison utilities for evaluating and comparing
multiple models on fish species classification tasks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings

from .metrics import MetricsCalculator
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class ModelComparison:
    """Comprehensive model comparison and analysis"""
    
    def __init__(self, output_dir: str = "model_comparison_results"):
        """
        Initialize model comparison
        
        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_calculator = MetricsCalculator()
        self.model_results = {}
        self.comparison_results = {}
    
    def add_model_results(self, model_name: str, y_true: np.ndarray, 
                         y_pred: np.ndarray, y_prob: np.ndarray = None,
                         model_info: Dict[str, Any] = None) -> None:
        """
        Add model results for comparison
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            model_info: Additional model information
        """
        logger.info(f"Adding results for model: {model_name}")
        
        # Calculate comprehensive metrics
        metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            y_true, y_pred, y_prob
        )
        
        self.model_results[model_name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'metrics': metrics,
            'model_info': model_info or {}
        }
        
        logger.info(f"Added results for {model_name} with {len(y_true)} samples")
    
    def compare_basic_metrics(self) -> pd.DataFrame:
        """
        Compare basic metrics across all models
        
        Returns:
            DataFrame with basic metrics comparison
        """
        if not self.model_results:
            logger.warning("No model results to compare")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, results in self.model_results.items():
            basic_metrics = results['metrics'].get('basic', {})
            
            row = {'Model': model_name}
            row.update(basic_metrics)
            
            # Add model info if available
            model_info = results.get('model_info', {})
            if 'parameters' in model_info:
                row['Parameters'] = model_info['parameters']
            if 'training_time' in model_info:
                row['Training_Time'] = model_info['training_time']
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Round numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].round(4)
        
        return df
    
    def compare_per_class_performance(self, class_names: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Compare per-class performance across models
        
        Args:
            class_names: List of class names
            
        Returns:
            Dictionary of DataFrames for each metric
        """
        if not self.model_results:
            return {}
        
        # Get per-class metrics for all models
        all_per_class = {}
        for model_name, results in self.model_results.items():
            per_class = results['metrics'].get('per_class', {})
            all_per_class[model_name] = per_class
        
        if not all_per_class:
            return {}
        
        # Extract class names
        if class_names is None:
            class_names = list(next(iter(all_per_class.values())).keys())
        
        # Create comparison DataFrames for each metric
        metrics_to_compare = ['precision', 'recall', 'f1_score', 'support']
        comparison_dfs = {}
        
        for metric in metrics_to_compare:
            data = []
            for class_name in class_names:
                row = {'Class': class_name}
                for model_name in all_per_class.keys():
                    if class_name in all_per_class[model_name]:
                        value = all_per_class[model_name][class_name].get(metric, np.nan)
                        row[model_name] = value
                    else:
                        row[model_name] = np.nan
                data.append(row)
            
            comparison_dfs[metric] = pd.DataFrame(data)
        
        return comparison_dfs
    
    def statistical_significance_test(self, metric: str = 'accuracy') -> Dict[str, Any]:
        """
        Perform statistical significance tests between models
        
        Args:
            metric: Metric to test ('accuracy', 'f1_macro', etc.)
            
        Returns:
            Dictionary with test results
        """
        if len(self.model_results) < 2:
            logger.warning("Need at least 2 models for significance testing")
            return {}
        
        model_names = list(self.model_results.keys())
        results = {}
        
        # Extract metric values for each model
        metric_values = {}
        for model_name, model_result in self.model_results.items():
            y_true = model_result['y_true']
            y_pred = model_result['y_pred']
            
            if metric == 'accuracy':
                values = (y_true == y_pred).astype(int)
            elif metric == 'f1_macro':
                # Calculate F1 for each sample (approximation)
                values = (y_true == y_pred).astype(int)  # Simplified
            else:
                logger.warning(f"Metric {metric} not implemented for significance testing")
                continue
            
            metric_values[model_name] = values
        
        # Pairwise comparisons
        pairwise_tests = {}
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                values1 = metric_values[model1]
                values2 = metric_values[model2]
                
                # McNemar's test for paired samples
                if len(values1) == len(values2):
                    # Create contingency table
                    correct1 = values1
                    correct2 = values2
                    
                    both_correct = np.sum((correct1 == 1) & (correct2 == 1))
                    only1_correct = np.sum((correct1 == 1) & (correct2 == 0))
                    only2_correct = np.sum((correct1 == 0) & (correct2 == 1))
                    both_wrong = np.sum((correct1 == 0) & (correct2 == 0))
                    
                    # McNemar's test
                    if only1_correct + only2_correct > 0:
                        chi2 = (abs(only1_correct - only2_correct) - 1)**2 / (only1_correct + only2_correct)
                        p_value = 1 - stats.chi2.cdf(chi2, 1)
                    else:
                        chi2 = 0
                        p_value = 1.0
                    
                    pairwise_tests[f"{model1}_vs_{model2}"] = {
                        'chi2_statistic': float(chi2),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'both_correct': int(both_correct),
                        'only_model1_correct': int(only1_correct),
                        'only_model2_correct': int(only2_correct),
                        'both_wrong': int(both_wrong)
                    }
        
        results['pairwise_tests'] = pairwise_tests
        results['metric_tested'] = metric
        
        return results
    
    def create_performance_ranking(self, metrics: List[str] = None) -> pd.DataFrame:
        """
        Create performance ranking of models
        
        Args:
            metrics: List of metrics to consider for ranking
            
        Returns:
            DataFrame with model rankings
        """
        if metrics is None:
            metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        
        ranking_data = []
        
        for model_name, results in self.model_results.items():
            basic_metrics = results['metrics'].get('basic', {})
            
            row = {'Model': model_name}
            score = 0
            valid_metrics = 0
            
            for metric in metrics:
                if metric in basic_metrics:
                    value = basic_metrics[metric]
                    row[metric] = value
                    score += value
                    valid_metrics += 1
                else:
                    row[metric] = np.nan
            
            row['Average_Score'] = score / valid_metrics if valid_metrics > 0 else 0
            ranking_data.append(row)
        
        df = pd.DataFrame(ranking_data)
        df = df.sort_values('Average_Score', ascending=False)
        df['Rank'] = range(1, len(df) + 1)
        
        # Reorder columns
        cols = ['Rank', 'Model'] + metrics + ['Average_Score']
        df = df[cols]
        
        return df
    
    def analyze_model_complexity(self) -> pd.DataFrame:
        """
        Analyze model complexity vs performance trade-offs
        
        Returns:
            DataFrame with complexity analysis
        """
        complexity_data = []
        
        for model_name, results in self.model_results.items():
            model_info = results.get('model_info', {})
            basic_metrics = results['metrics'].get('basic', {})
            
            row = {
                'Model': model_name,
                'Accuracy': basic_metrics.get('accuracy', np.nan),
                'F1_Macro': basic_metrics.get('f1_macro', np.nan),
                'Parameters': model_info.get('parameters', np.nan),
                'Training_Time': model_info.get('training_time', np.nan),
                'Inference_Time': model_info.get('inference_time', np.nan),
                'Model_Size': model_info.get('model_size_mb', np.nan)
            }
            
            # Calculate efficiency metrics
            if row['Parameters'] and row['Accuracy']:
                row['Accuracy_per_Parameter'] = row['Accuracy'] / (row['Parameters'] / 1e6)  # Per million parameters
            
            if row['Training_Time'] and row['Accuracy']:
                row['Accuracy_per_Hour'] = row['Accuracy'] / (row['Training_Time'] / 3600)  # Per hour
            
            complexity_data.append(row)
        
        return pd.DataFrame(complexity_data)
    
    def visualize_model_comparison(self, save_plots: bool = True) -> Dict[str, plt.Figure]:
        """
        Create comprehensive visualization of model comparison
        
        Args:
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary of matplotlib figures
        """
        figures = {}
        
        # 1. Basic metrics comparison
        basic_comparison = self.compare_basic_metrics()
        if not basic_comparison.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            metrics_to_plot = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
            plot_data = basic_comparison[['Model'] + [m for m in metrics_to_plot if m in basic_comparison.columns]]
            
            plot_data.set_index('Model')[metrics_to_plot].plot(kind='bar', ax=ax, width=0.8)
            ax.set_title('Model Performance Comparison - Basic Metrics')
            ax.set_ylabel('Score')
            ax.legend(title='Metrics')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            figures['basic_metrics'] = fig
            
            if save_plots:
                fig.savefig(self.output_dir / 'model_comparison_basic_metrics.png', dpi=300, bbox_inches='tight')
        
        # 2. Performance ranking
        ranking = self.create_performance_ranking()
        if not ranking.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.barh(ranking['Model'], ranking['Average_Score'])
            ax.set_xlabel('Average Score')
            ax.set_title('Model Performance Ranking')
            ax.grid(True, alpha=0.3)
            
            # Add rank annotations
            for i, (idx, row) in enumerate(ranking.iterrows()):
                ax.text(row['Average_Score'] + 0.01, i, f"#{row['Rank']}", 
                       va='center', fontweight='bold')
            
            plt.tight_layout()
            figures['ranking'] = fig
            
            if save_plots:
                fig.savefig(self.output_dir / 'model_ranking.png', dpi=300, bbox_inches='tight')
        
        # 3. Confusion matrix comparison (if 2-3 models)
        if 2 <= len(self.model_results) <= 3:
            n_models = len(self.model_results)
            fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
            if n_models == 1:
                axes = [axes]
            
            for idx, (model_name, results) in enumerate(self.model_results.items()):
                cm = results['metrics']['confusion_matrix']['normalized_confusion_matrix']
                
                im = axes[idx].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                axes[idx].set_title(f'{model_name}\nNormalized Confusion Matrix')
                
                plt.colorbar(im, ax=axes[idx])
            
            plt.tight_layout()
            figures['confusion_matrices'] = fig
            
            if save_plots:
                fig.savefig(self.output_dir / 'confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        
        # 4. Model complexity analysis
        complexity_df = self.analyze_model_complexity()
        if not complexity_df.empty and 'Parameters' in complexity_df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Accuracy vs Parameters
            valid_data = complexity_df.dropna(subset=['Parameters', 'Accuracy'])
            if not valid_data.empty:
                ax1.scatter(valid_data['Parameters'], valid_data['Accuracy'])
                for idx, row in valid_data.iterrows():
                    ax1.annotate(row['Model'], (row['Parameters'], row['Accuracy']), 
                               xytext=(5, 5), textcoords='offset points')
                ax1.set_xlabel('Number of Parameters')
                ax1.set_ylabel('Accuracy')
                ax1.set_title('Model Complexity vs Accuracy')
                ax1.grid(True, alpha=0.3)
            
            # Training Time vs Accuracy
            valid_data2 = complexity_df.dropna(subset=['Training_Time', 'Accuracy'])
            if not valid_data2.empty:
                ax2.scatter(valid_data2['Training_Time'], valid_data2['Accuracy'])
                for idx, row in valid_data2.iterrows():
                    ax2.annotate(row['Model'], (row['Training_Time'], row['Accuracy']), 
                               xytext=(5, 5), textcoords='offset points')
                ax2.set_xlabel('Training Time (seconds)')
                ax2.set_ylabel('Accuracy')
                ax2.set_title('Training Time vs Accuracy')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            figures['complexity_analysis'] = fig
            
            if save_plots:
                fig.savefig(self.output_dir / 'complexity_analysis.png', dpi=300, bbox_inches='tight')
        
        logger.info(f"Created {len(figures)} comparison visualizations")
        return figures
    
    def generate_comparison_report(self) -> str:
        """
        Generate comprehensive comparison report
        
        Returns:
            HTML report string
        """
        report_sections = []
        
        # Title
        report_sections.append("<h1>Model Comparison Report</h1>")
        report_sections.append(f"<p>Comparison of {len(self.model_results)} models on fish species classification task.</p>")
        
        # Basic metrics comparison
        basic_df = self.compare_basic_metrics()
        if not basic_df.empty:
            report_sections.append("<h2>Basic Metrics Comparison</h2>")
            report_sections.append(basic_df.to_html(index=False, table_id="basic_metrics"))
        
        # Performance ranking
        ranking_df = self.create_performance_ranking()
        if not ranking_df.empty:
            report_sections.append("<h2>Performance Ranking</h2>")
            report_sections.append(ranking_df.to_html(index=False, table_id="ranking"))
        
        # Statistical significance
        sig_results = self.statistical_significance_test()
        if sig_results:
            report_sections.append("<h2>Statistical Significance Tests</h2>")
            report_sections.append("<p>McNemar's test results for pairwise model comparisons:</p>")
            
            for comparison, test_result in sig_results.get('pairwise_tests', {}).items():
                significance = "Significant" if test_result['significant'] else "Not significant"
                report_sections.append(
                    f"<p><strong>{comparison}</strong>: p-value = {test_result['p_value']:.4f} ({significance})</p>"
                )
        
        # Model complexity analysis
        complexity_df = self.analyze_model_complexity()
        if not complexity_df.empty:
            report_sections.append("<h2>Model Complexity Analysis</h2>")
            report_sections.append(complexity_df.to_html(index=False, table_id="complexity"))
        
        # Best model summary
        if not ranking_df.empty:
            best_model = ranking_df.iloc[0]['Model']
            best_accuracy = ranking_df.iloc[0].get('accuracy', 'N/A')
            report_sections.append("<h2>Summary</h2>")
            report_sections.append(f"<p><strong>Best performing model:</strong> {best_model}</p>")
            report_sections.append(f"<p><strong>Best accuracy:</strong> {best_accuracy}</p>")
        
        # Combine all sections
        html_report = "\n".join(report_sections)
        
        # Save report
        report_path = self.output_dir / 'comparison_report.html'
        with open(report_path, 'w') as f:
            f.write(html_report)
        
        logger.info(f"Comparison report saved to {report_path}")
        
        return html_report
    
    def save_comparison_results(self) -> None:
        """Save all comparison results to files"""
        logger.info("Saving comparison results...")
        
        # Save basic metrics comparison
        basic_df = self.compare_basic_metrics()
        if not basic_df.empty:
            basic_df.to_csv(self.output_dir / 'basic_metrics_comparison.csv', index=False)
        
        # Save performance ranking
        ranking_df = self.create_performance_ranking()
        if not ranking_df.empty:
            ranking_df.to_csv(self.output_dir / 'performance_ranking.csv', index=False)
        
        # Save per-class comparisons
        per_class_dfs = self.compare_per_class_performance()
        for metric, df in per_class_dfs.items():
            df.to_csv(self.output_dir / f'per_class_{metric}_comparison.csv', index=False)
        
        # Save complexity analysis
        complexity_df = self.analyze_model_complexity()
        if not complexity_df.empty:
            complexity_df.to_csv(self.output_dir / 'complexity_analysis.csv', index=False)
        
        # Save statistical significance results
        sig_results = self.statistical_significance_test()
        if sig_results:
            with open(self.output_dir / 'significance_tests.json', 'w') as f:
                json.dump(sig_results, f, indent=2)
        
        # Generate and save visualizations
        self.visualize_model_comparison(save_plots=True)
        
        # Generate HTML report
        self.generate_comparison_report()
        
        logger.info(f"All comparison results saved to {self.output_dir}")
    
    def get_best_model(self, metric: str = 'accuracy') -> Tuple[str, float]:
        """
        Get the best performing model based on specified metric
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, metric_value)
        """
        best_model = None
        best_score = -1
        
        for model_name, results in self.model_results.items():
            basic_metrics = results['metrics'].get('basic', {})
            if metric in basic_metrics:
                score = basic_metrics[metric]
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        return best_model, best_score if best_model else (None, None)