"""Visualization utilities for metrics and plots."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings

from src.utils.logging_utils import get_logger
from src.utils.file_utils import ensure_directory

warnings.filterwarnings('ignore')


class PlotManager:
    """Manages plot creation and saving."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize plot manager.
        
        Args:
            output_dir: Directory for saving plots
        """
        self.output_dir = ensure_directory(output_dir)
        self.logger = get_logger(self.__class__.__name__)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = 300) -> Path:
        """
        Save plot to file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: Resolution
            
        Returns:
            Path to saved file
        """
        if not filename.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
            filename += '.png'
        
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        self.logger.info(f"Plot saved to {output_path}")
        
        return output_path
    
    def create_subplot_grid(self, rows: int, cols: int, figsize: Tuple[int, int] = None) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create subplot grid.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            figsize: Figure size
            
        Returns:
            Tuple of (figure, axes)
        """
        if figsize is None:
            figsize = (cols * 5, rows * 4)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Ensure axes is always 2D array
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        return fig, axes


class MetricsVisualizer:
    """Visualizes training metrics and model performance."""
    
    def __init__(self, plot_manager: PlotManager):
        """
        Initialize metrics visualizer.
        
        Args:
            plot_manager: PlotManager instance
        """
        self.plot_manager = plot_manager
        self.logger = get_logger(self.__class__.__name__)
    
    def plot_training_history(
        self, 
        history: Dict[str, List[float]], 
        title: str = "Training History",
        filename: str = "training_history.png"
    ) -> Path:
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            title: Plot title
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        # Determine metrics to plot
        metrics = [key for key in history.keys() if not key.startswith('val_')]
        
        fig, axes = self.plot_manager.create_subplot_grid(
            rows=len(metrics), 
            cols=1, 
            figsize=(10, 4 * len(metrics))
        )
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            ax = axes[i, 0] if len(metrics) > 1 else axes[0, 0]
            
            # Plot training metric
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], 'b-', label=f'Training {metric}', linewidth=2)
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in history:
                ax.plot(epochs, history[val_metric], 'r-', label=f'Validation {metric}', linewidth=2)
            
            ax.set_title(f'{metric.capitalize()} over Epochs')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.plot_manager.save_plot(fig, filename)
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        filename: str = "confusion_matrix.png",
        normalize: bool = False
    ) -> Path:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            title: Plot title
            filename: Output filename
            normalize: Whether to normalize values
            
        Returns:
            Path to saved plot
        """
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            cm = confusion_matrix
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        return self.plot_manager.save_plot(fig, filename)
    
    def plot_class_distribution(
        self,
        class_counts: Dict[str, int],
        title: str = "Class Distribution",
        filename: str = "class_distribution.png"
    ) -> Path:
        """
        Plot class distribution.
        
        Args:
            class_counts: Dictionary of class counts
            title: Plot title
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Create bar plot
        bars = ax.bar(classes, counts, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{int(height)}', ha='center', va='bottom')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Fish Species', fontsize=12)
        ax.set_ylabel('Number of Images', fontsize=12)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add grid
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return self.plot_manager.save_plot(fig, filename)
    
    def plot_prediction_confidence(
        self,
        predictions: List[Dict[str, Any]],
        title: str = "Prediction Confidence Distribution",
        filename: str = "prediction_confidence.png"
    ) -> Path:
        """
        Plot prediction confidence distribution.
        
        Args:
            predictions: List of prediction dictionaries
            title: Plot title
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        confidences = [pred.get('confidence', 0) for pred in predictions]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        ax1.set_title('Confidence Score Distribution')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(confidences)
        ax2.set_title('Confidence Score Box Plot')
        ax2.set_ylabel('Confidence Score')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return self.plot_manager.save_plot(fig, filename)
    
    def plot_prediction_results(
        self,
        results: List[Dict[str, Any]],
        top_k: int = 5,
        title: str = "Top Predictions",
        filename: str = "prediction_results.png"
    ) -> Path:
        """
        Plot prediction results.
        
        Args:
            results: List of prediction results
            top_k: Number of top predictions to show
            title: Plot title
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        # Take only top k results
        results = results[:top_k]
        
        species = [r.get('species', 'Unknown') for r in results]
        confidences = [r.get('confidence', 0) for r in results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create horizontal bar plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(species)))
        bars = ax.barh(range(len(species)), confidences, color=colors)
        
        # Customize plot
        ax.set_yticks(range(len(species)))
        ax.set_yticklabels(species)
        ax.set_xlabel('Confidence Score')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.invert_yaxis()  # Top prediction at top
        
        # Add value labels
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{conf:.3f}', va='center', fontweight='bold')
        
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 1.0)
        
        plt.tight_layout()
        return self.plot_manager.save_plot(fig, filename)
    
    def plot_model_comparison(
        self,
        model_results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        title: str = "Model Comparison",
        filename: str = "model_comparison.png"
    ) -> Path:
        """
        Plot comparison of multiple models.
        
        Args:
            model_results: Dictionary of model results
            metrics: List of metrics to compare
            title: Plot title
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        if metrics is None:
            # Get all available metrics
            all_metrics = set()
            for results in model_results.values():
                all_metrics.update(results.keys())
            metrics = list(all_metrics)
        
        # Create DataFrame for easier plotting
        data = []
        for model_name, results in model_results.items():
            for metric in metrics:
                if metric in results:
                    data.append({
                        'Model': model_name,
                        'Metric': metric,
                        'Value': results[metric]
                    })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.barplot(data=df, x='Metric', y='Value', hue='Model', ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('Score')
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return self.plot_manager.save_plot(fig, filename)


def save_plot(fig: plt.Figure, filename: str, output_dir: str = "plots") -> str:
    """Save plot to file (standalone function)."""
    ensure_directory(output_dir)
    filepath = Path(output_dir) / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    return str(filepath)


def create_prediction_chart(predictions: List[Dict], title: str = "Predictions") -> plt.Figure:
    """Create a prediction chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    classes = [pred.get('class', 'Unknown') for pred in predictions]
    confidences = [pred.get('confidence', 0) for pred in predictions]
    
    # Create bar chart
    ax.bar(classes, confidences)
    ax.set_title(title)
    ax.set_ylabel('Confidence')
    ax.set_xlabel('Classes')
    
    # Rotate labels if needed
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    
    return fig


def create_confidence_plot(predictions: Union[Dict, List], title: str = "Prediction Confidence") -> plt.Figure:
    """
    Create a confidence plot for model predictions.
    
    Args:
        predictions: Dictionary or list of predictions with confidence scores
        title: Title for the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Handle different input formats
    if isinstance(predictions, dict):
        # Dictionary format: {class_name: confidence_score}
        classes = list(predictions.keys())
        confidences = list(predictions.values())
    elif isinstance(predictions, list):
        # List of dictionaries format
        if predictions and isinstance(predictions[0], dict):
            classes = [pred.get('class', pred.get('label', f'Class_{i}')) for i, pred in enumerate(predictions)]
            confidences = [pred.get('confidence', pred.get('score', 0)) for pred in predictions]
        else:
            # Simple list of values
            classes = [f'Class_{i}' for i in range(len(predictions))]
            confidences = predictions
    else:
        raise ValueError("Predictions must be a dictionary or list")
    
    # Ensure we have valid data
    if not classes or not confidences:
        classes = ['No Predictions']
        confidences = [0]
    
    # Sort by confidence (descending)
    sorted_data = sorted(zip(classes, confidences), key=lambda x: x[1], reverse=True)
    classes, confidences = zip(*sorted_data) if sorted_data else (classes, confidences)
    
    # Create horizontal bar chart for better readability
    y_pos = np.arange(len(classes))
    
    # Color bars based on confidence level
    colors = ['#2E8B57' if conf > 0.7 else '#FFD700' if conf > 0.4 else '#FF6347' for conf in confidences]
    
    bars = ax.barh(y_pos, confidences, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()  # Highest confidence at top
    ax.set_xlabel('Confidence Score')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    
    # Add value labels on bars
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{conf:.3f}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3)
    
    # Add confidence level legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, color='#2E8B57', alpha=0.8, label='High Confidence (>0.7)'),
        plt.Rectangle((0,0),1,1, color='#FFD700', alpha=0.8, label='Medium Confidence (0.4-0.7)'),
        plt.Rectangle((0,0),1,1, color='#FF6347', alpha=0.8, label='Low Confidence (<0.4)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    return fig


def create_heatmap_visualization(data: np.ndarray, 
                                labels: Optional[List[str]] = None,
                                title: str = "Heatmap") -> plt.Figure:
    """Create a heatmap visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(data, 
                annot=True, 
                fmt='.2f', 
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax)
    
    ax.set_title(title)
    plt.tight_layout()
    
    return fig