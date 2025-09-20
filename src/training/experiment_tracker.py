"""
Experiment Tracking Module
==========================

Enhanced experiment tracking for fish species classification training
with support for metrics logging, visualization, and experiment comparison.
"""

import os
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class ExperimentTracker:
    """Enhanced experiment tracking with comprehensive logging and analysis"""
    
    def __init__(self, experiment_name: Optional[str] = None, 
                 base_dir: str = "experiments"):
        """
        Initialize experiment tracker
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for storing experiments
        """
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / self.experiment_name
        self.experiment_id = str(uuid.uuid4())[:8]
        
        # Create directory structure
        self._create_directories()
        
        # Initialize tracking data
        self.metrics = {}
        self.hyperparameters = {}
        self.artifacts = {}
        self.logs = []
        self.start_time = None
        self.end_time = None
        
        # Files
        self.metrics_file = self.experiment_dir / "metrics.json"
        self.hyperparams_file = self.experiment_dir / "hyperparameters.json"
        self.artifacts_file = self.experiment_dir / "artifacts.json"
        self.logs_file = self.experiment_dir / "logs.txt"
        self.summary_file = self.experiment_dir / "summary.json"
        
        logger.info(f"Initialized experiment tracker: {self.experiment_name}")
    
    def _create_directories(self):
        """Create experiment directory structure"""
        directories = [
            self.experiment_dir,
            self.experiment_dir / "plots",
            self.experiment_dir / "models",
            self.experiment_dir / "checkpoints",
            self.experiment_dir / "predictions",
            self.experiment_dir / "data"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def start_experiment(self, config: Dict[str, Any] = None):
        """
        Start tracking an experiment
        
        Args:
            config: Configuration dictionary to log
        """
        self.start_time = time.time()
        
        # Log initial information
        self.log_info(f"Starting experiment: {self.experiment_name}")
        self.log_info(f"Experiment ID: {self.experiment_id}")
        self.log_info(f"Start time: {datetime.now().isoformat()}")
        
        # Log configuration
        if config:
            self.log_hyperparameters(config)
        
        # Save initial state
        self._save_experiment_state()
    
    def end_experiment(self, status: str = "completed"):
        """
        End experiment tracking
        
        Args:
            status: Final status of the experiment
        """
        self.end_time = time.time()
        duration = self.end_time - self.start_time if self.start_time else 0
        
        self.log_info(f"Ending experiment: {status}")
        self.log_info(f"End time: {datetime.now().isoformat()}")
        self.log_info(f"Duration: {duration:.2f} seconds")
        
        # Add final metrics
        self.log_metric("experiment_duration_seconds", duration)
        self.log_metric("experiment_status", status)
        
        # Save final state
        self._save_experiment_state()
        self._generate_summary()
        
        logger.info(f"Experiment {self.experiment_name} ended with status: {status}")
    
    def log_metric(self, name: str, value: Union[float, int, str], step: Optional[int] = None):
        """
        Log a metric value
        
        Args:
            name: Metric name
            value: Metric value
            step: Step/epoch number (optional)
        """
        timestamp = time.time()
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        metric_entry = {
            "value": value,
            "timestamp": timestamp,
            "step": step
        }
        
        self.metrics[name].append(metric_entry)
        
        # Save metrics periodically
        if len(self.metrics[name]) % 10 == 0:
            self._save_metrics()
    
    def log_metrics(self, metrics_dict: Dict[str, Union[float, int, str]], step: Optional[int] = None):
        """
        Log multiple metrics at once
        
        Args:
            metrics_dict: Dictionary of metric names and values
            step: Step/epoch number (optional)
        """
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step)
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        self.hyperparameters.update(hyperparams)
        self._save_hyperparameters()
        
        self.log_info(f"Logged {len(hyperparams)} hyperparameters")
    
    def log_artifact(self, name: str, path: str, artifact_type: str = "file"):
        """
        Log an artifact (file, model, plot, etc.)
        
        Args:
            name: Artifact name
            path: Path to the artifact
            artifact_type: Type of artifact
        """
        artifact_entry = {
            "path": str(path),
            "type": artifact_type,
            "timestamp": time.time(),
            "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0
        }
        
        self.artifacts[name] = artifact_entry
        self._save_artifacts()
        
        self.log_info(f"Logged artifact: {name} ({artifact_type})")
    
    def log_model(self, model, model_name: str = "model"):
        """
        Log a trained model
        
        Args:
            model: Trained model object
            model_name: Name for the model
        """
        model_path = self.experiment_dir / "models" / f"{model_name}.keras"
        
        try:
            model.save(model_path)
            self.log_artifact(model_name, model_path, "model")
            
            # Log model summary
            if hasattr(model, 'summary'):
                summary_path = self.experiment_dir / "models" / f"{model_name}_summary.txt"
                with open(summary_path, 'w') as f:
                    model.summary(print_fn=lambda x: f.write(x + '\n'))
                self.log_artifact(f"{model_name}_summary", summary_path, "text")
            
            self.log_info(f"Model saved: {model_name}")
            
        except Exception as e:
            self.log_error(f"Failed to save model {model_name}: {e}")
    
    def log_plot(self, figure, plot_name: str, format: str = "png"):
        """
        Log a matplotlib figure
        
        Args:
            figure: Matplotlib figure
            plot_name: Name for the plot
            format: Image format (png, jpg, pdf)
        """
        plot_path = self.experiment_dir / "plots" / f"{plot_name}.{format}"
        
        try:
            figure.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.log_artifact(plot_name, plot_path, "plot")
            
            self.log_info(f"Plot saved: {plot_name}")
            
        except Exception as e:
            self.log_error(f"Failed to save plot {plot_name}: {e}")
        
        finally:
            plt.close(figure)
    
    def log_dataframe(self, df: pd.DataFrame, name: str, format: str = "csv"):
        """
        Log a pandas DataFrame
        
        Args:
            df: DataFrame to log
            name: Name for the data
            format: Format to save (csv, json, parquet)
        """
        data_path = self.experiment_dir / "data" / f"{name}.{format}"
        
        try:
            if format == "csv":
                df.to_csv(data_path, index=False)
            elif format == "json":
                df.to_json(data_path, orient="records", indent=2)
            elif format == "parquet":
                df.to_parquet(data_path, index=False)
            
            self.log_artifact(name, data_path, "data")
            self.log_info(f"DataFrame saved: {name} ({len(df)} rows)")
            
        except Exception as e:
            self.log_error(f"Failed to save DataFrame {name}: {e}")
    
    def log_info(self, message: str):
        """Log an info message"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] INFO: {message}"
        
        self.logs.append(log_entry)
        
        # Write to log file
        with open(self.logs_file, 'a') as f:
            f.write(log_entry + '\n')
        
        logger.info(message)
    
    def log_warning(self, message: str):
        """Log a warning message"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] WARNING: {message}"
        
        self.logs.append(log_entry)
        
        with open(self.logs_file, 'a') as f:
            f.write(log_entry + '\n')
        
        logger.warning(message)
    
    def log_error(self, message: str):
        """Log an error message"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] ERROR: {message}"
        
        self.logs.append(log_entry)
        
        with open(self.logs_file, 'a') as f:
            f.write(log_entry + '\n')
        
        logger.error(message)
    
    def get_metric_history(self, metric_name: str) -> List[Dict[str, Any]]:
        """
        Get history of a specific metric
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of metric entries
        """
        return self.metrics.get(metric_name, [])
    
    def get_latest_metric(self, metric_name: str) -> Optional[Any]:
        """
        Get the latest value of a metric
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Latest metric value or None
        """
        history = self.get_metric_history(metric_name)
        return history[-1]["value"] if history else None
    
    def plot_metric_history(self, metric_names: Union[str, List[str]], 
                           title: str = None, save: bool = True) -> plt.Figure:
        """
        Plot metric history over time
        
        Args:
            metric_names: Single metric name or list of metric names
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        if isinstance(metric_names, str):
            metric_names = [metric_names]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric_name in metric_names:
            history = self.get_metric_history(metric_name)
            if history:
                steps = [entry.get("step", i) for i, entry in enumerate(history)]
                values = [entry["value"] for entry in history]
                ax.plot(steps, values, label=metric_name, marker='o', markersize=4)
        
        ax.set_xlabel('Step/Epoch')
        ax.set_ylabel('Value')
        ax.set_title(title or f'Metric History: {", ".join(metric_names)}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            self.log_plot(fig, f"metrics_{'_'.join(metric_names)}")
        
        return fig
    
    def plot_training_curves(self, save: bool = True) -> plt.Figure:
        """
        Plot standard training curves (loss and accuracy)
        
        Args:
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        train_loss = self.get_metric_history("loss")
        val_loss = self.get_metric_history("val_loss")
        
        if train_loss:
            epochs = [entry.get("step", i) for i, entry in enumerate(train_loss)]
            train_values = [entry["value"] for entry in train_loss]
            ax1.plot(epochs, train_values, label="Training Loss", color="blue")
        
        if val_loss:
            epochs = [entry.get("step", i) for i, entry in enumerate(val_loss)]
            val_values = [entry["value"] for entry in val_loss]
            ax1.plot(epochs, val_values, label="Validation Loss", color="orange")
        
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Training and validation accuracy
        train_acc = self.get_metric_history("accuracy")
        val_acc = self.get_metric_history("val_accuracy")
        
        if train_acc:
            epochs = [entry.get("step", i) for i, entry in enumerate(train_acc)]
            train_values = [entry["value"] for entry in train_acc]
            ax2.plot(epochs, train_values, label="Training Accuracy", color="blue")
        
        if val_acc:
            epochs = [entry.get("step", i) for i, entry in enumerate(val_acc)]
            val_values = [entry["value"] for entry in val_acc]
            ax2.plot(epochs, val_values, label="Validation Accuracy", color="orange")
        
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate
        lr_history = self.get_metric_history("learning_rate")
        if lr_history:
            epochs = [entry.get("step", i) for i, entry in enumerate(lr_history)]
            lr_values = [entry["value"] for entry in lr_history]
            ax3.plot(epochs, lr_values, color="green")
            ax3.set_title("Learning Rate")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Learning Rate")
            ax3.set_yscale("log")
            ax3.grid(True, alpha=0.3)
        
        # Additional metrics
        other_metrics = [name for name in self.metrics.keys() 
                        if name not in ["loss", "val_loss", "accuracy", "val_accuracy", "learning_rate"]]
        
        if other_metrics:
            for i, metric_name in enumerate(other_metrics[:3]):  # Show up to 3 additional metrics
                history = self.get_metric_history(metric_name)
                if history:
                    epochs = [entry.get("step", i) for i, entry in enumerate(history)]
                    values = [entry["value"] for entry in history]
                    ax4.plot(epochs, values, label=metric_name)
            
            ax4.set_title("Additional Metrics")
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Value")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            self.log_plot(fig, "training_curves")
        
        return fig
    
    def generate_experiment_report(self) -> str:
        """
        Generate a comprehensive experiment report
        
        Returns:
            HTML report string
        """
        # Calculate experiment duration
        duration = (self.end_time - self.start_time) if (self.start_time and self.end_time) else 0
        
        # Get key metrics
        final_metrics = {}
        for metric_name in self.metrics:
            latest = self.get_latest_metric(metric_name)
            if latest is not None:
                final_metrics[metric_name] = latest
        
        # Generate report
        report = f"""
        <html>
        <head>
            <title>Experiment Report: {self.experiment_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 8px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9e9e9; border-radius: 4px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Experiment Report</h1>
                <h2>{self.experiment_name}</h2>
                <p><strong>Experiment ID:</strong> {self.experiment_id}</p>
                <p><strong>Duration:</strong> {duration:.2f} seconds</p>
            </div>
            
            <div class="section">
                <h3>Final Metrics</h3>
                {self._format_metrics_html(final_metrics)}
            </div>
            
            <div class="section">
                <h3>Hyperparameters</h3>
                {self._format_hyperparams_html()}
            </div>
            
            <div class="section">
                <h3>Artifacts</h3>
                {self._format_artifacts_html()}
            </div>
        </body>
        </html>
        """
        
        # Save report
        report_path = self.experiment_dir / "report.html"
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.log_artifact("experiment_report", report_path, "report")
        
        return report
    
    def _format_metrics_html(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as HTML"""
        if not metrics:
            return "<p>No metrics recorded</p>"
        
        html = "<div>"
        for name, value in metrics.items():
            html += f'<div class="metric"><strong>{name}:</strong> {value}</div>'
        html += "</div>"
        return html
    
    def _format_hyperparams_html(self) -> str:
        """Format hyperparameters as HTML table"""
        if not self.hyperparameters:
            return "<p>No hyperparameters recorded</p>"
        
        html = "<table><tr><th>Parameter</th><th>Value</th></tr>"
        for name, value in self.hyperparameters.items():
            html += f"<tr><td>{name}</td><td>{value}</td></tr>"
        html += "</table>"
        return html
    
    def _format_artifacts_html(self) -> str:
        """Format artifacts as HTML table"""
        if not self.artifacts:
            return "<p>No artifacts recorded</p>"
        
        html = "<table><tr><th>Name</th><th>Type</th><th>Size</th></tr>"
        for name, info in self.artifacts.items():
            size = info.get("size_bytes", 0)
            size_str = f"{size / 1024:.1f} KB" if size < 1024*1024 else f"{size / (1024*1024):.1f} MB"
            html += f"<tr><td>{name}</td><td>{info['type']}</td><td>{size_str}</td></tr>"
        html += "</table>"
        return html
    
    def _save_metrics(self):
        """Save metrics to file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def _save_hyperparameters(self):
        """Save hyperparameters to file"""
        with open(self.hyperparams_file, 'w') as f:
            json.dump(self.hyperparameters, f, indent=2)
    
    def _save_artifacts(self):
        """Save artifacts metadata to file"""
        with open(self.artifacts_file, 'w') as f:
            json.dump(self.artifacts, f, indent=2)
    
    def _save_experiment_state(self):
        """Save complete experiment state"""
        self._save_metrics()
        self._save_hyperparameters()
        self._save_artifacts()
    
    def _generate_summary(self):
        """Generate experiment summary"""
        summary = {
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": (self.end_time - self.start_time) if (self.start_time and self.end_time) else 0,
            "final_metrics": {name: self.get_latest_metric(name) for name in self.metrics},
            "hyperparameters": self.hyperparameters,
            "artifacts_count": len(self.artifacts),
            "logs_count": len(self.logs)
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

class ExperimentComparison:
    """Compare multiple experiments"""
    
    def __init__(self, experiments_dir: str = "experiments"):
        """
        Initialize experiment comparison
        
        Args:
            experiments_dir: Directory containing experiments
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments = self._load_experiments()
    
    def _load_experiments(self) -> Dict[str, Dict[str, Any]]:
        """Load all experiments from directory"""
        experiments = {}
        
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir():
                summary_file = exp_dir / "summary.json"
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r') as f:
                            experiments[exp_dir.name] = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load experiment {exp_dir.name}: {e}")
        
        return experiments
    
    def compare_metrics(self, metric_name: str) -> pd.DataFrame:
        """
        Compare a specific metric across experiments
        
        Args:
            metric_name: Name of metric to compare
            
        Returns:
            DataFrame with comparison results
        """
        data = []
        
        for exp_name, exp_data in self.experiments.items():
            final_metrics = exp_data.get("final_metrics", {})
            if metric_name in final_metrics:
                data.append({
                    "experiment": exp_name,
                    "metric": metric_name,
                    "value": final_metrics[metric_name],
                    "duration": exp_data.get("duration_seconds", 0)
                })
        
        return pd.DataFrame(data)
    
    def plot_metric_comparison(self, metric_name: str) -> plt.Figure:
        """
        Plot comparison of a metric across experiments
        
        Args:
            metric_name: Name of metric to compare
            
        Returns:
            Matplotlib figure
        """
        df = self.compare_metrics(metric_name)
        
        if df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'No data for metric: {metric_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(df["experiment"], df["value"])
        ax.set_title(f'Comparison: {metric_name}')
        ax.set_xlabel('Experiment')
        ax.set_ylabel(metric_name)
        
        # Add value labels on bars
        for bar, value in zip(bars, df["value"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.4f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def get_best_experiment(self, metric_name: str, higher_is_better: bool = True) -> Optional[str]:
        """
        Get the best experiment based on a metric
        
        Args:
            metric_name: Name of metric to optimize
            higher_is_better: Whether higher values are better
            
        Returns:
            Name of best experiment or None
        """
        df = self.compare_metrics(metric_name)
        
        if df.empty:
            return None
        
        if higher_is_better:
            best_idx = df["value"].idxmax()
        else:
            best_idx = df["value"].idxmin()
        
        return df.loc[best_idx, "experiment"]