"""Application settings and configuration management."""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import json


@dataclass
class DatasetConfig:
    """Dataset configuration settings."""
    
    # Paths
    base_path: Path = Path("data")
    raw_data_path: Path = field(default_factory=lambda: Path("data/raw"))
    processed_data_path: Path = field(default_factory=lambda: Path("data/processed"))
    unified_dataset_path: Path = field(default_factory=lambda: Path("data/processed/UnifiedFishDataset"))
    
    # Image processing
    image_size: tuple = (96, 96)
    image_formats: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.JPG'])
    
    # Data splits
    train_split: float = 0.7
    val_split: float = 0.2
    test_split: float = 0.1
    
    # Dataset parameters
    max_samples_per_class: Optional[int] = None
    min_samples_per_class: int = 10


@dataclass
class TrainingConfig:
    """Training configuration settings."""
    
    # Model training
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 0.001
    
    # Early stopping
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 8
    reduce_lr_factor: float = 0.5
    
    # Callbacks
    save_best_only: bool = True
    monitor_metric: str = 'val_loss'
    monitor_mode: str = 'min'
    
    # Data augmentation
    use_augmentation: bool = True
    rotation_range: int = 25
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    shear_range: float = 0.15
    zoom_range: float = 0.2
    horizontal_flip: bool = True
    vertical_flip: bool = False
    brightness_range: tuple = (0.8, 1.2)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Model parameters
    input_shape: tuple = (96, 96, 3)
    num_classes: int = 35
    dropout_rate: float = 0.5
    
    # Architecture
    architecture: str = "cnn"  # cnn, lightweight_cnn, efficient_net, resnet
    model_type: str = "lightweight_cnn"  # lightweight_cnn, efficient_net, resnet
    use_batch_normalization: bool = True
    use_global_average_pooling: bool = True
    
    # Transfer learning
    use_pretrained: bool = False
    freeze_base_layers: bool = True
    fine_tune_layers: int = 0


@dataclass
class UIConfig:
    """User interface configuration."""
    
    # Streamlit
    page_title: str = "🐟 Fish Species Classifier"
    page_icon: str = "🐟"
    layout: str = "wide"
    
    # Display
    max_upload_size: int = 200 * 1024 * 1024  # 200MB
    allowed_file_types: List[str] = field(default_factory=lambda: ['png', 'jpg', 'jpeg'])
    
    # Prediction display
    top_k_predictions: int = 5
    confidence_threshold: float = 0.01
    show_technical_details: bool = True


@dataclass
class Settings:
    """Main application settings."""
    
    # Environment
    debug: bool = False
    log_level: str = "INFO"
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path.cwd())
    models_path: Path = field(default_factory=lambda: Path("data/models"))
    results_path: Path = field(default_factory=lambda: Path("results"))
    logs_path: Path = field(default_factory=lambda: Path("results/logs"))
    
    # Model files
    class_mapping_file: str = "class_mapping.json"
    default_model_file: str = "demo_fish_classifier.keras"
    
    # Configuration objects
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    def __post_init__(self):
        """Post initialization processing."""
        # Ensure paths are Path objects
        for attr_name in ['project_root', 'models_path', 'results_path', 'logs_path']:
            if hasattr(self, attr_name):
                path_value = getattr(self, attr_name)
                if isinstance(path_value, str):
                    setattr(self, attr_name, Path(path_value))
        
        # Update nested paths relative to project root
        self.models_path = self.project_root / self.models_path
        self.results_path = self.project_root / self.results_path
        self.logs_path = self.project_root / self.logs_path
        
        # Update dataset paths
        self.dataset.base_path = self.project_root / self.dataset.base_path
        self.dataset.raw_data_path = self.project_root / self.dataset.raw_data_path
        self.dataset.processed_data_path = self.project_root / self.dataset.processed_data_path
        self.dataset.unified_dataset_path = self.project_root / self.dataset.unified_dataset_path
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.models_path,
            self.results_path,
            self.logs_path,
            self.dataset.raw_data_path,
            self.dataset.processed_data_path,
            self.results_path / "experiments",
            self.results_path / "evaluations",
            self.results_path / "visualizations"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_class_mapping(self) -> Dict[str, int]:
        """Load class mapping from file."""
        mapping_file = self.models_path / self.class_mapping_file
        
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                return json.load(f)
        else:
            # Return default mapping if file doesn't exist
            return {}
    
    def save_class_mapping(self, class_mapping: Dict[str, int]) -> None:
        """Save class mapping to file."""
        mapping_file = self.models_path / self.class_mapping_file
        mapping_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(mapping_file, 'w') as f:
            json.dump(class_mapping, f, indent=2)
    
    @classmethod
    def from_env(cls) -> 'Settings':
        """Create settings from environment variables."""
        settings = cls()
        
        # Override with environment variables
        if os.getenv('DEBUG'):
            settings.debug = os.getenv('DEBUG', '').lower() in ('true', '1', 'yes')
        
        if os.getenv('LOG_LEVEL'):
            settings.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        if os.getenv('PROJECT_ROOT'):
            settings.project_root = Path(os.getenv('PROJECT_ROOT'))
        
        # Training overrides
        if os.getenv('BATCH_SIZE'):
            settings.training.batch_size = int(os.getenv('BATCH_SIZE'))
        
        if os.getenv('EPOCHS'):
            settings.training.epochs = int(os.getenv('EPOCHS'))
        
        if os.getenv('LEARNING_RATE'):
            settings.training.learning_rate = float(os.getenv('LEARNING_RATE'))
        
        return settings


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
        _settings.create_directories()
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance (mainly for testing)."""
    global _settings
    _settings = None


# Backward compatibility alias
ConfigManager = Settings