"""
Training Configuration Module
============================

Enhanced configuration management for training fish species classification models
with support for different architectures, hyperparameter optimization, and experiment tracking.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from enum import Enum

class ModelArchitecture(Enum):
    """Supported model architectures"""
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B1 = "efficientnet_b1"
    EFFICIENTNET_B2 = "efficientnet_b2"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    MOBILENET_V2 = "mobilenet_v2"
    VGG16 = "vgg16"
    CUSTOM_CNN = "custom_cnn"
    INCEPTION_V3 = "inception_v3"

class OptimizerType(Enum):
    """Supported optimizers"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"

class SchedulerType(Enum):
    """Supported learning rate schedulers"""
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    COSINE_ANNEALING = "cosine_annealing"
    EXPONENTIAL = "exponential"
    STEP = "step"
    NONE = "none"

@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    type: OptimizerType = OptimizerType.ADAM
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    momentum: float = 0.9  # For SGD
    beta1: float = 0.9     # For Adam/AdamW
    beta2: float = 0.999   # For Adam/AdamW
    epsilon: float = 1e-7
    clipnorm: Optional[float] = None
    clipvalue: Optional[float] = None

@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    type: SchedulerType = SchedulerType.REDUCE_ON_PLATEAU
    factor: float = 0.5
    patience: int = 5
    min_lr: float = 1e-7
    cooldown: int = 0
    verbose: bool = True

@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_range: float = 15.0
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    zoom_range: float = 0.1
    shear_range: float = 0.1
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    hue_range: float = 0.1
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    cutout_holes: int = 0
    cutout_length: int = 16

@dataclass
class CallbackConfig:
    """Training callbacks configuration"""
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_restore_best: bool = True
    
    model_checkpoint: bool = True
    save_best_only: bool = True
    save_weights_only: bool = False
    monitor_metric: str = "val_accuracy"
    
    reduce_lr_on_plateau: bool = True
    lr_patience: int = 5
    lr_factor: float = 0.5
    lr_min_delta: float = 0.0001
    
    tensorboard: bool = True
    csv_logger: bool = True
    
    custom_callbacks: List[str] = field(default_factory=list)

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    architecture: ModelArchitecture = ModelArchitecture.EFFICIENTNET_B0
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    num_classes: int = 35
    dropout_rate: float = 0.3
    use_batch_normalization: bool = True
    activation: str = "relu"
    final_activation: str = "softmax"
    
    # Transfer learning settings
    use_pretrained: bool = True
    freeze_base: bool = True
    unfreeze_from_layer: Optional[int] = None
    fine_tune_epochs: int = 10
    
    # Custom CNN settings (if using CUSTOM_CNN)
    custom_layers: List[Dict[str, Any]] = field(default_factory=list)
    
    # Model regularization
    l1_regularization: float = 0.0
    l2_regularization: float = 0.001

@dataclass
class TrainingConfig:
    """Comprehensive training configuration"""
    
    # Basic training parameters
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42
    
    # Data configuration
    dataset_path: str = "FishImgDataset"
    image_size: Tuple[int, int] = (224, 224)
    normalize_pixels: bool = True
    shuffle_data: bool = True
    cache_dataset: bool = False
    prefetch_buffer: int = 2
    
    # Model configuration
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Optimizer configuration
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    
    # Scheduler configuration
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Augmentation configuration
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    
    # Callback configuration
    callbacks: CallbackConfig = field(default_factory=CallbackConfig)
    
    # Paths and output
    output_dir: str = "training_outputs"
    model_save_path: str = "models"
    logs_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    
    # Training strategy
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    use_class_weights: bool = True
    label_smoothing: float = 0.0
    
    # Validation and testing
    validation_frequency: int = 1  # Validate every N epochs
    save_predictions: bool = True
    generate_confusion_matrix: bool = True
    calculate_class_metrics: bool = True
    
    # Distributed training
    use_distributed: bool = False
    num_gpus: int = 1
    
    # Experiment tracking
    experiment_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Advanced options
    resume_from_checkpoint: Optional[str] = None
    transfer_learning_source: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Validate paths
        self.dataset_path = Path(self.dataset_path)
        self.output_dir = Path(self.output_dir)
        self.model_save_path = Path(self.model_save_path)
        self.logs_dir = Path(self.logs_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        
        # Validate image size consistency
        if self.model.input_shape[:2] != self.image_size:
            self.model.input_shape = (*self.image_size, self.model.input_shape[2])
        
        # Generate experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = f"{self.model.architecture.value}_{self.epochs}ep_{self.batch_size}bs"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        def convert_value(value):
            if hasattr(value, '__dict__'):
                return {k: convert_value(v) for k, v in value.__dict__.items()}
            elif isinstance(value, Enum):
                return value.value
            elif isinstance(value, Path):
                return str(value)
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            else:
                return value
        
        return convert_value(self)
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'TrainingConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Convert enum strings back to enums
        if 'model' in config_dict and 'architecture' in config_dict['model']:
            config_dict['model']['architecture'] = ModelArchitecture(config_dict['model']['architecture'])
        
        if 'optimizer' in config_dict and 'type' in config_dict['optimizer']:
            config_dict['optimizer']['type'] = OptimizerType(config_dict['optimizer']['type'])
        
        if 'scheduler' in config_dict and 'type' in config_dict['scheduler']:
            config_dict['scheduler']['type'] = SchedulerType(config_dict['scheduler']['type'])
        
        # Reconstruct nested dataclasses
        if 'model' in config_dict:
            config_dict['model'] = ModelConfig(**config_dict['model'])
        
        if 'optimizer' in config_dict:
            config_dict['optimizer'] = OptimizerConfig(**config_dict['optimizer'])
        
        if 'scheduler' in config_dict:
            config_dict['scheduler'] = SchedulerConfig(**config_dict['scheduler'])
        
        if 'augmentation' in config_dict:
            config_dict['augmentation'] = AugmentationConfig(**config_dict['augmentation'])
        
        if 'callbacks' in config_dict:
            config_dict['callbacks'] = CallbackConfig(**config_dict['callbacks'])
        
        return cls(**config_dict)
    
    def get_model_name(self) -> str:
        """Generate model name based on configuration"""
        return f"{self.model.architecture.value}_{self.model.num_classes}classes_{self.epochs}ep"
    
    def get_experiment_dir(self) -> Path:
        """Get full experiment directory path"""
        return self.output_dir / self.experiment_name
    
    def create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.output_dir,
            self.get_experiment_dir(),
            self.get_experiment_dir() / "models",
            self.get_experiment_dir() / "logs",
            self.get_experiment_dir() / "checkpoints",
            self.get_experiment_dir() / "plots",
            self.get_experiment_dir() / "predictions"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Basic validation
        if self.epochs <= 0:
            issues.append("epochs must be positive")
        
        if self.batch_size <= 0:
            issues.append("batch_size must be positive")
        
        if not (0 < self.validation_split < 1):
            issues.append("validation_split must be between 0 and 1")
        
        if not (0 <= self.test_split < 1):
            issues.append("test_split must be between 0 and 1")
        
        if self.validation_split + self.test_split >= 1:
            issues.append("validation_split + test_split must be less than 1")
        
        # Model validation
        if self.model.num_classes <= 0:
            issues.append("num_classes must be positive")
        
        if not (0 <= self.model.dropout_rate <= 1):
            issues.append("dropout_rate must be between 0 and 1")
        
        # Optimizer validation
        if self.optimizer.learning_rate <= 0:
            issues.append("learning_rate must be positive")
        
        # Path validation
        if not self.dataset_path.exists():
            issues.append(f"dataset_path {self.dataset_path} does not exist")
        
        return issues
    
    def get_summary(self) -> str:
        """Get a summary of the training configuration"""
        return f"""
Training Configuration Summary:
===============================

Model: {self.model.architecture.value}
Classes: {self.model.num_classes}
Input Shape: {self.model.input_shape}
Epochs: {self.epochs}
Batch Size: {self.batch_size}

Optimizer: {self.optimizer.type.value}
Learning Rate: {self.optimizer.learning_rate}
Scheduler: {self.scheduler.type.value}

Dataset: {self.dataset_path}
Augmentation: {'Enabled' if any([
    self.augmentation.horizontal_flip,
    self.augmentation.vertical_flip,
    self.augmentation.rotation_range > 0
]) else 'Disabled'}

Experiment: {self.experiment_name}
Output Dir: {self.output_dir}
        """.strip()

# Predefined configurations for common use cases
def get_quick_training_config() -> TrainingConfig:
    """Get configuration for quick training/testing"""
    config = TrainingConfig()
    config.epochs = 5
    config.batch_size = 16
    config.model.architecture = ModelArchitecture.EFFICIENTNET_B0
    config.callbacks.early_stopping_patience = 3
    return config

def get_full_training_config() -> TrainingConfig:
    """Get configuration for full training"""
    config = TrainingConfig()
    config.epochs = 100
    config.batch_size = 32
    config.model.architecture = ModelArchitecture.EFFICIENTNET_B1
    config.callbacks.early_stopping_patience = 15
    config.mixed_precision = True
    return config

def get_lightweight_config() -> TrainingConfig:
    """Get configuration for lightweight model"""
    config = TrainingConfig()
    config.epochs = 30
    config.batch_size = 64
    config.model.architecture = ModelArchitecture.MOBILENET_V2
    config.model.input_shape = (128, 128, 3)
    config.image_size = (128, 128)
    return config