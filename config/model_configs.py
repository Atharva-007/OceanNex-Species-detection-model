"""Model architecture configurations."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


@dataclass
class ModelConfig:
    """Configuration for model training and architecture."""
    
    architecture: str = "cnn"
    num_classes: int = 35
    input_shape: tuple = (224, 224, 3)
    learning_rate: float = 0.001
    optimizer: str = "adam"
    batch_size: int = 16
    epochs: int = 10
    early_stopping_patience: int = 5
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass
class ModelArchitectureConfig:
    """Configuration for model architecture."""
    
    name: str
    description: str
    input_shape: tuple = (96, 96, 3)
    num_classes: int = 35
    parameters: Dict[str, Any] = field(default_factory=dict)


class ModelFactory:
    """Factory class for creating different model architectures."""
    
    @staticmethod
    def create_lightweight_cnn(input_shape: tuple, num_classes: int, **kwargs) -> tf.keras.Model:
        """Create lightweight CNN for quick training and inference."""
        
        dropout_rate = kwargs.get('dropout_rate', 0.5)
        use_batch_norm = kwargs.get('use_batch_normalization', True)
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # First block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization() if use_batch_norm else layers.Lambda(lambda x: x),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Second block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization() if use_batch_norm else layers.Lambda(lambda x: x),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Third block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization() if use_batch_norm else layers.Lambda(lambda x: x),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Classifier
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation='softmax')
        ], name='lightweight_cnn')
        
        return model
    
    @staticmethod
    def create_advanced_cnn(input_shape: tuple, num_classes: int, **kwargs) -> tf.keras.Model:
        """Create advanced CNN with more layers and features."""
        
        dropout_rate = kwargs.get('dropout_rate', 0.5)
        use_batch_norm = kwargs.get('use_batch_normalization', True)
        use_global_pooling = kwargs.get('use_global_average_pooling', True)
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization() if use_batch_norm else layers.Lambda(lambda x: x),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization() if use_batch_norm else layers.Lambda(lambda x: x),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization() if use_batch_norm else layers.Lambda(lambda x: x),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization() if use_batch_norm else layers.Lambda(lambda x: x),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Classifier
            layers.GlobalAveragePooling2D() if use_global_pooling else layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization() if use_batch_norm else layers.Lambda(lambda x: x),
            layers.Dropout(dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate * 0.6),
            layers.Dense(num_classes, activation='softmax')
        ], name='advanced_cnn')
        
        return model
    
    @staticmethod
    def create_transfer_learning_model(input_shape: tuple, num_classes: int, **kwargs) -> tf.keras.Model:
        """Create transfer learning model using pretrained backbone."""
        
        base_model_name = kwargs.get('base_model', 'EfficientNetB0')
        freeze_base = kwargs.get('freeze_base_layers', True)
        fine_tune_layers = kwargs.get('fine_tune_layers', 0)
        dropout_rate = kwargs.get('dropout_rate', 0.5)
        
        # Create base model
        if base_model_name == 'EfficientNetB0':
            from tensorflow.keras.applications import EfficientNetB0
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif base_model_name == 'ResNet50V2':
            from tensorflow.keras.applications import ResNet50V2
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif base_model_name == 'MobileNetV3Large':
            from tensorflow.keras.applications import MobileNetV3Large
            base_model = MobileNetV3Large(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        # Freeze base model layers
        if freeze_base:
            base_model.trainable = False
        else:
            # Fine-tune top layers
            if fine_tune_layers > 0:
                for layer in base_model.layers[:-fine_tune_layers]:
                    layer.trainable = False
        
        # Add custom classifier
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate * 0.6),
            layers.Dense(256, activation='relu'),
            layers.Dropout(dropout_rate * 0.3),
            layers.Dense(num_classes, activation='softmax')
        ], name=f'transfer_learning_{base_model_name.lower()}')
        
        return model


# Model configurations registry
MODEL_CONFIGS = {
    'lightweight_cnn': ModelArchitectureConfig(
        name='lightweight_cnn',
        description='Fast and efficient CNN for quick training and inference',
        parameters={
            'dropout_rate': 0.5,
            'use_batch_normalization': True,
            'use_global_average_pooling': False
        }
    ),
    
    'advanced_cnn': ModelArchitectureConfig(
        name='advanced_cnn',
        description='Advanced CNN with deeper architecture and more features',
        parameters={
            'dropout_rate': 0.5,
            'use_batch_normalization': True,
            'use_global_average_pooling': True
        }
    ),
    
    'efficientnet_b0': ModelArchitectureConfig(
        name='transfer_learning',
        description='EfficientNetB0 with transfer learning',
        parameters={
            'base_model': 'EfficientNetB0',
            'freeze_base_layers': True,
            'fine_tune_layers': 0,
            'dropout_rate': 0.5
        }
    ),
    
    'resnet50v2': ModelArchitectureConfig(
        name='transfer_learning',
        description='ResNet50V2 with transfer learning',
        parameters={
            'base_model': 'ResNet50V2',
            'freeze_base_layers': True,
            'fine_tune_layers': 0,
            'dropout_rate': 0.5
        }
    ),
    
    'mobilenetv3': ModelArchitectureConfig(
        name='transfer_learning',
        description='MobileNetV3Large with transfer learning',
        parameters={
            'base_model': 'MobileNetV3Large',
            'freeze_base_layers': True,
            'fine_tune_layers': 0,
            'dropout_rate': 0.5
        }
    )
}


class ModelConfigManager:
    """Manager for model configurations."""
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available model configurations."""
        return list(MODEL_CONFIGS.keys())
    
    @staticmethod
    def get_model_config(model_name: str) -> ModelArchitectureConfig:
        """Get configuration for a specific model."""
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}")
        return MODEL_CONFIGS[model_name]
    
    @staticmethod
    def create_model(model_name: str, input_shape: tuple, num_classes: int, **kwargs) -> tf.keras.Model:
        """Create model instance based on configuration."""
        config = ModelConfigManager.get_model_config(model_name)
        
        # Merge config parameters with kwargs
        parameters = {**config.parameters, **kwargs}
        
        # Create model based on architecture type
        if config.name == 'lightweight_cnn':
            return ModelFactory.create_lightweight_cnn(input_shape, num_classes, **parameters)
        elif config.name == 'advanced_cnn':
            return ModelFactory.create_advanced_cnn(input_shape, num_classes, **parameters)
        elif config.name == 'transfer_learning':
            return ModelFactory.create_transfer_learning_model(input_shape, num_classes, **parameters)
        else:
            raise ValueError(f"Unknown architecture type: {config.name}")
    
    @staticmethod
    def get_recommended_optimizer(model_name: str, learning_rate: float = 0.001) -> tf.keras.optimizers.Optimizer:
        """Get recommended optimizer for model."""
        config = ModelConfigManager.get_model_config(model_name)
        
        if config.name == 'transfer_learning':
            # Lower learning rate for transfer learning
            return optimizers.Adam(learning_rate=learning_rate * 0.1)
        else:
            return optimizers.Adam(learning_rate=learning_rate)


def get_model_config(model_name: str) -> ModelArchitectureConfig:
    """Get model configuration by name."""
    return ModelConfigManager.get_model_config(model_name)