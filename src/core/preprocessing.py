"""Image preprocessing and data augmentation utilities."""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any, List
import warnings

try:
    from PIL import Image, ImageEnhance, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from ..utils.logger import get_logger
from ..utils.exceptions import PreprocessingError, AugmentationError


logger = get_logger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing operations."""
    
    def __init__(self, target_size: Tuple[int, int] = (96, 96)):
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target image size (width, height)
        """
        self.target_size = target_size
        self.logger = get_logger(self.__class__.__name__)
        
        if not PIL_AVAILABLE:
            raise ImportError("PIL (Pillow) is required but not installed")
    
    def load_image(self, image_path: Union[str, Path]) -> Optional[Image.Image]:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image or None if failed
            
        Raises:
            PreprocessingError: If image loading fails
        """
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                raise PreprocessingError(f"Image file not found: {image_path}")
            
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            self.logger.debug(f"Loaded image: {image_path} - Size: {image.size}, Mode: {image.mode}")
            return image
            
        except Exception as e:
            raise PreprocessingError(f"Failed to load image {image_path}: {str(e)}") from e
    
    def resize_image(
        self, 
        image: Image.Image, 
        size: Optional[Tuple[int, int]] = None,
        maintain_aspect_ratio: bool = False
    ) -> Image.Image:
        """
        Resize image to specified size.
        
        Args:
            image: PIL Image
            size: Target size (width, height), defaults to self.target_size
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized PIL Image
        """
        if size is None:
            size = self.target_size
        
        try:
            if maintain_aspect_ratio:
                # Calculate new size maintaining aspect ratio
                original_width, original_height = image.size
                target_width, target_height = size
                
                ratio = min(target_width / original_width, target_height / original_height)
                new_width = int(original_width * ratio)
                new_height = int(original_height * ratio)
                
                # Resize and pad if needed
                resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Create new image with target size and paste resized image
                new_image = Image.new('RGB', size, (0, 0, 0))
                x_offset = (target_width - new_width) // 2
                y_offset = (target_height - new_height) // 2
                new_image.paste(resized, (x_offset, y_offset))
                
                return new_image
            else:
                return image.resize(size, Image.Resampling.LANCZOS)
                
        except Exception as e:
            raise PreprocessingError(f"Failed to resize image: {str(e)}") from e
    
    def normalize_image(self, image: Image.Image, method: str = 'standard') -> np.ndarray:
        """
        Normalize image to array.
        
        Args:
            image: PIL Image
            method: Normalization method ('standard', 'minmax')
            
        Returns:
            Normalized numpy array
        """
        try:
            # Convert to numpy array
            img_array = np.array(image, dtype=np.float32)
            
            if method == 'standard':
                # Normalize to [0, 1]
                img_array = img_array / 255.0
            elif method == 'minmax':
                # Normalize to [-1, 1]
                img_array = (img_array / 127.5) - 1.0
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            return img_array
            
        except Exception as e:
            raise PreprocessingError(f"Failed to normalize image: {str(e)}") from e
    
    def preprocess_single_image(
        self, 
        image_path: Union[str, Path],
        add_batch_dimension: bool = True,
        normalization: str = 'standard'
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline for single image.
        
        Args:
            image_path: Path to image file
            add_batch_dimension: Whether to add batch dimension
            normalization: Normalization method
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load and resize image
            image = self.load_image(image_path)
            image = self.resize_image(image)
            
            # Normalize to array
            img_array = self.normalize_image(image, method=normalization)
            
            # Add batch dimension if requested
            if add_batch_dimension:
                img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            raise PreprocessingError(f"Failed to preprocess image {image_path}: {str(e)}") from e
    
    def preprocess_batch_images(
        self, 
        image_paths: List[Union[str, Path]],
        normalization: str = 'standard'
    ) -> np.ndarray:
        """
        Preprocess batch of images.
        
        Args:
            image_paths: List of image paths
            normalization: Normalization method
            
        Returns:
            Batch of preprocessed images
        """
        try:
            batch_arrays = []
            
            for image_path in image_paths:
                img_array = self.preprocess_single_image(
                    image_path, 
                    add_batch_dimension=False,
                    normalization=normalization
                )
                batch_arrays.append(img_array)
            
            return np.array(batch_arrays)
            
        except Exception as e:
            raise PreprocessingError(f"Failed to preprocess batch images: {str(e)}") from e
    
    def enhance_image(
        self, 
        image: Image.Image,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        sharpness: float = 1.0
    ) -> Image.Image:
        """
        Apply image enhancements.
        
        Args:
            image: PIL Image
            brightness: Brightness factor (1.0 = no change)
            contrast: Contrast factor (1.0 = no change)
            saturation: Saturation factor (1.0 = no change)
            sharpness: Sharpness factor (1.0 = no change)
            
        Returns:
            Enhanced PIL Image
        """
        try:
            enhanced = image
            
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(brightness)
            
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(contrast)
            
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(saturation)
            
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(sharpness)
            
            return enhanced
            
        except Exception as e:
            raise PreprocessingError(f"Failed to enhance image: {str(e)}") from e


class DataAugmentation:
    """Handles data augmentation for training."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data augmentation.
        
        Args:
            config: Augmentation configuration
        """
        self.logger = get_logger(self.__class__.__name__)
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required but not installed")
        
        # Default configuration
        self.config = {
            'rotation_range': 25,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'shear_range': 0.15,
            'zoom_range': 0.2,
            'horizontal_flip': True,
            'vertical_flip': False,
            'brightness_range': (0.8, 1.2),
            'fill_mode': 'nearest'
        }
        
        if config:
            self.config.update(config)
    
    def create_training_generator(
        self,
        directory: Union[str, Path],
        target_size: Tuple[int, int] = (96, 96),
        batch_size: int = 32,
        class_mode: str = 'categorical',
        shuffle: bool = True,
        validation_split: float = 0.0
    ) -> tf.keras.preprocessing.image.DirectoryIterator:
        """
        Create training data generator with augmentation.
        
        Args:
            directory: Data directory
            target_size: Target image size
            batch_size: Batch size
            class_mode: Class mode for generator
            shuffle: Whether to shuffle data
            validation_split: Validation split ratio
            
        Returns:
            Training data generator
        """
        try:
            # Create data generator with augmentation
            datagen = ImageDataGenerator(
                rescale=1.0/255.0,
                rotation_range=self.config['rotation_range'],
                width_shift_range=self.config['width_shift_range'],
                height_shift_range=self.config['height_shift_range'],
                shear_range=self.config['shear_range'],
                zoom_range=self.config['zoom_range'],
                horizontal_flip=self.config['horizontal_flip'],
                vertical_flip=self.config['vertical_flip'],
                brightness_range=self.config.get('brightness_range'),
                fill_mode=self.config['fill_mode'],
                validation_split=validation_split
            )
            
            # Create generator
            generator = datagen.flow_from_directory(
                str(directory),
                target_size=target_size,
                batch_size=batch_size,
                class_mode=class_mode,
                shuffle=shuffle,
                subset='training' if validation_split > 0 else None
            )
            
            self.logger.info(f"Created training generator: {generator.samples} samples, {len(generator.class_indices)} classes")
            return generator
            
        except Exception as e:
            raise AugmentationError(f"Failed to create training generator: {str(e)}") from e
    
    def create_validation_generator(
        self,
        directory: Union[str, Path],
        target_size: Tuple[int, int] = (96, 96),
        batch_size: int = 32,
        class_mode: str = 'categorical',
        shuffle: bool = False,
        validation_split: float = 0.0
    ) -> tf.keras.preprocessing.image.DirectoryIterator:
        """
        Create validation data generator without augmentation.
        
        Args:
            directory: Data directory
            target_size: Target image size
            batch_size: Batch size
            class_mode: Class mode for generator
            shuffle: Whether to shuffle data
            validation_split: Validation split ratio
            
        Returns:
            Validation data generator
        """
        try:
            # Create data generator without augmentation
            datagen = ImageDataGenerator(
                rescale=1.0/255.0,
                validation_split=validation_split
            )
            
            # Create generator
            generator = datagen.flow_from_directory(
                str(directory),
                target_size=target_size,
                batch_size=batch_size,
                class_mode=class_mode,
                shuffle=shuffle,
                subset='validation' if validation_split > 0 else None
            )
            
            self.logger.info(f"Created validation generator: {generator.samples} samples, {len(generator.class_indices)} classes")
            return generator
            
        except Exception as e:
            raise AugmentationError(f"Failed to create validation generator: {str(e)}") from e
    
    def preview_augmentation(
        self,
        image_path: Union[str, Path],
        num_previews: int = 9,
        save_path: Optional[Union[str, Path]] = None
    ) -> Optional[Path]:
        """
        Preview augmentation effects on a sample image.
        
        Args:
            image_path: Path to sample image
            num_previews: Number of augmented samples to show
            save_path: Optional path to save preview
            
        Returns:
            Path to saved preview or None
        """
        try:
            import matplotlib.pyplot as plt
            
            # Load image
            preprocessor = ImagePreprocessor()
            image = preprocessor.load_image(image_path)
            image_array = np.array(image)
            
            # Create augmentation generator
            datagen = ImageDataGenerator(
                rotation_range=self.config['rotation_range'],
                width_shift_range=self.config['width_shift_range'],
                height_shift_range=self.config['height_shift_range'],
                shear_range=self.config['shear_range'],
                zoom_range=self.config['zoom_range'],
                horizontal_flip=self.config['horizontal_flip'],
                vertical_flip=self.config['vertical_flip'],
                brightness_range=self.config.get('brightness_range'),
                fill_mode=self.config['fill_mode']
            )
            
            # Generate augmented samples
            img_array = image_array.reshape((1,) + image_array.shape)
            
            # Create subplot grid
            cols = 3
            rows = (num_previews + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else axes
            
            # Show original image
            if len(axes) > 0:
                axes[0].imshow(image_array.astype('uint8'))
                axes[0].set_title('Original')
                axes[0].axis('off')
            
            # Generate and show augmented samples
            i = 1
            for batch in datagen.flow(img_array, batch_size=1):
                if i >= num_previews:
                    break
                
                if i < len(axes):
                    augmented = batch[0].astype('uint8')
                    axes[i].imshow(augmented)
                    axes[i].set_title(f'Augmented {i}')
                    axes[i].axis('off')
                
                i += 1
            
            # Hide unused subplots
            for j in range(i, len(axes)):
                axes[j].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                if not save_path.name.endswith(('.png', '.jpg', '.jpeg')):
                    save_path = save_path.with_suffix('.png')
                
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Augmentation preview saved to {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create augmentation preview: {e}")
            return None