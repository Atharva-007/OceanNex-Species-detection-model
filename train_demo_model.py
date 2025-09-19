"""
Quick Demo Fish Species CNN Training
===================================

A lightweight training script optimized for demonstration and quick results.
Uses a smaller subset of data and a more efficient model architecture.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import random

# Set memory growth for GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("🚀 GPU detected and configured")
else:
    print("💻 Using CPU for training")

class QuickDemoFishCNNTrainer:
    """Quick demonstration CNN trainer"""
    
    def __init__(self, dataset_path="UnifiedFishDataset", demo_samples_per_class=50):
        self.dataset_path = Path(dataset_path)
        self.demo_samples_per_class = demo_samples_per_class
        self.img_size = (96, 96)  # Smaller for faster training
        self.batch_size = 16
        self.demo_dataset_path = Path("DemoFishDataset")
        
        # Load class mapping
        with open('class_mapping.json', 'r') as f:
            self.class_mapping = json.load(f)
        
        self.num_classes = len(self.class_mapping)
        
        # Training parameters
        self.epochs = 20  # Fewer epochs for demo
        self.learning_rate = 0.001
        
    def create_demo_dataset(self):
        """Create a smaller demo dataset for quick training"""
        print(f"📦 Creating demo dataset with {self.demo_samples_per_class} samples per class...")
        
        # Remove existing demo dataset
        if self.demo_dataset_path.exists():
            import shutil
            shutil.rmtree(self.demo_dataset_path)
        
        # Create demo dataset structure
        for split in ['train', 'val', 'test']:
            (self.demo_dataset_path / split).mkdir(parents=True, exist_ok=True)
        
        demo_stats = {'train': 0, 'val': 0, 'test': 0}
        
        # Process each class
        for class_name in self.class_mapping.keys():
            print(f"  📁 Processing {class_name}...")
            
            # Collect all images for this class
            all_images = []
            for split in ['train', 'val', 'test']:
                source_dir = self.dataset_path / split / class_name
                if source_dir.exists():
                    images = list(source_dir.glob('*'))
                    all_images.extend([(img, split) for img in images])
            
            # Randomly sample images
            if len(all_images) > self.demo_samples_per_class:
                sampled_images = random.sample(all_images, self.demo_samples_per_class)
            else:
                sampled_images = all_images
            
            # Distribute to splits (70% train, 20% val, 10% test)
            random.shuffle(sampled_images)
            n_train = int(len(sampled_images) * 0.7)
            n_val = int(len(sampled_images) * 0.2)
            
            train_images = sampled_images[:n_train]
            val_images = sampled_images[n_train:n_train + n_val]
            test_images = sampled_images[n_train + n_val:]
            
            # Copy images to demo dataset
            import shutil
            
            for split_data, split_name in [(train_images, 'train'), (val_images, 'val'), (test_images, 'test')]:
                target_dir = self.demo_dataset_path / split_name / class_name
                target_dir.mkdir(exist_ok=True)
                
                for i, (img_path, _) in enumerate(split_data):
                    target_path = target_dir / f"{class_name}_{split_name}_{i+1}{img_path.suffix}"
                    shutil.copy2(img_path, target_path)
                    demo_stats[split_name] += 1
        
        print(f"✅ Demo dataset created:")
        print(f"   📊 Train: {demo_stats['train']} images")
        print(f"   📊 Val: {demo_stats['val']} images")
        print(f"   📊 Test: {demo_stats['test']} images")
        print(f"   📊 Total: {sum(demo_stats.values())} images")
        
        return demo_stats
    
    def setup_data_generators(self):
        """Setup data generators for demo dataset"""
        print("📊 Setting up data generators...")
        
        # Simple augmentation for faster training
        train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation and test data (no augmentation)
        val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.demo_dataset_path / 'train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        self.val_generator = val_test_datagen.flow_from_directory(
            self.demo_dataset_path / 'val',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.test_generator = val_test_datagen.flow_from_directory(
            self.demo_dataset_path / 'test',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"✅ Training samples: {self.train_generator.samples}")
        print(f"✅ Validation samples: {self.val_generator.samples}")
        print(f"✅ Test samples: {self.test_generator.samples}")
        
        return self.train_generator, self.val_generator, self.test_generator
    
    def create_lightweight_model(self):
        """Create lightweight CNN for quick training"""
        print("🏗️ Creating lightweight CNN architecture...")
        
        model = models.Sequential([
            # First block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Second block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Third block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Classifier
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Lightweight model created with {model.count_params():,} parameters")
        
        self.model = model
        return model
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        callbacks_list = []
        
        # Model checkpoint
        checkpoint = callbacks.ModelCheckpoint(
            filepath="demo_fish_classifier.keras",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        callbacks_list.append(checkpoint)
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
        callbacks_list.append(early_stopping)
        
        # CSV logger
        csv_logger = callbacks.CSVLogger('demo_training_log.csv')
        callbacks_list.append(csv_logger)
        
        return callbacks_list
    
    def train_model(self):
        """Train the model"""
        print("🚀 Starting demo model training...")
        
        # Create demo dataset
        self.create_demo_dataset()
        
        # Setup data and model
        self.setup_data_generators()
        self.create_lightweight_model()
        
        # Setup callbacks
        callbacks_list = self.setup_callbacks()
        
        # Train model
        print(f"🎯 Training for {self.epochs} epochs...")
        
        history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.history = history
        print("✅ Demo training completed")
        return history
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("📊 Evaluating demo model...")
        
        # Load best model
        if Path("demo_fish_classifier.keras").exists():
            self.model = keras.models.load_model("demo_fish_classifier.keras")
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(self.test_generator, verbose=1)
        
        print(f"📈 Demo Test Accuracy: {test_accuracy:.4f}")
        print(f"📈 Demo Test Loss: {test_loss:.4f}")
        
        # Generate predictions
        predictions = self.model.predict(self.test_generator, verbose=1)
        true_labels = self.test_generator.classes
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Classification report for available classes
        available_classes = list(set(true_labels))
        species_names = [list(self.class_mapping.keys())[i] for i in available_classes]
        
        class_report = classification_report(
            true_labels, 
            predicted_labels, 
            labels=available_classes,
            target_names=species_names,
            output_dict=True,
            zero_division=0
        )
        
        # Save results
        results = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'classification_report': class_report,
            'total_parameters': int(self.model.count_params()),
            'demo_samples_per_class': self.demo_samples_per_class,
            'total_classes': self.num_classes,
            'available_classes_in_test': len(available_classes)
        }
        
        with open('demo_model_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualizations
        self.create_demo_visualizations()
        
        print("✅ Demo evaluation completed")
        return results
    
    def create_demo_visualizations(self):
        """Create visualization for demo results"""
        print("📈 Creating demo visualizations...")
        
        # Training history
        if hasattr(self, 'history') and self.history is not None:
            history = self.history.history
            epochs = range(1, len(history['accuracy']) + 1)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Accuracy
            ax1.plot(epochs, history['accuracy'], 'b-', label='Training', linewidth=2)
            ax1.plot(epochs, history['val_accuracy'], 'r-', label='Validation', linewidth=2)
            ax1.set_title('Demo Model Accuracy', fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Loss
            ax2.plot(epochs, history['loss'], 'b-', label='Training', linewidth=2)
            ax2.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
            ax2.set_title('Demo Model Loss', fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('demo_training_results.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Demo visualizations saved")
    
    def complete_demo_pipeline(self):
        """Execute complete demo training pipeline"""
        print("🚀 Starting Fish Species Demo Training Pipeline\\n")
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        
        print("=" * 60)
        print("DEMO TRAINING")
        print("=" * 60)
        self.train_model()
        
        print("\\n" + "=" * 60)
        print("DEMO EVALUATION")
        print("=" * 60)
        results = self.evaluate_model()
        
        print("\\n" + "=" * 60)
        print("DEMO COMPLETED!")
        print("=" * 60)
        print(f"🎉 Demo Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"🔢 Model Parameters: {results['total_parameters']:,}")
        print(f"📊 Samples per class: {results['demo_samples_per_class']}")
        print(f"🐟 Total classes: {results['total_classes']}")
        print("✅ Demo model and results saved!")
        
        return results

def main():
    """Main execution function"""
    print("🐟 Quick Demo Fish Species CNN Training")
    print("=" * 50)
    
    # Initialize trainer with reduced samples for demo
    trainer = QuickDemoFishCNNTrainer(
        dataset_path="UnifiedFishDataset",
        demo_samples_per_class=30  # Small sample for quick demo
    )
    
    # Execute demo pipeline
    results = trainer.complete_demo_pipeline()
    
    return results

if __name__ == "__main__":
    results = main()