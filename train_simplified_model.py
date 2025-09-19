"""
Simplified Unified Fish Species CNN Training
===========================================

A more reliable training script with custom CNN architecture and fallback options.
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
from sklearn.utils.class_weight import compute_class_weight

# Set memory growth for GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("🚀 GPU detected and configured")
else:
    print("💻 Using CPU for training")

class SimplifiedFishCNNTrainer:
    """Simplified CNN trainer with reliable architecture"""
    
    def __init__(self, dataset_path="UnifiedFishDataset"):
        self.dataset_path = Path(dataset_path)
        self.img_size = (150, 150)  # Smaller size for faster training
        self.batch_size = 32
        self.num_classes = 35
        
        # Load class mapping
        with open('class_mapping.json', 'r') as f:
            self.class_mapping = json.load(f)
        
        # Reverse mapping for predictions
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        
        # Training parameters
        self.epochs = 50
        self.learning_rate = 0.001
        
        # Data generators
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        
        # Model
        self.model = None
        self.history = None
        
    def setup_data_generators(self):
        """Setup data generators with augmentation"""
        print("📊 Setting up data generators...")
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation and test data (no augmentation)
        val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.dataset_path / 'train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        self.val_generator = val_test_datagen.flow_from_directory(
            self.dataset_path / 'val',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )
        
        self.test_generator = val_test_datagen.flow_from_directory(
            self.dataset_path / 'test',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )
        
        print(f"✅ Training samples: {self.train_generator.samples}")
        print(f"✅ Validation samples: {self.val_generator.samples}")
        print(f"✅ Test samples: {self.test_generator.samples}")
        print(f"✅ Classes found: {self.train_generator.num_classes}")
        
        return self.train_generator, self.val_generator, self.test_generator
    
    def calculate_class_weights(self):
        """Calculate class weights to handle imbalanced classes"""
        print("⚖️ Calculating class weights...")
        
        # Get class distribution from training data
        class_counts = {}
        train_path = self.dataset_path / 'train'
        
        for class_dir in train_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                class_idx = self.class_mapping[class_name]
                count = len(list(class_dir.glob('*')))
                class_counts[class_idx] = count
        
        # Calculate weights
        class_indices = list(class_counts.keys())
        class_sample_counts = list(class_counts.values())
        
        weights = compute_class_weight(
            'balanced',
            classes=np.array(class_indices),
            y=np.repeat(class_indices, class_sample_counts)
        )
        
        class_weights = dict(zip(class_indices, weights))
        
        print(f"✅ Class weights calculated for {len(class_weights)} classes")
        return class_weights
    
    def create_custom_cnn_model(self):
        """Create custom CNN architecture"""
        print("🏗️ Creating custom CNN architecture...")
        
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Fifth convolutional block
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Classifier
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Custom CNN created with {model.count_params():,} parameters")
        
        self.model = model
        return model
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        callbacks_list = []
        
        # Model checkpoint
        checkpoint = callbacks.ModelCheckpoint(
            filepath="best_fish_classifier_simplified.keras",
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='max'
        )
        callbacks_list.append(checkpoint)
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        )
        callbacks_list.append(early_stopping)
        
        # Learning rate reduction
        lr_scheduler = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        )
        callbacks_list.append(lr_scheduler)
        
        # CSV logger
        csv_logger = callbacks.CSVLogger(
            'training_log_simplified.csv',
            append=True
        )
        callbacks_list.append(csv_logger)
        
        return callbacks_list
    
    def train_model(self):
        """Train the model"""
        print("🚀 Starting model training...")
        
        # Setup data and model
        self.setup_data_generators()
        class_weights = self.calculate_class_weights()
        self.create_custom_cnn_model()
        
        # Setup callbacks
        callbacks_list = self.setup_callbacks()
        
        # Train model
        print(f"🎯 Training for {self.epochs} epochs...")
        
        history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator,
            class_weight=class_weights,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.history = history
        print("✅ Training completed")
        return history
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("📊 Evaluating model performance...")
        
        # Load best model
        best_model_path = "best_fish_classifier_simplified.keras"
        if Path(best_model_path).exists():
            self.model = keras.models.load_model(best_model_path)
            print(f"✅ Loaded best model from {best_model_path}")
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(
            self.test_generator,
            verbose=1
        )
        
        print(f"📈 Test Accuracy: {test_accuracy:.4f}")
        print(f"📈 Test Loss: {test_loss:.4f}")
        
        # Generate predictions for detailed analysis
        print("🔍 Generating detailed predictions...")
        self.test_generator.reset()
        predictions = self.model.predict(self.test_generator, verbose=1)
        
        # Get true labels
        true_labels = self.test_generator.classes
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Classification report
        species_names = [self.idx_to_class[i] for i in range(self.num_classes)]
        class_report = classification_report(
            true_labels, 
            predicted_labels, 
            target_names=species_names,
            output_dict=True
        )
        
        # Save evaluation results
        evaluation_results = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'classification_report': class_report,
            'total_parameters': int(self.model.count_params()),
            'dataset_size': {
                'train': self.train_generator.samples,
                'val': self.val_generator.samples,
                'test': self.test_generator.samples,
                'total': self.train_generator.samples + self.val_generator.samples + self.test_generator.samples
            }
        }
        
        # Save results
        with open('simplified_model_evaluation.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Create visualizations
        self.create_training_plots()
        self.create_confusion_matrix(true_labels, predicted_labels, species_names)
        
        print("✅ Model evaluation completed")
        return evaluation_results
    
    def create_training_plots(self):
        """Create training history plots"""
        print("📈 Creating training plots...")
        
        if self.history is None:
            print("⚠️ No training history available")
            return
        
        history = self.history.history
        epochs = range(1, len(history['accuracy']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history_simplified.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Training plots saved as 'training_history_simplified.png'")
    
    def create_confusion_matrix(self, true_labels, predicted_labels, species_names):
        """Create and save confusion matrix"""
        print("📊 Creating confusion matrix...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Create figure
        plt.figure(figsize=(20, 16))
        
        # Plot confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=species_names,
            yticklabels=species_names,
            cbar_kws={'label': 'Number of Predictions'}
        )
        
        plt.title('Confusion Matrix - 35 Fish Species Classification (Simplified Model)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Species', fontsize=12)
        plt.ylabel('True Species', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('confusion_matrix_simplified.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Confusion matrix saved as 'confusion_matrix_simplified.png'")
    
    def save_model_info(self):
        """Save model information"""
        print("💾 Saving model information...")
        
        # Save model architecture
        with open('simplified_model_architecture.json', 'w') as f:
            f.write(self.model.to_json())
        
        # Save training configuration
        config = {
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'num_classes': self.num_classes,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'class_mapping': self.class_mapping
        }
        
        with open('simplified_training_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Model information saved")
    
    def complete_training_pipeline(self):
        """Execute complete training pipeline"""
        print("🚀 Starting simplified training pipeline...\n")
        
        # Train model
        print("=" * 60)
        print("TRAINING PHASE")
        print("=" * 60)
        self.train_model()
        
        # Evaluate model
        print("\n" + "=" * 60)
        print("EVALUATION PHASE")
        print("=" * 60)
        results = self.evaluate_model()
        
        # Save model info
        print("\n" + "=" * 60)
        print("SAVING MODEL")
        print("=" * 60)
        self.save_model_info()
        
        print("\n" + "=" * 60)
        print("TRAINING PIPELINE COMPLETED!")
        print("=" * 60)
        print(f"🎉 Final Test Accuracy: {results['test_accuracy']:.4f}")
        print("✅ Model and results saved!")
        
        return results

def main():
    """Main execution function"""
    print("🐟 Simplified Fish Species CNN Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = SimplifiedFishCNNTrainer(dataset_path="UnifiedFishDataset")
    
    # Execute complete training pipeline
    results = trainer.complete_training_pipeline()
    
    return results

if __name__ == "__main__":
    results = main()