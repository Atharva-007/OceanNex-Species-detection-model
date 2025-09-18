#!/usr/bin/env python3
"""
Fish Species Classification Model
=================================

This script performs comprehensive analysis of the fish dataset and trains a CNN model
for image-based fish species classification.

Dataset: FishImgDataset with 31 fish species
Author: GitHub Copilot
Date: September 19, 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import json
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class FishDatasetAnalyzer:
    """Comprehensive analyzer for the fish species dataset"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.train_path = os.path.join(dataset_path, 'train')
        self.val_path = os.path.join(dataset_path, 'val')
        self.test_path = os.path.join(dataset_path, 'test')
        self.species_list = []
        self.dataset_stats = {}
        
    def analyze_dataset_structure(self):
        """Analyze the structure and distribution of the dataset"""
        print("="*60)
        print("FISH SPECIES DATASET ANALYSIS")
        print("="*60)
        
        # Get species list
        self.species_list = sorted([d for d in os.listdir(self.train_path) 
                                   if os.path.isdir(os.path.join(self.train_path, d))])
        
        print(f"Total number of fish species: {len(self.species_list)}")
        print(f"Species found: {', '.join(self.species_list[:5])}{'...' if len(self.species_list) > 5 else ''}")
        
        # Count images in each split
        splits = {'train': self.train_path, 'val': self.val_path, 'test': self.test_path}
        
        for split_name, split_path in splits.items():
            if os.path.exists(split_path):
                split_counts = {}
                total_images = 0
                
                for species in self.species_list:
                    species_path = os.path.join(split_path, species)
                    if os.path.exists(species_path):
                        count = len([f for f in os.listdir(species_path) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                        split_counts[species] = count
                        total_images += count
                
                self.dataset_stats[split_name] = split_counts
                print(f"\n{split_name.upper()} SET: {total_images} images")
        
        return self.dataset_stats
    
    def create_distribution_visualization(self):
        """Create visualizations of class distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Fish Species Dataset Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Overall distribution across splits
        splits_total = {split: sum(counts.values()) for split, counts in self.dataset_stats.items()}
        ax1 = axes[0, 0]
        bars = ax1.bar(splits_total.keys(), splits_total.values(), 
                      color=['#2E86AB', '#A23B72', '#F18F01'])
        ax1.set_title('Total Images per Split')
        ax1.set_ylabel('Number of Images')
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Plot 2: Train set distribution by species
        ax2 = axes[0, 1]
        train_counts = list(self.dataset_stats['train'].values())
        species_names = list(self.dataset_stats['train'].keys())
        
        ax2.barh(range(len(species_names)), train_counts, color='skyblue')
        ax2.set_yticks(range(len(species_names)))
        ax2.set_yticklabels([name.replace(' ', '\n') for name in species_names], fontsize=8)
        ax2.set_title('Training Set Distribution by Species')
        ax2.set_xlabel('Number of Images')
        
        # Plot 3: Statistics summary
        ax3 = axes[1, 0]
        stats_data = []
        for split, counts in self.dataset_stats.items():
            values = list(counts.values())
            stats_data.append([split, len(values), sum(values), 
                             np.mean(values), np.std(values), 
                             min(values), max(values)])
        
        stats_df = pd.DataFrame(stats_data, 
                               columns=['Split', 'Classes', 'Total', 'Mean', 'Std', 'Min', 'Max'])
        
        # Create table
        ax3.axis('tight')
        ax3.axis('off')
        table = ax3.table(cellText=stats_df.round(1).values,
                         colLabels=stats_df.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax3.set_title('Dataset Statistics Summary')
        
        # Plot 4: Class balance visualization
        ax4 = axes[1, 1]
        train_counts_array = np.array(train_counts)
        ax4.hist(train_counts_array, bins=15, color='lightcoral', alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(train_counts_array), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(train_counts_array):.1f}')
        ax4.axvline(np.median(train_counts_array), color='blue', linestyle='--', 
                   label=f'Median: {np.median(train_counts_array):.1f}')
        ax4.set_title('Distribution of Images per Class (Training Set)')
        ax4.set_xlabel('Number of Images per Class')
        ax4.set_ylabel('Number of Classes')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('fish_dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return stats_df
    
    def sample_images_visualization(self):
        """Display sample images from each species"""
        fig, axes = plt.subplots(6, 6, figsize=(18, 18))
        fig.suptitle('Sample Fish Images by Species', fontsize=16, fontweight='bold')
        
        for idx, species in enumerate(self.species_list[:36]):  # Show first 36 species
            if idx >= 36:
                break
                
            row, col = idx // 6, idx % 6
            species_path = os.path.join(self.train_path, species)
            
            # Get first available image
            images = [f for f in os.listdir(species_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if images:
                img_path = os.path.join(species_path, images[0])
                try:
                    img = Image.open(img_path)
                    axes[row, col].imshow(img)
                    axes[row, col].set_title(species, fontsize=10, fontweight='bold')
                    axes[row, col].axis('off')
                except Exception as e:
                    axes[row, col].text(0.5, 0.5, f'Error loading\n{species}', 
                                       ha='center', va='center', transform=axes[row, col].transAxes)
                    axes[row, col].axis('off')
        
        # Hide remaining subplots
        for idx in range(len(self.species_list), 36):
            row, col = idx // 6, idx % 6
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('fish_species_samples.png', dpi=300, bbox_inches='tight')
        plt.show()

class FishClassificationModel:
    """CNN model for fish species classification"""
    
    def __init__(self, dataset_path, img_size=(224, 224), batch_size=32):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = None
        
    def prepare_data_generators(self):
        """Prepare data generators with augmentation"""
        print("\nPreparing data generators...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation and test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        self.val_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'val'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.test_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.dataset_path, 'test'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.class_names = list(self.train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        print(f"Found {self.train_generator.samples} training images")
        print(f"Found {self.val_generator.samples} validation images")
        print(f"Found {self.test_generator.samples} test images")
        print(f"Number of classes: {self.num_classes}")
        
        return self.train_generator, self.val_generator, self.test_generator
    
    def build_model(self):
        """Build CNN model architecture"""
        print("\nBuilding CNN model...")
        
        self.model = models.Sequential([
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Fourth Conv Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Global Average Pooling instead of Flatten
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        print(f"Model built successfully!")
        self.model.summary()
        
        return self.model
    
    def train_model(self, epochs=50):
        """Train the model with callbacks"""
        print(f"\nStarting training for {epochs} epochs...")
        
        # Calculate class weights for balanced training
        y_train = self.train_generator.classes
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'best_fish_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            class_weight=class_weight_dict,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-3 Accuracy
        axes[1, 0].plot(self.history.history['top_3_accuracy'], label='Training Top-3 Accuracy')
        axes[1, 0].plot(self.history.history['val_top_3_accuracy'], label='Validation Top-3 Accuracy')
        axes[1, 0].set_title('Top-3 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-3 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            axes[1, 1].set_yscale('log')
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self):
        """Evaluate model on test set"""
        print("\nEvaluating model on test set...")
        
        # Get predictions
        test_loss, test_accuracy, test_top3_accuracy = self.model.evaluate(
            self.test_generator, verbose=1
        )
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Top-3 Accuracy: {test_top3_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Get detailed predictions for confusion matrix
        self.test_generator.reset()
        predictions = self.model.predict(self.test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_generator.classes
        
        # Classification report
        report = classification_report(
            true_classes, predicted_classes,
            target_names=self.class_names,
            output_dict=True
        )
        
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=self.class_names))
        
        # Confusion Matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - Fish Species Classification')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'test_accuracy': test_accuracy,
            'test_top3_accuracy': test_top3_accuracy,
            'test_loss': test_loss,
            'classification_report': report
        }
    
    def predict_single_image(self, image_path, top_k=5):
        """Predict single image with confidence scores"""
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get top k predictions
            top_indices = np.argsort(predictions[0])[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    'species': self.class_names[idx],
                    'confidence': float(predictions[0][idx])
                })
            
            return results
            
        except Exception as e:
            print(f"Error predicting image: {e}")
            return None

def main():
    """Main execution function"""
    # Configuration
    DATASET_PATH = "FishImgDataset"
    
    print("Starting Fish Species Classification Analysis")
    print("=" * 60)
    
    # Phase 1: Dataset Analysis
    print("\n📊 PHASE 1: DATASET ANALYSIS")
    analyzer = FishDatasetAnalyzer(DATASET_PATH)
    dataset_stats = analyzer.analyze_dataset_structure()
    stats_df = analyzer.create_distribution_visualization()
    analyzer.sample_images_visualization()
    
    # Phase 2: Model Training
    print("\n🤖 PHASE 2: MODEL TRAINING")
    model_trainer = FishClassificationModel(DATASET_PATH)
    
    # Prepare data
    train_gen, val_gen, test_gen = model_trainer.prepare_data_generators()
    
    # Build model
    model = model_trainer.build_model()
    
    # Train model
    history = model_trainer.train_model(epochs=50)
    
    # Visualize training
    model_trainer.plot_training_history()
    
    # Phase 3: Model Evaluation
    print("\n📈 PHASE 3: MODEL EVALUATION")
    evaluation_results = model_trainer.evaluate_model()
    
    # Save results
    results = {
        'dataset_stats': dataset_stats,
        'model_architecture': {
            'input_shape': model_trainer.img_size,
            'num_classes': model_trainer.num_classes,
            'total_parameters': model.count_params()
        },
        'training_results': {
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
        },
        'evaluation_results': evaluation_results,
        'class_names': model_trainer.class_names,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results to JSON
    with open('fish_classification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Analysis Complete!")
    print(f"Results saved to: fish_classification_results.json")
    print(f"Model saved to: best_fish_model.h5")
    print(f"Visualizations saved as PNG files")
    
    return model_trainer, results

if __name__ == "__main__":
    model_trainer, results = main()