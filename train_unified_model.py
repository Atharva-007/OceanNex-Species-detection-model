"""
Unified Fish Species CNN Model Training
======================================

This script trains a CNN model on the unified fish dataset containing 35 species
and 13,555 images with proper train/validation/test splits.

Key Features:
- Advanced CNN architecture with transfer learning
- Data augmentation for improved generalization
- Early stopping and learning rate scheduling
- Comprehensive model evaluation
- Model checkpointing and saving
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
from tensorflow.keras.applications import EfficientNetB3, ResNet50V2, MobileNetV3Large
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

class UnifiedFishCNNTrainer:
    """Advanced CNN trainer for unified fish species dataset"""
    
    def __init__(self, dataset_path="UnifiedFishDataset", model_name="EfficientNetB3"):
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name
        self.img_size = (224, 224)
        self.batch_size = 32
        self.num_classes = 35
        
        # Load class mapping
        with open('class_mapping.json', 'r') as f:
            self.class_mapping = json.load(f)
        
        # Reverse mapping for predictions
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        
        # Training parameters
        self.initial_epochs = 50
        self.fine_tune_epochs = 30
        self.initial_learning_rate = 0.001
        self.fine_tune_learning_rate = 0.0001
        
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
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1
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
    
    def create_model_architecture(self):
        """Create advanced CNN architecture with transfer learning"""
        print(f"🏗️ Creating {self.model_name} model architecture...")
        
        # Base model selection with proper input handling
        if self.model_name == "EfficientNetB3":
            try:
                base_model = EfficientNetB3(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(*self.img_size, 3)
                )
            except Exception as e:
                print(f"⚠️ EfficientNetB3 failed ({e}), switching to ResNet50V2")
                self.model_name = "ResNet50V2"
                base_model = ResNet50V2(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(*self.img_size, 3)
                )
        elif self.model_name == "ResNet50V2":
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif self.model_name == "MobileNetV3Large":
            base_model = MobileNetV3Large(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        else:
            # Fallback to ResNet50V2
            print(f"⚠️ Unknown model {self.model_name}, using ResNet50V2")
            self.model_name = "ResNet50V2"
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classifier
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax', name='predictions')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.initial_learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_5_accuracy']
        )
        
        print(f"✅ Model created with {model.count_params():,} parameters")
        print(f"✅ Trainable parameters: {sum(p.numel() for p in model.trainable_variables):,}")
        
        self.model = model
        self.base_model = base_model
        return model
    
    def setup_callbacks(self, stage="initial"):
        """Setup training callbacks"""
        callbacks_list = []
        
        # Model checkpoint
        checkpoint_path = f"best_unified_fish_model_{stage}.keras"
        checkpoint = callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
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
            patience=10 if stage == "initial" else 15,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        )
        callbacks_list.append(early_stopping)
        
        # Learning rate reduction
        lr_scheduler = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5 if stage == "initial" else 8,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        )
        callbacks_list.append(lr_scheduler)
        
        # CSV logger
        csv_logger = callbacks.CSVLogger(
            f'training_log_{stage}.csv',
            append=True
        )
        callbacks_list.append(csv_logger)
        
        return callbacks_list
    
    def train_initial_phase(self):
        """Initial training phase with frozen base model"""
        print("🚀 Starting initial training phase...")
        
        # Setup data and model
        self.setup_data_generators()
        class_weights = self.calculate_class_weights()
        self.create_model_architecture()
        
        # Setup callbacks
        callbacks_list = self.setup_callbacks("initial")
        
        # Train model
        history = self.model.fit(
            self.train_generator,
            epochs=self.initial_epochs,
            validation_data=self.val_generator,
            class_weight=class_weights,
            callbacks=callbacks_list,
            verbose=1
        )
        
        print("✅ Initial training phase completed")
        return history
    
    def fine_tune_model(self):
        """Fine-tune the model with unfrozen layers"""
        print("🔧 Starting fine-tuning phase...")
        
        # Unfreeze base model layers gradually
        self.base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(self.base_model.layers) // 2
        
        # Freeze layers before fine_tune_at
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.fine_tune_learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_5_accuracy']
        )
        
        print(f"✅ Unfrozen {len([l for l in self.base_model.layers if l.trainable])} layers")
        print(f"✅ Trainable parameters: {sum(p.numel() for p in self.model.trainable_variables):,}")
        
        # Setup callbacks for fine-tuning
        callbacks_list = self.setup_callbacks("fine_tune")
        
        # Continue training
        history = self.model.fit(
            self.train_generator,
            epochs=self.fine_tune_epochs,
            validation_data=self.val_generator,
            callbacks=callbacks_list,
            verbose=1
        )
        
        print("✅ Fine-tuning phase completed")
        return history
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("📊 Evaluating model performance...")
        
        # Load best model
        best_model_path = "best_unified_fish_model_fine_tune.keras"
        if Path(best_model_path).exists():
            self.model = keras.models.load_model(best_model_path)
            print(f"✅ Loaded best model from {best_model_path}")
        
        # Evaluate on test set
        test_loss, test_accuracy, test_top5_accuracy = self.model.evaluate(
            self.test_generator,
            verbose=1
        )
        
        print(f"📈 Test Accuracy: {test_accuracy:.4f}")
        print(f"📈 Test Top-5 Accuracy: {test_top5_accuracy:.4f}")
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
        
        # Save detailed results
        evaluation_results = {
            'test_accuracy': float(test_accuracy),
            'test_top5_accuracy': float(test_top5_accuracy),
            'test_loss': float(test_loss),
            'classification_report': class_report,
            'model_architecture': self.model_name,
            'total_parameters': int(self.model.count_params()),
            'dataset_size': {
                'train': self.train_generator.samples,
                'val': self.val_generator.samples,
                'test': self.test_generator.samples,
                'total': self.train_generator.samples + self.val_generator.samples + self.test_generator.samples
            }
        }
        
        # Save results
        with open('model_evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Create confusion matrix
        self.create_confusion_matrix(true_labels, predicted_labels, species_names)
        
        # Create performance visualizations
        self.create_performance_visualizations(evaluation_results)
        
        print("✅ Model evaluation completed")
        return evaluation_results
    
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
        
        plt.title('Confusion Matrix - 35 Fish Species Classification', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Species', fontsize=12)
        plt.ylabel('True Species', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Confusion matrix saved as 'confusion_matrix.png'")
    
    def create_performance_visualizations(self, results):
        """Create comprehensive performance visualizations"""
        print("📈 Creating performance visualizations...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Overall metrics
        ax1 = plt.subplot(2, 3, 1)
        metrics = ['Accuracy', 'Top-5 Accuracy']
        values = [results['test_accuracy'], results['test_top5_accuracy']]
        colors = ['#2E86AB', '#A23B72']
        
        bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Metrics', fontweight='bold')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Per-class accuracy
        ax2 = plt.subplot(2, 3, 2)
        class_report = results['classification_report']
        class_accuracies = []
        class_names = []
        
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict) and 'f1-score' in metrics:
                class_names.append(class_name)
                class_accuracies.append(metrics['f1-score'])
        
        # Sort by f1-score
        sorted_data = sorted(zip(class_names, class_accuracies), key=lambda x: x[1], reverse=True)
        class_names, class_accuracies = zip(*sorted_data)
        
        # Plot top 15 and bottom 15
        top_15 = list(class_accuracies[:15])
        ax2.barh(range(len(top_15)), top_15, color='lightgreen')
        ax2.set_yticks(range(len(top_15)))
        ax2.set_yticklabels(list(class_names[:15]), fontsize=8)
        ax2.set_xlabel('F1-Score')
        ax2.set_title('Top 15 Species Performance (F1-Score)', fontweight='bold')
        ax2.invert_yaxis()
        
        # 3. Bottom performers
        ax3 = plt.subplot(2, 3, 3)
        bottom_15 = list(class_accuracies[-15:])
        bottom_names = list(class_names[-15:])
        
        ax3.barh(range(len(bottom_15)), bottom_15, color='lightcoral')
        ax3.set_yticks(range(len(bottom_15)))
        ax3.set_yticklabels(bottom_names, fontsize=8)
        ax3.set_xlabel('F1-Score')
        ax3.set_title('Bottom 15 Species Performance (F1-Score)', fontweight='bold')
        ax3.invert_yaxis()
        
        # 4. Precision vs Recall
        ax4 = plt.subplot(2, 3, 4)
        precisions = []
        recalls = []
        
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
        
        ax4.scatter(precisions, recalls, alpha=0.6, s=50)
        ax4.plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Diagonal line
        ax4.set_xlabel('Precision')
        ax4.set_ylabel('Recall')
        ax4.set_title('Precision vs Recall by Species', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Model complexity info
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        
        model_info = f"""
MODEL INFORMATION

Architecture: {results['model_architecture']}
Total Parameters: {results['total_parameters']:,}

Dataset Statistics:
• Training: {results['dataset_size']['train']:,} images
• Validation: {results['dataset_size']['val']:,} images  
• Test: {results['dataset_size']['test']:,} images
• Total: {results['dataset_size']['total']:,} images

Performance Summary:
• Test Accuracy: {results['test_accuracy']:.4f}
• Test Top-5 Accuracy: {results['test_top5_accuracy']:.4f}
• Test Loss: {results['test_loss']:.4f}
"""
        
        ax5.text(0.05, 0.95, model_info, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 6. F1-score distribution
        ax6 = plt.subplot(2, 3, 6)
        f1_scores = [metrics['f1-score'] for class_name, metrics in class_report.items() 
                    if isinstance(metrics, dict) and 'f1-score' in metrics]
        
        ax6.hist(f1_scores, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax6.axvline(np.mean(f1_scores), color='red', linestyle='--', label=f'Mean: {np.mean(f1_scores):.3f}')
        ax6.axvline(np.median(f1_scores), color='blue', linestyle='--', label=f'Median: {np.median(f1_scores):.3f}')
        ax6.set_xlabel('F1-Score')
        ax6.set_ylabel('Number of Species')
        ax6.set_title('F1-Score Distribution Across Species', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Performance visualizations saved as 'model_performance_analysis.png'")
    
    def save_final_model(self):
        """Save the final trained model in multiple formats"""
        print("💾 Saving final model...")
        
        # Save complete model
        self.model.save('unified_fish_classifier_final.keras')
        
        # Save model weights
        self.model.save_weights('unified_fish_classifier_weights.h5')
        
        # Save model architecture
        with open('unified_fish_classifier_architecture.json', 'w') as f:
            f.write(self.model.to_json())
        
        # Save training configuration
        training_config = {
            'model_name': self.model_name,
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'num_classes': self.num_classes,
            'initial_epochs': self.initial_epochs,
            'fine_tune_epochs': self.fine_tune_epochs,
            'initial_learning_rate': self.initial_learning_rate,
            'fine_tune_learning_rate': self.fine_tune_learning_rate,
            'class_mapping': self.class_mapping
        }
        
        with open('training_config.json', 'w') as f:
            json.dump(training_config, f, indent=2)
        
        print("✅ Model saved in multiple formats:")
        print("   - unified_fish_classifier_final.keras (complete model)")
        print("   - unified_fish_classifier_weights.h5 (weights only)")
        print("   - unified_fish_classifier_architecture.json (architecture)")
        print("   - training_config.json (training configuration)")
    
    def train_complete_pipeline(self):
        """Execute complete training pipeline"""
        print("🚀 Starting complete training pipeline...\n")
        
        # Phase 1: Initial training
        print("=" * 60)
        print("PHASE 1: INITIAL TRAINING")
        print("=" * 60)
        initial_history = self.train_initial_phase()
        
        # Phase 2: Fine-tuning
        print("\n" + "=" * 60)
        print("PHASE 2: FINE-TUNING")
        print("=" * 60)
        fine_tune_history = self.fine_tune_model()
        
        # Phase 3: Evaluation
        print("\n" + "=" * 60)
        print("PHASE 3: MODEL EVALUATION")
        print("=" * 60)
        evaluation_results = self.evaluate_model()
        
        # Phase 4: Save final model
        print("\n" + "=" * 60)
        print("PHASE 4: SAVING FINAL MODEL")
        print("=" * 60)
        self.save_final_model()
        
        print("\n" + "=" * 60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"🎉 Final Test Accuracy: {evaluation_results['test_accuracy']:.4f}")
        print(f"🎉 Final Top-5 Accuracy: {evaluation_results['test_top5_accuracy']:.4f}")
        print("✅ All models and results saved!")
        
        return evaluation_results

def main():
    """Main execution function"""
    print("🐟 Unified Fish Species CNN Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = UnifiedFishCNNTrainer(
        dataset_path="UnifiedFishDataset",
        model_name="EfficientNetB3"  # You can change to "ResNet50V2" or "MobileNetV3Large"
    )
    
    # Execute complete training pipeline
    results = trainer.train_complete_pipeline()
    
    return results

if __name__ == "__main__":
    results = main()