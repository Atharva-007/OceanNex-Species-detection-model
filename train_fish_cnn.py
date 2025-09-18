"""
Fish Species CNN Model Training
==============================

Complete CNN model implementation for fish species classification.
Includes data preprocessing, model architecture, training, and evaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class FishCNNTrainer:
    """Complete CNN trainer for fish species classification"""
    
    def __init__(self, dataset_path="FishImgDataset", img_size=(224, 224), batch_size=32):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = None
        self.num_classes = None
        
        # Check GPU availability
        self.setup_gpu()
        
    def setup_gpu(self):
        """Setup GPU configuration"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"🚀 GPU Available: {len(gpus)} GPU(s) detected")
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        else:
            print("💻 Running on CPU")
            
    def prepare_data_generators(self):
        """Prepare data generators with comprehensive augmentation"""
        print("\\n📊 Preparing data generators...")
        
        # Enhanced data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,
            width_shift_range=0.25,
            height_shift_range=0.25,
            shear_range=0.2,
            zoom_range=0.25,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.8, 1.2],
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
        
        print(f"✅ Training samples: {self.train_generator.samples:,}")
        print(f"✅ Validation samples: {self.val_generator.samples:,}")
        print(f"✅ Test samples: {self.test_generator.samples:,}")
        print(f"✅ Number of classes: {self.num_classes}")
        print(f"✅ Image size: {self.img_size}")
        print(f"✅ Batch size: {self.batch_size}")
        
        return self.train_generator, self.val_generator, self.test_generator
        
    def build_cnn_model(self):
        """Build enhanced CNN model architecture"""
        print("\\n🏗️  Building CNN model...")
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(*self.img_size, 3)),
            
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Fifth Convolutional Block
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model with advanced optimizer
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        self.model = model
        
        print(f"✅ Model built successfully!")
        print(f"✅ Total parameters: {model.count_params():,}")
        print(f"✅ Trainable parameters: {sum([np.prod(v.get_shape()) for v in model.trainable_weights]):,}")
        
        return model
    
    def calculate_class_weights(self):
        """Calculate class weights for balanced training"""
        y_train = self.train_generator.classes
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        print(f"\\n⚖️  Class weights calculated for balanced training")
        print(f"   Weight range: {min(class_weights):.2f} - {max(class_weights):.2f}")
        
        return class_weight_dict
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1,
                cooldown=3
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                'best_fish_cnn_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='max'
            ),
            
            # CSV Logger
            callbacks.CSVLogger(
                'training_log.csv',
                separator=',',
                append=False
            )
        ]
        
        return callbacks_list
    
    def train_model(self, epochs=100):
        """Train the CNN model"""
        print(f"\\n🚀 Starting training for up to {epochs} epochs...")
        
        # Calculate class weights
        class_weight_dict = self.calculate_class_weights()
        
        # Setup callbacks
        callbacks_list = self.setup_callbacks()
        
        # Calculate steps per epoch
        steps_per_epoch = self.train_generator.samples // self.batch_size
        validation_steps = self.val_generator.samples // self.batch_size
        
        print(f"📈 Training configuration:")
        print(f"   Steps per epoch: {steps_per_epoch}")
        print(f"   Validation steps: {validation_steps}")
        print(f"   Using class weights: Yes")
        print(f"   Callbacks: {len(callbacks_list)}")
        
        # Start training
        start_time = datetime.now()
        
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=validation_steps,
            class_weight=class_weight_dict,
            callbacks=callbacks_list,
            verbose=1
        )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        print(f"\\n✅ Training completed!")
        print(f"⏱️  Training duration: {training_duration}")
        print(f"📊 Total epochs: {len(self.history.history['loss'])}")
        
        return self.history
    
    def plot_training_history(self):
        """Create comprehensive training history plots"""
        print("\\n📊 Creating training history visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fish Species CNN Training History', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.history.history['loss']) + 1)
        
        # Accuracy plot
        axes[0, 0].plot(epochs, self.history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(epochs, self.history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[0, 1].plot(epochs, self.history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 1].plot(epochs, self.history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-3 Accuracy
        axes[0, 2].plot(epochs, self.history.history['top_3_accuracy'], 'b-', label='Training Top-3', linewidth=2)
        axes[0, 2].plot(epochs, self.history.history['val_top_3_accuracy'], 'r-', label='Validation Top-3', linewidth=2)
        axes[0, 2].set_title('Top-3 Accuracy')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Top-3 Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning Rate (if available)
        if 'lr' in self.history.history:
            axes[1, 0].plot(epochs, self.history.history['lr'], 'g-', linewidth=2)
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].axis('off')
        
        # Training summary statistics
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        final_train_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        best_val_acc = max(self.history.history['val_accuracy'])
        
        summary_text = f"""Training Summary:
        
Final Training Accuracy: {final_train_acc:.4f}
Final Validation Accuracy: {final_val_acc:.4f}
Best Validation Accuracy: {best_val_acc:.4f}

Final Training Loss: {final_train_loss:.4f}
Final Validation Loss: {final_val_loss:.4f}

Total Epochs: {len(epochs)}
Parameters: {self.model.count_params():,}"""
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        # Validation accuracy improvement
        val_acc_diff = np.diff(self.history.history['val_accuracy'])
        axes[1, 2].plot(epochs[1:], val_acc_diff, 'purple', linewidth=2)
        axes[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 2].set_title('Validation Accuracy Improvement')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Accuracy Change')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fish_cnn_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Training history visualization saved: fish_cnn_training_history.png")
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("\\n🎯 Evaluating model performance...")
        
        # Evaluate on test set
        test_loss, test_accuracy, test_top3_accuracy = self.model.evaluate(
            self.test_generator, verbose=1
        )
        
        print(f"\\n📊 Test Set Performance:")
        print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"   Test Top-3 Accuracy: {test_top3_accuracy:.4f} ({test_top3_accuracy*100:.2f}%)")
        print(f"   Test Loss: {test_loss:.4f}")
        
        # Get detailed predictions
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
        
        print("\\n📋 Detailed Classification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=self.class_names))
        
        # Create confusion matrix visualization
        self.plot_confusion_matrix(true_classes, predicted_classes)
        
        # Per-class accuracy analysis
        self.analyze_per_class_performance(true_classes, predicted_classes, predictions)
        
        return {
            'test_accuracy': test_accuracy,
            'test_top3_accuracy': test_top3_accuracy,
            'test_loss': test_loss,
            'classification_report': report
        }
    
    def plot_confusion_matrix(self, true_classes, predicted_classes):
        """Create and display confusion matrix"""
        cm = confusion_matrix(true_classes, predicted_classes)
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Number of Predictions'})
        plt.title('Confusion Matrix - Fish Species Classification', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Species', fontsize=12)
        plt.ylabel('True Species', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('fish_cnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Confusion matrix saved: fish_cnn_confusion_matrix.png")
    
    def analyze_per_class_performance(self, true_classes, predicted_classes, predictions):
        """Analyze per-class performance"""
        print("\\n🔍 Per-class Performance Analysis:")
        
        # Calculate per-class accuracy
        class_accuracies = []
        for i, class_name in enumerate(self.class_names):
            class_mask = true_classes == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(predicted_classes[class_mask] == i)
                class_accuracies.append((class_name, class_acc, np.sum(class_mask)))
        
        # Sort by accuracy
        class_accuracies.sort(key=lambda x: x[1], reverse=True)
        
        print("\\nTop 5 Best Performing Classes:")
        for name, acc, count in class_accuracies[:5]:
            print(f"   {name:<25}: {acc*100:5.1f}% ({count} samples)")
        
        print("\\nBottom 5 Performing Classes:")
        for name, acc, count in class_accuracies[-5:]:
            print(f"   {name:<25}: {acc*100:5.1f}% ({count} samples)")
    
    def save_model_and_results(self, evaluation_results):
        """Save model and comprehensive results"""
        print("\\n💾 Saving model and results...")
        
        # Save model
        self.model.save('fish_species_cnn_final.h5')
        
        # Save model architecture as JSON
        model_json = self.model.to_json()
        with open('fish_cnn_architecture.json', 'w') as json_file:
            json_file.write(model_json)
        
        # Prepare comprehensive results
        results = {
            'model_info': {
                'architecture': 'Custom CNN',
                'input_shape': self.img_size,
                'num_classes': self.num_classes,
                'total_parameters': int(self.model.count_params()),
                'trainable_parameters': int(sum([np.prod(v.get_shape()) for v in self.model.trainable_weights]))
            },
            'training_config': {
                'batch_size': self.batch_size,
                'image_size': self.img_size,
                'data_augmentation': True,
                'class_weights': True
            },
            'training_results': {
                'total_epochs': len(self.history.history['loss']),
                'final_train_accuracy': float(self.history.history['accuracy'][-1]),
                'final_val_accuracy': float(self.history.history['val_accuracy'][-1]),
                'best_val_accuracy': float(max(self.history.history['val_accuracy'])),
                'final_train_loss': float(self.history.history['loss'][-1]),
                'final_val_loss': float(self.history.history['val_loss'][-1])
            },
            'evaluation_results': evaluation_results,
            'class_names': self.class_names,
            'training_history': {
                'accuracy': [float(x) for x in self.history.history['accuracy']],
                'val_accuracy': [float(x) for x in self.history.history['val_accuracy']],
                'loss': [float(x) for x in self.history.history['loss']],
                'val_loss': [float(x) for x in self.history.history['val_loss']]
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results to JSON
        with open('fish_cnn_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Model saved: fish_species_cnn_final.h5")
        print(f"✅ Architecture saved: fish_cnn_architecture.json")
        print(f"✅ Results saved: fish_cnn_training_results.json")
        
        return results

def main():
    """Main training pipeline"""
    print("🐟 FISH SPECIES CNN TRAINING PIPELINE")
    print("="*60)
    print("This script will train a CNN model for fish species classification.")
    print("Expected dataset structure: FishImgDataset/[train|val|test]/[species_folders]")
    print()
    
    try:
        # Initialize trainer
        trainer = FishCNNTrainer(
            dataset_path="FishImgDataset",
            img_size=(224, 224),
            batch_size=32
        )
        
        # Prepare data
        print("\\n" + "="*60)
        print("PHASE 1: DATA PREPARATION")
        print("="*60)
        train_gen, val_gen, test_gen = trainer.prepare_data_generators()
        
        # Build model
        print("\\n" + "="*60)
        print("PHASE 2: MODEL BUILDING")
        print("="*60)
        model = trainer.build_cnn_model()
        
        # Display model summary
        print("\\n📋 Model Architecture Summary:")
        model.summary()
        
        # Train model
        print("\\n" + "="*60)
        print("PHASE 3: MODEL TRAINING")
        print("="*60)
        history = trainer.train_model(epochs=100)
        
        # Plot training history
        trainer.plot_training_history()
        
        # Evaluate model
        print("\\n" + "="*60)
        print("PHASE 4: MODEL EVALUATION")
        print("="*60)
        evaluation_results = trainer.evaluate_model()
        
        # Save everything
        print("\\n" + "="*60)
        print("PHASE 5: SAVING RESULTS")
        print("="*60)
        results = trainer.save_model_and_results(evaluation_results)
        
        # Final summary
        print("\\n" + "="*60)
        print("🎉 TRAINING COMPLETE!")
        print("="*60)
        print(f"✅ Model trained successfully")
        print(f"✅ Final validation accuracy: {results['training_results']['final_val_accuracy']*100:.2f}%")
        print(f"✅ Test accuracy: {results['evaluation_results']['test_accuracy']*100:.2f}%")
        print(f"✅ Test top-3 accuracy: {results['evaluation_results']['test_top3_accuracy']*100:.2f}%")
        print(f"✅ Model saved as: fish_species_cnn_final.h5")
        print(f"✅ All results and visualizations saved")
        
        return trainer, results
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("Please check that:")
        print("  • FishImgDataset directory exists")
        print("  • Required packages are installed")
        print("  • Sufficient disk space available")
        return None, None

if __name__ == "__main__":
    trainer, results = main()