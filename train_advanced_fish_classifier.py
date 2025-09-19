"""
Advanced Fish Classification Model Training
==========================================

This script trains a state-of-the-art CNN model on the unified fish dataset.
Features:
- Transfer learning with multiple pre-trained models
- Advanced data augmentation
- Class balancing techniques
- Comprehensive monitoring and evaluation
- Model ensembling capabilities
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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
try:
    import cv2
except ImportError:
    print("⚠️ OpenCV not available, but not required for basic training")
    cv2 = None
from PIL import Image
import time
from datetime import datetime

# Configure GPU
print("🔧 Configuring GPU...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU configured: {len(gpus)} GPU(s) available")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
else:
    print("ℹ️ No GPU detected, using CPU")

class AdvancedFishClassifier:
    """Advanced fish classification model with comprehensive training pipeline"""
    
    def __init__(self, dataset_dir="UnifiedFishDataset", model_name="EfficientNetB3"):
        self.dataset_dir = Path(dataset_dir)
        self.model_name = model_name
        self.img_size = (224, 224)
        self.batch_size = 32
        self.num_classes = 0
        self.class_names = []
        self.class_mapping = {}
        self.model = None
        self.history = None
        
        # Training parameters
        self.learning_rate = 0.001
        self.fine_tune_learning_rate = 0.0001
        self.epochs = 30
        self.fine_tune_epochs = 20
        self.patience = 5
        
        # Results storage
        self.results_dir = Path("training_results")
        self.results_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🎯 Initializing Advanced Fish Classifier")
        print(f"📁 Dataset: {self.dataset_dir}")
        print(f"🏗️ Architecture: {self.model_name}")
        
    def load_class_mapping(self):
        """Load class mapping from unified dataset preparation"""
        try:
            with open('class_mapping.json', 'r') as f:
                self.class_mapping = json.load(f)
            
            self.class_names = sorted(self.class_mapping.keys(), key=lambda x: self.class_mapping[x])
            self.num_classes = len(self.class_names)
            
            print(f"✅ Loaded {self.num_classes} classes")
            return True
            
        except FileNotFoundError:
            print("⚠️ class_mapping.json not found. Detecting classes from directory structure...")
            return self.detect_classes_from_directory()
    
    def detect_classes_from_directory(self):
        """Detect classes from directory structure if mapping file not found"""
        train_dir = self.dataset_dir / 'train'
        if not train_dir.exists():
            raise ValueError(f"Training directory not found: {train_dir}")
        
        class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
        self.class_names = sorted([d.name for d in class_dirs])
        self.num_classes = len(self.class_names)
        self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}
        
        print(f"✅ Detected {self.num_classes} classes from directory structure")
        return True
    
    def create_advanced_data_generators(self):
        """Create advanced data generators with augmentation"""
        print("🔄 Creating data generators with advanced augmentation...")
        
        # Training data augmentation - more aggressive
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=False,  # Fish usually don't swim upside down
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.1,
            fill_mode='reflect'
        )
        
        # Validation data - only rescaling
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Test data - only rescaling
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.dataset_dir / 'train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        self.val_generator = val_datagen.flow_from_directory(
            self.dataset_dir / 'val',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            self.dataset_dir / 'test',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )
        
        print(f"✅ Data generators created:")
        print(f"   📚 Training: {self.train_generator.samples} images")
        print(f"   🔍 Validation: {self.val_generator.samples} images")
        print(f"   🧪 Testing: {self.test_generator.samples} images")
        
        return self.train_generator, self.val_generator, self.test_generator
    
    def compute_class_weights(self):
        """Compute class weights to handle imbalanced dataset"""
        print("⚖️ Computing class weights for balanced training...")
        
        # Count samples per class
        class_counts = {}
        train_dir = self.dataset_dir / 'train'
        
        for class_name in self.class_names:
            class_dir = train_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob('*')))
                class_counts[class_name] = count
        
        # Calculate weights
        total_samples = sum(class_counts.values())
        class_weights = {}
        
        for i, class_name in enumerate(self.class_names):
            if class_name in class_counts:
                weight = total_samples / (self.num_classes * class_counts[class_name])
                class_weights[i] = weight
            else:
                class_weights[i] = 1.0
        
        print(f"✅ Class weights computed for {len(class_weights)} classes")
        
        # Show weight distribution
        weights_df = pd.DataFrame([
            {'Class': class_name, 'Samples': class_counts.get(class_name, 0), 'Weight': class_weights[i]}
            for i, class_name in enumerate(self.class_names)
        ])
        
        print("Top 5 classes by weight:")
        print(weights_df.nlargest(5, 'Weight')[['Class', 'Samples', 'Weight']])
        
        return class_weights
    
    def create_advanced_model(self):
        """Create advanced model with transfer learning"""
        print(f"🏗️ Creating {self.model_name} model...")
        
        # Input layer
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # Pre-trained base model
        if self.model_name == "EfficientNetB0":
            base_model = applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
            
        elif self.model_name == "ResNet50":
            base_model = applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
            
        elif self.model_name == "InceptionV3":
            # InceptionV3 requires 299x299 input
            self.img_size = (299, 299)
            inputs = keras.Input(shape=(*self.img_size, 3))
            base_model = applications.InceptionV3(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
            
        else:  # Default to MobileNetV2 - stable and efficient
            self.model_name = "MobileNetV2"
            base_model = applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Create the model
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Advanced head with dropout and batch normalization
        x = layers.Dense(512, activation='relu', name='dense_1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        self.model = keras.Model(inputs, outputs)
        self.base_model = base_model  # Store reference for fine-tuning
        
        print(f"✅ {self.model_name} model created")
        print(f"📊 Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def compile_model(self, learning_rate=None):
        """Compile model with advanced optimizer and metrics"""
        if learning_rate is None:
            learning_rate = self.learning_rate
            
        # Advanced optimizer with learning rate scheduling
        optimizer = optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        print(f"✅ Model compiled with learning rate: {learning_rate}")
    
    def create_callbacks(self, stage="initial"):
        """Create comprehensive callbacks for training"""
        callbacks_list = []
        
        # Model checkpoint
        checkpoint_path = self.results_dir / f"best_model_{stage}_{self.timestamp}.h5"
        checkpoint = callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        callbacks_list.append(checkpoint)
        
        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=self.patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks_list.append(early_stop)
        
        # Learning rate reduction
        lr_reducer = callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        callbacks_list.append(lr_reducer)
        
        # CSV logger
        csv_logger = callbacks.CSVLogger(
            self.results_dir / f"training_log_{stage}_{self.timestamp}.csv"
        )
        callbacks_list.append(csv_logger)
        
        print(f"✅ Created {len(callbacks_list)} callbacks for {stage} training")
        return callbacks_list
    
    def train_initial_phase(self, class_weights):
        """Train the model with frozen base (transfer learning phase 1)"""
        print("🚀 Starting Phase 1: Transfer Learning (Frozen Base)")
        
        # Compile model
        self.compile_model(self.learning_rate)
        
        # Create callbacks
        callbacks_list = self.create_callbacks("phase1")
        
        # Train
        self.history_phase1 = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator,
            class_weight=class_weights,
            callbacks=callbacks_list,
            verbose=1
        )
        
        print("✅ Phase 1 training completed")
        return self.history_phase1
    
    def train_fine_tuning_phase(self, class_weights):
        """Fine-tune the model with unfrozen layers (transfer learning phase 2)"""
        print("🔥 Starting Phase 2: Fine-tuning (Unfrozen Base)")
        
        # Unfreeze some layers of the base model
        base_model = self.base_model  # Use stored reference
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) // 2  # Unfreeze top 50% of layers
        
        # Freeze lower layers
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        print(f"🔧 Unfrozen {len(base_model.layers) - fine_tune_at} layers for fine-tuning")
        
        # Recompile with lower learning rate
        self.compile_model(self.fine_tune_learning_rate)
        
        # Create callbacks
        callbacks_list = self.create_callbacks("phase2_finetune")
        
        # Continue training
        initial_epochs = len(self.history_phase1.history['loss'])
        total_epochs = initial_epochs + self.fine_tune_epochs
        
        self.history_phase2 = self.model.fit(
            self.train_generator,
            epochs=total_epochs,
            initial_epoch=initial_epochs,
            validation_data=self.val_generator,
            class_weight=class_weights,
            callbacks=callbacks_list,
            verbose=1
        )
        
        print("✅ Phase 2 fine-tuning completed")
        return self.history_phase2
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("📊 Starting comprehensive model evaluation...")
        
        # Evaluate on test set
        test_loss, test_accuracy, test_top3_acc, test_precision, test_recall = self.model.evaluate(
            self.test_generator, verbose=1
        )
        
        print(f"\\n🎯 Test Results:")
        print(f"   Accuracy: {test_accuracy:.4f}")
        print(f"   Top-3 Accuracy: {test_top3_acc:.4f}")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall: {test_recall:.4f}")
        print(f"   Loss: {test_loss:.4f}")
        
        # Detailed predictions for confusion matrix
        print("🔍 Generating detailed predictions...")
        
        self.test_generator.reset()
        predictions = self.model.predict(self.test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        true_classes = self.test_generator.classes
        class_labels = list(self.test_generator.class_indices.keys())
        
        # Classification report
        report = classification_report(
            true_classes, predicted_classes,
            target_names=class_labels,
            output_dict=True
        )
        
        # Save detailed results
        results = {
            'timestamp': self.timestamp,
            'model_name': self.model_name,
            'test_accuracy': float(test_accuracy),
            'test_top3_accuracy': float(test_top3_acc),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_loss': float(test_loss),
            'classification_report': report,
            'total_parameters': int(self.model.count_params()),
            'training_samples': int(self.train_generator.samples),
            'validation_samples': int(self.val_generator.samples),
            'test_samples': int(self.test_generator.samples)
        }
        
        # Save results
        with open(self.results_dir / f"evaluation_results_{self.timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Evaluation completed. Results saved.")
        
        return results, predictions, predicted_classes, true_classes
    
    def create_comprehensive_visualizations(self, results, predictions, predicted_classes, true_classes):
        """Create comprehensive training and evaluation visualizations"""
        print("📊 Creating comprehensive visualizations...")
        
        plt.style.use('default')
        fig = plt.figure(figsize=(24, 16))
        
        # Combine training histories
        if hasattr(self, 'history_phase2'):
            # Combine both phases
            combined_history = {}
            for key in self.history_phase1.history.keys():
                combined_history[key] = (self.history_phase1.history[key] + 
                                       self.history_phase2.history[key])
        else:
            combined_history = self.history_phase1.history
        
        # 1. Training History - Accuracy
        ax1 = plt.subplot(3, 4, 1)
        epochs = range(1, len(combined_history['accuracy']) + 1)
        ax1.plot(epochs, combined_history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(epochs, combined_history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        
        # Mark phase transition if applicable
        if hasattr(self, 'history_phase2'):
            phase1_epochs = len(self.history_phase1.history['accuracy'])
            ax1.axvline(x=phase1_epochs, color='gray', linestyle='--', alpha=0.7, label='Fine-tuning Start')
        
        ax1.set_title('Model Accuracy', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Training History - Loss
        ax2 = plt.subplot(3, 4, 2)
        ax2.plot(epochs, combined_history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax2.plot(epochs, combined_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        if hasattr(self, 'history_phase2'):
            phase1_epochs = len(self.history_phase1.history['loss'])
            ax2.axvline(x=phase1_epochs, color='gray', linestyle='--', alpha=0.7, label='Fine-tuning Start')
        
        ax2.set_title('Model Loss', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Confusion Matrix
        ax3 = plt.subplot(3, 4, (3, 4))
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # For better visualization, show only if not too many classes
        if len(self.class_names) <= 20:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, yticklabels=self.class_names, ax=ax3)
            ax3.set_title('Confusion Matrix', fontweight='bold')
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('Actual')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax3.get_yticklabels(), rotation=0)
        else:
            # For many classes, show heatmap without annotations
            sns.heatmap(cm, cmap='Blues', ax=ax3, cbar=True)
            ax3.set_title(f'Confusion Matrix ({len(self.class_names)} classes)', fontweight='bold')
            ax3.set_xlabel('Predicted Class Index')
            ax3.set_ylabel('Actual Class Index')
        
        # 4. Top-K Accuracy if available
        ax4 = plt.subplot(3, 4, 5)
        if 'top_3_accuracy' in combined_history:
            ax4.plot(epochs, combined_history['top_3_accuracy'], 'g-', label='Top-3 Accuracy', linewidth=2)
            ax4.plot(epochs, combined_history['val_top_3_accuracy'], 'orange', label='Val Top-3 Accuracy', linewidth=2)
            ax4.set_title('Top-3 Accuracy', fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Top-3 Accuracy')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Precision and Recall
        ax5 = plt.subplot(3, 4, 6)
        if 'precision' in combined_history and 'recall' in combined_history:
            ax5.plot(epochs, combined_history['precision'], 'purple', label='Precision', linewidth=2)
            ax5.plot(epochs, combined_history['recall'], 'brown', label='Recall', linewidth=2)
            ax5.plot(epochs, combined_history['val_precision'], 'purple', linestyle='--', label='Val Precision', linewidth=2)
            ax5.plot(epochs, combined_history['val_recall'], 'brown', linestyle='--', label='Val Recall', linewidth=2)
            ax5.set_title('Precision & Recall', fontweight='bold')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Score')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Class-wise Performance
        ax6 = plt.subplot(3, 4, 7)
        class_report = results['classification_report']
        class_f1_scores = [class_report[class_name]['f1-score'] 
                          for class_name in self.class_names 
                          if class_name in class_report]
        
        if len(class_f1_scores) <= 15:  # Show individual bars for smaller datasets
            bars = ax6.bar(range(len(class_f1_scores)), class_f1_scores, color='lightgreen')
            ax6.set_xticks(range(len(self.class_names[:len(class_f1_scores)])))
            ax6.set_xticklabels(self.class_names[:len(class_f1_scores)], rotation=45, ha='right')
            ax6.set_ylabel('F1-Score')
        else:  # Show histogram for larger datasets
            ax6.hist(class_f1_scores, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
            ax6.set_xlabel('F1-Score')
            ax6.set_ylabel('Number of Classes')
        
        ax6.set_title('Per-Class F1-Scores', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 7. Model Performance Summary
        ax7 = plt.subplot(3, 4, 8)
        ax7.axis('off')
        
        summary_text = f"""
MODEL PERFORMANCE SUMMARY

🏗️ Architecture: {self.model_name}
📊 Parameters: {results['total_parameters']:,}

📈 Test Results:
• Accuracy: {results['test_accuracy']:.4f}
• Top-3 Accuracy: {results['test_top3_accuracy']:.4f}
• Precision: {results['test_precision']:.4f}
• Recall: {results['test_recall']:.4f}
• F1-Score: {class_report['macro avg']['f1-score']:.4f}

📚 Dataset:
• Training: {results['training_samples']:,} images
• Validation: {results['validation_samples']:,} images  
• Testing: {results['test_samples']:,} images
• Classes: {len(self.class_names)}

⏱️ Training: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 8. Prediction Confidence Distribution
        ax8 = plt.subplot(3, 4, 9)
        max_confidences = np.max(predictions, axis=1)
        ax8.hist(max_confidences, bins=30, color='salmon', alpha=0.7, edgecolor='black')
        ax8.set_xlabel('Prediction Confidence')
        ax8.set_ylabel('Number of Predictions')
        ax8.set_title('Prediction Confidence Distribution', fontweight='bold')
        ax8.axvline(np.mean(max_confidences), color='red', linestyle='--', label=f'Mean: {np.mean(max_confidences):.3f}')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Learning Rate Schedule (if available)
        ax9 = plt.subplot(3, 4, 10)
        if 'lr' in combined_history:
            ax9.plot(epochs, combined_history['lr'], 'orange', linewidth=2)
            ax9.set_title('Learning Rate Schedule', fontweight='bold')
            ax9.set_xlabel('Epoch')
            ax9.set_ylabel('Learning Rate')
            ax9.set_yscale('log')
            ax9.grid(True, alpha=0.3)
        else:
            ax9.text(0.5, 0.5, 'Learning Rate\\nSchedule\\nNot Available', 
                    ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('Learning Rate Schedule', fontweight='bold')
        
        # 10. Top Performing Classes
        ax10 = plt.subplot(3, 4, 11)
        class_accuracies = []
        for class_name in self.class_names:
            if class_name in class_report:
                class_accuracies.append((class_name, class_report[class_name]['f1-score']))
        
        class_accuracies.sort(key=lambda x: x[1], reverse=True)
        top_classes = class_accuracies[:10]  # Top 10
        
        if top_classes:
            names, scores = zip(*top_classes)
            bars = ax10.barh(range(len(names)), scores, color='lightcoral')
            ax10.set_yticks(range(len(names)))
            ax10.set_yticklabels(names)
            ax10.set_xlabel('F1-Score')
            ax10.set_title('Top 10 Performing Classes', fontweight='bold')
            ax10.invert_yaxis()
            
            # Add score labels
            for bar, score in zip(bars, scores):
                ax10.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                         f'{score:.3f}', va='center', fontsize=9)
        
        # 11. Worst Performing Classes
        ax11 = plt.subplot(3, 4, 12)
        worst_classes = class_accuracies[-10:]  # Bottom 10
        
        if worst_classes:
            names, scores = zip(*worst_classes)
            bars = ax11.barh(range(len(names)), scores, color='lightgray')
            ax11.set_yticks(range(len(names)))
            ax11.set_yticklabels(names)
            ax11.set_xlabel('F1-Score')
            ax11.set_title('Bottom 10 Performing Classes', fontweight='bold')
            ax11.invert_yaxis()
            
            # Add score labels
            for bar, score in zip(bars, scores):
                ax11.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                         f'{score:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        viz_path = self.results_dir / f"comprehensive_training_analysis_{self.timestamp}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comprehensive visualizations saved: {viz_path}")
        
        return fig
    
    def save_model_and_results(self, results):
        """Save the trained model and all results"""
        print("💾 Saving model and results...")
        
        # Save model
        model_path = self.results_dir / f"final_model_{self.model_name}_{self.timestamp}.h5"
        self.model.save(model_path)
        
        # Save model architecture plot
        try:
            plot_path = self.results_dir / f"model_architecture_{self.timestamp}.png"
            plot_model(self.model, to_file=plot_path, show_shapes=True, show_layer_names=True)
            print(f"   📊 Model architecture: {plot_path}")
        except:
            print("   ⚠️ Could not save model architecture plot")
        
        # Save training history
        if hasattr(self, 'history_phase2'):
            combined_history = {}
            for key in self.history_phase1.history.keys():
                combined_history[key] = (self.history_phase1.history[key] + 
                                       self.history_phase2.history[key])
        else:
            combined_history = self.history_phase1.history
        
        history_df = pd.DataFrame(combined_history)
        history_path = self.results_dir / f"training_history_{self.timestamp}.csv"
        history_df.to_csv(history_path, index=False)
        
        print(f"✅ Model and results saved:")
        print(f"   🤖 Model: {model_path}")
        print(f"   📈 History: {history_path}")
        
        return model_path
    
    def train_complete_model(self):
        """Complete training pipeline"""
        print("🚀 Starting complete fish classification model training pipeline\\n")
        
        start_time = time.time()
        
        try:
            # Step 1: Load class mapping
            self.load_class_mapping()
            
            # Step 2: Create data generators
            self.create_advanced_data_generators()
            
            # Step 3: Compute class weights
            class_weights = self.compute_class_weights()
            
            # Step 4: Create model
            self.create_advanced_model()
            
            # Step 5: Phase 1 - Transfer learning
            self.train_initial_phase(class_weights)
            
            # Step 6: Phase 2 - Fine-tuning
            self.train_fine_tuning_phase(class_weights)
            
            # Step 7: Evaluation
            results, predictions, predicted_classes, true_classes = self.evaluate_model()
            
            # Step 8: Visualizations
            self.create_comprehensive_visualizations(results, predictions, predicted_classes, true_classes)
            
            # Step 9: Save everything
            model_path = self.save_model_and_results(results)
            
            training_time = time.time() - start_time
            
            print(f"\\n🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"⏱️ Total training time: {training_time/3600:.2f} hours")
            print(f"🎯 Final test accuracy: {results['test_accuracy']:.4f}")
            print(f"📁 All results saved in: {self.results_dir}")
            
            return results, model_path
            
        except Exception as e:
            print(f"❌ Training failed with error: {e}")
            raise e

def main():
    """Main execution function"""
    print("🐟 Advanced Fish Classification Model Training")
    print("=" * 60)
    
    # Available models
    available_models = ["MobileNetV2", "ResNet50", "EfficientNetB0", "InceptionV3"]
    
    print(f"Available models: {', '.join(available_models)}")
    
    # For this run, use MobileNetV2 (stable and efficient)
    model_name = "MobileNetV2"
    
    print(f"Using model: {model_name}")
    print()
    
    # Initialize and train
    classifier = AdvancedFishClassifier(
        dataset_dir="UnifiedFishDataset",
        model_name=model_name
    )
    
    results, model_path = classifier.train_complete_model()
    
    print("\\n" + "="*60)
    print("ADVANCED FISH CLASSIFICATION TRAINING COMPLETED")
    print("="*60)
    
    return results, model_path, classifier

if __name__ == "__main__":
    results, model_path, classifier = main()