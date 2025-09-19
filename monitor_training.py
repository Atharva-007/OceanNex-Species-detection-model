"""
Training Progress Monitor
=========================

Monitor the training progress of the unified fish species model.
"""

import os
import time
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def check_training_progress():
    """Check and display current training progress"""
    base_path = Path(".")
    
    print("🔍 Checking training progress...\n")
    
    # Check if training logs exist
    initial_log = base_path / "training_log_initial.csv"
    fine_tune_log = base_path / "training_log_fine_tune.csv"
    simplified_log = base_path / "training_log_simplified.csv"
    
    # Check for model checkpoints
    initial_checkpoint = base_path / "best_unified_fish_model_initial.keras"
    fine_tune_checkpoint = base_path / "best_unified_fish_model_fine_tune.keras"
    simplified_checkpoint = base_path / "best_fish_classifier_simplified.keras"
    
    print("📊 Training Status:")
    print("-" * 40)
    
    if initial_log.exists():
        try:
            df = pd.read_csv(initial_log)
            latest_epoch = len(df)
            if latest_epoch > 0:
                latest_acc = df['accuracy'].iloc[-1]
                latest_val_acc = df['val_accuracy'].iloc[-1]
                latest_loss = df['loss'].iloc[-1]
                latest_val_loss = df['val_loss'].iloc[-1]
                
                print(f"✅ Initial Training: Epoch {latest_epoch}")
                print(f"   📈 Train Accuracy: {latest_acc:.4f}")
                print(f"   📈 Val Accuracy: {latest_val_acc:.4f}")
                print(f"   📉 Train Loss: {latest_loss:.4f}")
                print(f"   📉 Val Loss: {latest_val_loss:.4f}")
            else:
                print("🔄 Initial Training: Starting...")
        except Exception as e:
            print(f"⚠️ Error reading initial log: {e}")
    else:
        print("🔄 Initial Training: Not started or no log yet")
    
    if fine_tune_log.exists():
        try:
            df = pd.read_csv(fine_tune_log)
            latest_epoch = len(df)
            if latest_epoch > 0:
                latest_acc = df['accuracy'].iloc[-1]
                latest_val_acc = df['val_accuracy'].iloc[-1]
                latest_loss = df['loss'].iloc[-1]
                latest_val_loss = df['val_loss'].iloc[-1]
                
                print(f"✅ Fine-tuning: Epoch {latest_epoch}")
                print(f"   📈 Train Accuracy: {latest_acc:.4f}")
                print(f"   📈 Val Accuracy: {latest_val_acc:.4f}")
                print(f"   📉 Train Loss: {latest_loss:.4f}")
                print(f"   📉 Val Loss: {latest_val_loss:.4f}")
            else:
                print("🔄 Fine-tuning: Starting...")
        except Exception as e:
            print(f"⚠️ Error reading fine-tune log: {e}")
    else:
        print("🔄 Fine-tuning: Not started yet")
    
    if simplified_log.exists():
        try:
            df = pd.read_csv(simplified_log)
            latest_epoch = len(df)
            if latest_epoch > 0:
                latest_acc = df['accuracy'].iloc[-1]
                latest_val_acc = df['val_accuracy'].iloc[-1]
                latest_loss = df['loss'].iloc[-1]
                latest_val_loss = df['val_loss'].iloc[-1]
                
                print(f"✅ Simplified Training: Epoch {latest_epoch}")
                print(f"   📈 Train Accuracy: {latest_acc:.4f}")
                print(f"   📈 Val Accuracy: {latest_val_acc:.4f}")
                print(f"   📉 Train Loss: {latest_loss:.4f}")
                print(f"   📉 Val Loss: {latest_val_loss:.4f}")
            else:
                print("🔄 Simplified Training: Starting...")
        except Exception as e:
            print(f"⚠️ Error reading simplified log: {e}")
    else:
        print("🔄 Simplified Training: Not started yet")
    
    print()
    print("📁 Model Checkpoints:")
    print("-" * 40)
    
    if initial_checkpoint.exists():
        size_mb = initial_checkpoint.stat().st_size / (1024 * 1024)
        print(f"✅ Initial model checkpoint: {size_mb:.1f} MB")
    else:
        print("⏳ Initial model checkpoint: Not saved yet")
    
    if fine_tune_checkpoint.exists():
        size_mb = fine_tune_checkpoint.stat().st_size / (1024 * 1024)
        print(f"✅ Fine-tune model checkpoint: {size_mb:.1f} MB")
    else:
        print("⏳ Fine-tune model checkpoint: Not saved yet")
    
    if simplified_checkpoint.exists():
        size_mb = simplified_checkpoint.stat().st_size / (1024 * 1024)
        print(f"✅ Simplified model checkpoint: {size_mb:.1f} MB")
    else:
        print("⏳ Simplified model checkpoint: Not saved yet")
    
    # Check for evaluation results
    eval_results = base_path / "model_evaluation_results.json"
    if eval_results.exists():
        print("\n🎉 Training Completed!")
        print("-" * 40)
        try:
            with open(eval_results, 'r') as f:
                results = json.load(f)
            
            print(f"📊 Final Test Accuracy: {results['test_accuracy']:.4f}")
            print(f"📊 Final Top-5 Accuracy: {results['test_top5_accuracy']:.4f}")
            print(f"📊 Final Test Loss: {results['test_loss']:.4f}")
            print(f"🏗️ Model Architecture: {results['model_architecture']}")
            print(f"🔢 Total Parameters: {results['total_parameters']:,}")
        except Exception as e:
            print(f"⚠️ Error reading evaluation results: {e}")
    else:
        print("\n⏳ Evaluation: Not completed yet")
    
    print("\n" + "=" * 50)

def create_progress_visualization():
    """Create visualization of training progress"""
    try:
        # Check for training logs
        initial_log = Path("training_log_initial.csv")
        fine_tune_log = Path("training_log_fine_tune.csv")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress Monitor', fontsize=16, fontweight='bold')
        
        # Initial training plots
        if initial_log.exists():
            df_initial = pd.read_csv(initial_log)
            
            # Accuracy plot
            axes[0, 0].plot(df_initial['epoch'], df_initial['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
            axes[0, 0].plot(df_initial['epoch'], df_initial['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
            axes[0, 0].set_title('Initial Training - Accuracy')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Loss plot
            axes[0, 1].plot(df_initial['epoch'], df_initial['loss'], 'b-', label='Train Loss', linewidth=2)
            axes[0, 1].plot(df_initial['epoch'], df_initial['val_loss'], 'r-', label='Val Loss', linewidth=2)
            axes[0, 1].set_title('Initial Training - Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'Initial Training\nNot Started', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 1].text(0.5, 0.5, 'Initial Training\nNot Started', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 0].set_title('Initial Training - Accuracy')
            axes[0, 1].set_title('Initial Training - Loss')
        
        # Fine-tuning plots
        if fine_tune_log.exists():
            df_fine_tune = pd.read_csv(fine_tune_log)
            
            # Accuracy plot
            axes[1, 0].plot(df_fine_tune['epoch'], df_fine_tune['accuracy'], 'g-', label='Train Accuracy', linewidth=2)
            axes[1, 0].plot(df_fine_tune['epoch'], df_fine_tune['val_accuracy'], 'orange', label='Val Accuracy', linewidth=2)
            axes[1, 0].set_title('Fine-tuning - Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Loss plot
            axes[1, 1].plot(df_fine_tune['epoch'], df_fine_tune['loss'], 'g-', label='Train Loss', linewidth=2)
            axes[1, 1].plot(df_fine_tune['epoch'], df_fine_tune['val_loss'], 'orange', label='Val Loss', linewidth=2)
            axes[1, 1].set_title('Fine-tuning - Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Fine-tuning\nNot Started', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 1].text(0.5, 0.5, 'Fine-tuning\nNot Started', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 0].set_title('Fine-tuning - Accuracy')
            axes[1, 1].set_title('Fine-tuning - Loss')
        
        plt.tight_layout()
        plt.savefig('training_progress_monitor.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("📈 Progress visualization saved as 'training_progress_monitor.png'")
        
    except Exception as e:
        print(f"⚠️ Error creating visualization: {e}")

if __name__ == "__main__":
    print("🔍 Fish Species Training Progress Monitor")
    print("=" * 50)
    
    # Check current progress
    check_training_progress()
    
    # Create visualization
    create_progress_visualization()
    
    print(f"📅 Last checked: {time.strftime('%Y-%m-%d %H:%M:%S')}")