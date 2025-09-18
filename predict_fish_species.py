"""
Fish Species Prediction and Visualization
=========================================

Complete prediction system for trained fish classification model.
Includes single image prediction, batch prediction, and visualization tools.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance
import json
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import load_model

class FishSpeciesPredictor:
    """Complete prediction system for fish species classification"""
    
    def __init__(self, model_path="fish_species_cnn_final.h5", results_path="fish_cnn_training_results.json"):
        self.model_path = model_path
        self.results_path = results_path
        self.model = None
        self.class_names = None
        self.img_size = None
        self.load_model_and_config()
        
    def load_model_and_config(self):
        """Load trained model and configuration"""
        try:
            print("🔄 Loading trained model...")
            self.model = load_model(self.model_path)
            print(f"✅ Model loaded from: {self.model_path}")
            
            # Load configuration
            if os.path.exists(self.results_path):
                with open(self.results_path, 'r') as f:
                    config = json.load(f)
                    self.class_names = config['class_names']
                    self.img_size = tuple(config['model_info']['input_shape'])
                    print(f"✅ Configuration loaded: {len(self.class_names)} classes")
            else:
                print("⚠️  Results file not found, using default configuration")
                self.class_names = None
                self.img_size = (224, 224)
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """Preprocess single image for prediction"""
        try:
            # Load and convert image
            img = Image.open(image_path).convert('RGB')
            
            # Resize to model input size
            img = img.resize(self.img_size)
            
            # Convert to array and normalize
            img_array = np.array(img) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array, img
            
        except Exception as e:
            print(f"❌ Error preprocessing image {image_path}: {e}")
            return None, None
    
    def predict_single_image(self, image_path, top_k=5, confidence_threshold=0.01):
        """Predict single image with detailed results"""
        print(f"🔍 Analyzing: {os.path.basename(image_path)}")
        
        # Preprocess image
        img_array, original_img = self.preprocess_image(image_path)
        if img_array is None:
            return None
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = {
            'image_path': image_path,
            'predictions': [],
            'max_confidence': float(np.max(predictions)),
            'prediction_entropy': float(-np.sum(predictions * np.log(predictions + 1e-10)))
        }
        
        print(f"\\n📊 Top {top_k} Predictions:")
        print("-" * 50)
        
        for i, idx in enumerate(top_indices):
            confidence = float(predictions[idx])
            if confidence >= confidence_threshold:
                species_name = self.class_names[idx] if self.class_names else f"Class_{idx}"
                results['predictions'].append({
                    'rank': i + 1,
                    'species': species_name,
                    'confidence': confidence,
                    'percentage': confidence * 100
                })
                
                print(f"{i+1}. {species_name:<25}: {confidence:.4f} ({confidence*100:5.1f}%)")
        
        # Prediction quality assessment
        max_conf = results['max_confidence']
        entropy = results['prediction_entropy']
        
        if max_conf > 0.8:
            quality = "High Confidence"
            quality_color = "green"
        elif max_conf > 0.5:
            quality = "Medium Confidence"
            quality_color = "orange"
        else:
            quality = "Low Confidence"
            quality_color = "red"
        
        results['prediction_quality'] = quality
        print(f"\\n🎯 Prediction Quality: {quality}")
        print(f"   Max Confidence: {max_conf:.3f}")
        print(f"   Entropy: {entropy:.3f}")
        
        return results, original_img
    
    def visualize_prediction(self, image_path, results, original_img, save_path=None):
        """Create comprehensive prediction visualization"""
        if results is None:
            return
        
        fig = plt.figure(figsize=(16, 10))
        
        # Main image display
        ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
        ax1.imshow(original_img)
        ax1.set_title(f"Input Image: {os.path.basename(image_path)}", 
                     fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Top predictions bar chart
        ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2, rowspan=1)
        
        if results['predictions']:
            species = [p['species'][:15] + '...' if len(p['species']) > 15 else p['species'] 
                      for p in results['predictions']]
            confidences = [p['confidence'] for p in results['predictions']]
            colors = plt.cm.viridis(np.linspace(0, 1, len(species)))
            
            bars = ax2.barh(range(len(species)), confidences, color=colors)
            ax2.set_yticks(range(len(species)))
            ax2.set_yticklabels(species)
            ax2.set_xlabel('Confidence Score')
            ax2.set_title('Top Predictions', fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            
            # Add confidence values on bars
            for i, (bar, conf) in enumerate(zip(bars, confidences)):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{conf:.3f}', va='center', fontsize=9)
        
        # Prediction details
        ax3 = plt.subplot2grid((3, 4), (1, 2), colspan=2, rowspan=1)
        
        details_text = f"""Prediction Analysis:
        
🎯 Best Prediction: {results['predictions'][0]['species'] if results['predictions'] else 'N/A'}
🎲 Confidence: {results['max_confidence']:.3f} ({results['max_confidence']*100:.1f}%)
📊 Quality: {results['prediction_quality']}
🔢 Entropy: {results['prediction_entropy']:.3f}

Model Info:
📝 Classes: {len(self.class_names) if self.class_names else 'Unknown'}
🖼️  Input Size: {self.img_size[0]}x{self.img_size[1]}"""
        
        ax3.text(0.05, 0.95, details_text, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        ax3.set_title('Prediction Details', fontweight='bold')
        ax3.axis('off')
        
        # Confidence distribution (if we have multiple predictions)
        ax4 = plt.subplot2grid((3, 4), (2, 0), colspan=4, rowspan=1)
        
        if results['predictions'] and len(results['predictions']) > 1:
            all_confidences = [p['confidence'] for p in results['predictions']]
            positions = range(len(all_confidences))
            
            bars = ax4.bar(positions, all_confidences, 
                          color=['red' if i == 0 else 'skyblue' for i in positions])
            ax4.set_xlabel('Prediction Rank')
            ax4.set_ylabel('Confidence')
            ax4.set_title('Confidence Distribution Across Top Predictions')
            ax4.set_xticks(positions)
            ax4.set_xticklabels([f'#{i+1}' for i in positions])
            ax4.grid(axis='y', alpha=0.3)
            
            # Highlight the best prediction
            if len(all_confidences) > 0:
                bars[0].set_color('red')
                bars[0].set_alpha(0.8)
        else:
            ax4.text(0.5, 0.5, 'Insufficient predictions for distribution', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved: {save_path}")
        
        plt.show()
    
    def predict_batch_images(self, image_folder, output_csv="batch_predictions.csv"):
        """Predict multiple images and save results"""
        print(f"🔄 Processing images from: {image_folder}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for file in os.listdir(image_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_folder, file))
        
        print(f"Found {len(image_files)} images to process")
        
        if not image_files:
            print("❌ No valid image files found")
            return
        
        batch_results = []
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\\nProcessing {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            results, _ = self.predict_single_image(image_path, top_k=3, confidence_threshold=0.01)
            
            if results and results['predictions']:
                batch_results.append({
                    'image_file': os.path.basename(image_path),
                    'image_path': image_path,
                    'top_prediction': results['predictions'][0]['species'],
                    'confidence': results['predictions'][0]['confidence'],
                    'confidence_percent': results['predictions'][0]['percentage'],
                    'prediction_quality': results['prediction_quality'],
                    'entropy': results['prediction_entropy']
                })
        
        # Save results to CSV
        if batch_results:
            import pandas as pd
            df = pd.DataFrame(batch_results)
            df.to_csv(output_csv, index=False)
            print(f"\\n✅ Batch results saved to: {output_csv}")
            
            # Print summary
            print(f"\\n📊 Batch Prediction Summary:")
            print(f"   Total images processed: {len(batch_results)}")
            print(f"   Average confidence: {df['confidence'].mean():.3f}")
            print(f"   High confidence predictions: {len(df[df['confidence'] > 0.8])}")
            print(f"   Medium confidence predictions: {len(df[(df['confidence'] > 0.5) & (df['confidence'] <= 0.8)])}")
            print(f"   Low confidence predictions: {len(df[df['confidence'] <= 0.5])}")
        
        return batch_results
    
    def create_prediction_report(self, image_path):
        """Create comprehensive prediction report"""
        results, original_img = self.predict_single_image(image_path, top_k=10)
        
        if results is None:
            return None
        
        # Create visualization
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        viz_path = f"prediction_{base_name}.png"
        self.visualize_prediction(image_path, results, original_img, viz_path)
        
        # Create detailed report
        report = {
            'image_analysis': {
                'file_name': os.path.basename(image_path),
                'file_path': image_path,
                'processed_size': self.img_size
            },
            'prediction_results': results,
            'model_info': {
                'model_file': self.model_path,
                'total_classes': len(self.class_names) if self.class_names else 'Unknown',
                'input_size': self.img_size
            },
            'visualization_saved': viz_path
        }
        
        # Save report
        report_path = f"prediction_report_{base_name}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Complete report saved: {report_path}")
        
        return report

def demo_prediction():
    """Demonstrate prediction capabilities"""
    print("🐟 FISH SPECIES PREDICTION DEMO")
    print("=" * 50)
    
    # Check if trained model exists
    if not os.path.exists("fish_species_cnn_final.h5"):
        print("❌ Trained model not found. Please run training first.")
        return
    
    try:
        # Initialize predictor
        predictor = FishSpeciesPredictor()
        
        # Look for sample images in test directory
        test_dir = "FishImgDataset/test"
        if os.path.exists(test_dir):
            print(f"\\n🔍 Looking for sample images in {test_dir}")
            
            # Get first species directory
            species_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
            
            if species_dirs:
                sample_species = species_dirs[0]
                species_path = os.path.join(test_dir, sample_species)
                image_files = [f for f in os.listdir(species_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if image_files:
                    sample_image = os.path.join(species_path, image_files[0])
                    print(f"\\n📸 Testing with sample image: {sample_image}")
                    print(f"   True species: {sample_species}")
                    
                    # Create prediction report
                    report = predictor.create_prediction_report(sample_image)
                    
                    if report:
                        print("\\n✅ Demo completed successfully!")
                        return predictor
        
        print("⚠️  No sample images found for demo")
        return predictor
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None

def main():
    """Main prediction interface"""
    print("🎯 FISH SPECIES PREDICTION SYSTEM")
    print("=" * 50)
    print("Options:")
    print("1. Run demo with sample image")
    print("2. Predict single image")
    print("3. Batch predict folder")
    print()
    
    # For now, just run the demo
    predictor = demo_prediction()
    
    if predictor:
        print("\\n🚀 Predictor ready for use!")
        print("\\nTo use the predictor programmatically:")
        print("  predictor = FishSpeciesPredictor()")
        print("  results, img = predictor.predict_single_image('path/to/image.jpg')")
        print("  predictor.visualize_prediction('path/to/image.jpg', results, img)")
    
    return predictor

if __name__ == "__main__":
    predictor = main()