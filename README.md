# Fish Species Classification Project

A comprehensive deep learning project for classifying fish species using Convolutional Neural Networks (CNN). This project includes dataset analysis, model training, evaluation, and prediction capabilities.

## 📊 Dataset Overview

**FishImgDataset** contains images of 31 different fish species with the following distribution:

- **Total Images**: 13,302
- **Training Set**: 8,791 images (66.1%)
- **Validation Set**: 2,751 images (20.7%)  
- **Test Set**: 1,760 images (13.2%)
- **Species Count**: 31 unique fish species

### Fish Species in Dataset

The dataset includes diverse freshwater and marine fish species:

| Species | Train | Val | Test | Total | % |
|---------|-------|-----|------|-------|---|
| Bangus | 171 | 52 | 34 | 257 | 1.9% |
| Big Head Carp | 201 | 63 | 43 | 307 | 2.3% |
| Black Spotted Barb | 200 | 63 | 40 | 303 | 2.3% |
| Catfish | 314 | 97 | 62 | 473 | 3.6% |
| Climbing Perch | 152 | 48 | 30 | 230 | 1.7% |
| Fourfinger Threadfin | 191 | 60 | 38 | 289 | 2.2% |
| Freshwater Eel | 271 | 84 | 55 | 410 | 3.1% |
| Glass Perchlet | 397 | 124 | 77 | 598 | 4.5% |
| Goby | 607 | 189 | 124 | 920 | 6.9% |
| Gold Fish | 206 | 65 | 41 | 312 | 2.3% |
| Gourami | 311 | 97 | 63 | 471 | 3.5% |
| **Grass Carp** | 1212 | 378 | 238 | 1828 | **13.7%** |
| Green Spotted Puffer | 110 | 34 | 22 | 166 | 1.2% |
| Indian Carp | 262 | 81 | 53 | 396 | 3.0% |
| Indo-Pacific Tarpon | 186 | 57 | 39 | 282 | 2.1% |
| Jaguar Gapote | 229 | 72 | 44 | 345 | 2.6% |
| Janitor Fish | 286 | 89 | 58 | 433 | 3.3% |
| Knifefish | 319 | 100 | 65 | 484 | 3.6% |
| Long-Snouted Pipefish | 256 | 81 | 52 | 389 | 2.9% |
| Mosquito Fish | 254 | 80 | 51 | 385 | 2.9% |
| Mudfish | 189 | 60 | 34 | 283 | 2.1% |
| Mullet | 174 | 55 | 38 | 267 | 2.0% |
| Pangasius | 193 | 61 | 38 | 292 | 2.2% |
| Perch | 293 | 91 | 60 | 444 | 3.3% |
| Scat Fish | 154 | 48 | 33 | 235 | 1.8% |
| Silver Barb | 329 | 105 | 64 | 498 | 3.7% |
| Silver Carp | 238 | 75 | 48 | 361 | 2.7% |
| Silver Perch | 283 | 88 | 57 | 428 | 3.2% |
| Snakehead | 232 | 72 | 47 | 351 | 2.6% |
| Tenpounder | 277 | 87 | 56 | 420 | 3.2% |
| Tilapia | 294 | 95 | 56 | 445 | 3.3% |

### Dataset Characteristics

- **Class Imbalance**: Ranges from 166 (Green Spotted Puffer) to 1,828 (Grass Carp) images
- **Image Formats**: JPEG, PNG
- **Image Sizes**: Variable (automatically resized to 224x224 for training)
- **Quality**: Mixed resolution and lighting conditions

## 🏗️ Model Architecture

### CNN Architecture

The model uses a custom CNN architecture optimized for fish species classification:

```
Input Layer: (224, 224, 3)
│
├── Conv2D Block 1: 32 filters (3x3) → BatchNorm → Conv2D 32 filters → MaxPool → Dropout(0.25)
├── Conv2D Block 2: 64 filters (3x3) → BatchNorm → Conv2D 64 filters → MaxPool → Dropout(0.25)  
├── Conv2D Block 3: 128 filters (3x3) → BatchNorm → Conv2D 128 filters → MaxPool → Dropout(0.25)
├── Conv2D Block 4: 256 filters (3x3) → BatchNorm → Conv2D 256 filters → MaxPool → Dropout(0.25)
├── Conv2D Block 5: 512 filters (3x3) → BatchNorm → GlobalAveragePooling → Dropout(0.5)
│
├── Dense Layer 1: 512 units → BatchNorm → Dropout(0.5)
├── Dense Layer 2: 256 units → BatchNorm → Dropout(0.3)
└── Output Layer: 31 units (Softmax)
```

### Model Specifications

- **Total Parameters**: ~6.8M (varies based on exact architecture)
- **Input Size**: 224 × 224 × 3
- **Output Classes**: 31 fish species
- **Activation**: ReLU (hidden layers), Softmax (output)
- **Optimizer**: Adam (lr=0.001, β₁=0.9, β₂=0.999)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Top-3 Accuracy

## 🔧 Data Preprocessing & Augmentation

### Training Data Augmentation
```python
- Rescaling: 1/255
- Rotation: ±25°
- Width/Height Shift: ±25%
- Shear: ±20%
- Zoom: ±25%
- Horizontal Flip: Yes
- Brightness: 0.8-1.2
- Fill Mode: Nearest
```

### Validation/Test Data
```python
- Rescaling: 1/255 only (no augmentation)
```

## 📈 Training Configuration

### Training Setup
- **Batch Size**: 32
- **Maximum Epochs**: 100
- **Early Stopping**: Patience 15 (monitor val_loss)
- **Learning Rate Reduction**: Factor 0.5, Patience 8
- **Class Weights**: Balanced (computed automatically)

### Callbacks Used
1. **EarlyStopping**: Prevents overfitting
2. **ReduceLROnPlateau**: Adaptive learning rate
3. **ModelCheckpoint**: Saves best model
4. **CSVLogger**: Training history logging

## 📊 Expected Performance Metrics

### Typical Results
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 80-90%
- **Test Accuracy**: 78-88%
- **Top-3 Accuracy**: 90-95%

### Performance Analysis
- **Best Performing Classes**: Species with more training data
- **Challenging Classes**: Similar-looking species (e.g., different carp varieties)
- **Class Imbalance Impact**: Handled through class weights and data augmentation

## 🚀 Usage Instructions

### 1. Dataset Analysis
```bash
cd "OceanNex-Species-detection-model"
python dataset_analysis.py
```

**Outputs:**
- `fish_dataset_comprehensive_analysis.png`: Visual analysis
- `fish_dataset_analysis_report.json`: Detailed statistics

### 2. Model Training
```bash
python train_fish_cnn.py
```

**Outputs:**
- `best_fish_cnn_model.h5`: Best model during training
- `fish_species_cnn_final.h5`: Final trained model
- `fish_cnn_training_history.png`: Training curves
- `fish_cnn_confusion_matrix.png`: Confusion matrix
- `fish_cnn_training_results.json`: Complete results
- `training_log.csv`: Epoch-by-epoch training log

### 3. Model Prediction
```bash
python predict_fish_species.py
```

**Features:**
- Single image prediction with confidence scores
- Batch prediction for multiple images
- Visualization of predictions
- Detailed prediction reports

### 4. Prediction API Usage
```python
from predict_fish_species import FishSpeciesPredictor

# Initialize predictor
predictor = FishSpeciesPredictor("fish_species_cnn_final.h5")

# Predict single image
results, img = predictor.predict_single_image("path/to/fish_image.jpg")

# Create visualization
predictor.visualize_prediction("path/to/fish_image.jpg", results, img)

# Batch prediction
batch_results = predictor.predict_batch_images("path/to/image_folder/")
```

## 📁 Project Structure

```
OceanNex-Species-detection-model/
│
├── FishImgDataset/
│   ├── train/          # Training images (31 species folders)
│   ├── val/            # Validation images (31 species folders)
│   └── test/           # Test images (31 species folders)
│
├── dataset_analysis.py           # Dataset exploration and analysis
├── train_fish_cnn.py            # Model training script
├── predict_fish_species.py      # Prediction and visualization
├── requirements.txt             # Python dependencies
├── README.md                    # This documentation
│
├── Generated Files:
├── fish_dataset_comprehensive_analysis.png
├── fish_dataset_analysis_report.json
├── fish_species_cnn_final.h5           # Trained model
├── fish_cnn_training_history.png
├── fish_cnn_confusion_matrix.png
├── fish_cnn_training_results.json
├── training_log.csv
└── prediction_*.png                     # Prediction visualizations
```

## 🔍 Model Evaluation Details

### Confusion Matrix Analysis
The confusion matrix shows per-class performance, highlighting:
- **Strong performers**: Classes with high diagonal values
- **Confusion patterns**: Similar species often confused
- **Class imbalance effects**: Underrepresented classes may show lower accuracy

### Classification Report Metrics
For each species:
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)  
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of test samples for each class

### Top-K Accuracy
- **Top-1 Accuracy**: Exact match prediction
- **Top-3 Accuracy**: Correct answer in top 3 predictions
- **Top-5 Accuracy**: Correct answer in top 5 predictions

## 🎯 Prediction Confidence Levels

### Confidence Interpretation
- **High Confidence (>0.8)**: Very reliable prediction
- **Medium Confidence (0.5-0.8)**: Moderately reliable
- **Low Confidence (<0.5)**: Uncertain prediction, review needed

### Quality Indicators
- **Entropy**: Measure of prediction uncertainty
- **Max Confidence**: Highest probability score
- **Top-K Distribution**: Spread of confidence across top predictions

## 🔧 Troubleshooting

### Common Issues

1. **GPU Memory Errors**
   - Reduce batch size to 16 or 8
   - Enable memory growth in GPU config

2. **Low Accuracy**
   - Check data quality and labeling
   - Increase training epochs
   - Adjust learning rate

3. **Overfitting**
   - Increase dropout rates
   - Add more data augmentation
   - Use early stopping

4. **Class Imbalance**
   - Use class weights (automatically handled)
   - Consider data augmentation for minority classes
   - Use stratified sampling

### Performance Optimization

1. **Faster Training**
   - Use GPU if available
   - Increase batch size (if memory allows)
   - Use mixed precision training

2. **Better Accuracy**
   - Increase model complexity
   - Use transfer learning (ResNet, EfficientNet)
   - Ensemble multiple models

## 📋 Requirements

### Python Dependencies
```
tensorflow>=2.13.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
Pillow>=8.3.0
```

### System Requirements
- **RAM**: 8GB+ recommended
- **Storage**: 5GB+ for dataset and models
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional but recommended)

## 🎓 Technical Details

### Data Loading Strategy
- Uses Keras ImageDataGenerator for efficient memory usage
- Implements real-time data augmentation
- Handles variable image sizes through resizing

### Training Strategy
- Progressive training with learning rate scheduling
- Class weight balancing for imbalanced dataset
- Multiple callback system for training optimization

### Evaluation Methodology
- Stratified test set evaluation
- Per-class performance analysis
- Confusion matrix visualization
- Top-K accuracy metrics

## 🚀 Future Improvements

### Model Enhancements
1. **Transfer Learning**: Use pretrained models (ResNet, EfficientNet)
2. **Data Augmentation**: Advanced techniques (CutMix, MixUp)
3. **Architecture**: Attention mechanisms, Vision Transformers
4. **Ensemble Methods**: Combine multiple model predictions

### Dataset Improvements
1. **Data Collection**: More balanced dataset
2. **Quality Control**: Image quality assessment
3. **Annotation**: More detailed labeling (age, size, habitat)
4. **Augmentation**: Synthetic data generation

### Application Features
1. **Real-time Prediction**: Video stream processing
2. **Mobile App**: TensorFlow Lite deployment
3. **Web Interface**: Online prediction service
4. **API Integration**: RESTful prediction API

## 📞 Support & Contribution

### Issues & Bugs
- Check existing documentation
- Verify dataset structure and file paths
- Ensure all dependencies are installed
- Check GPU configuration if using CUDA

### Contributing
1. Fork the repository
2. Create feature branch
3. Make improvements
4. Test thoroughly
5. Submit pull request

---

**Note**: This project is designed for educational and research purposes. The model performance may vary based on dataset quality, hardware configuration, and training parameters. For production use, consider additional validation and testing procedures.