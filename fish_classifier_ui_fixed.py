"""
Fish Species Classifier - Streamlit Web UI
==========================================

A user-friendly web interface for fish species classification using a trained CNN model.
Upload an image and get predictions with confidence scores.

Usage:
    streamlit run fish_classifier_ui_fixed.py

Requirements:
    - streamlit
    - tensorflow  
    - pillow
    - matplotlib
    - pandas
    - numpy
"""

import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Check TensorFlow availability
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.error("❌ TensorFlow not installed. Please install with: `pip install tensorflow`")

# Configure page
st.set_page_config(
    page_title="🐟 Fish Species Classifier",
    page_icon="🐟",
    layout="wide"
)

# Global variables
MODEL = None
CLASS_NAMES = [
    'Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch',
    'Fourfinger Threadfin', 'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish',
    'Gourami', 'Grass Carp', 'Green Spotted Puffer', 'Indian Carp', 'Indo-Pacific Tarpon',
    'Jaguar Gapote', 'Janitor Fish', 'Knifefish', 'Long-Snouted Pipefish', 'Mosquito Fish',
    'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish', 'Silver Barb', 'Silver Carp',
    'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia'
]
IMG_SIZE = (224, 224)

@st.cache_resource
def load_fish_model():
    """Load the trained fish classification model"""
    if not TF_AVAILABLE:
        st.error("❌ TensorFlow not installed. Please install with: `pip install tensorflow`")
        return None
    
    # Look for available models
    model_files = [
        "fish_species_cnn_final.h5",
        "best_fish_cnn_model.h5",
        "fish_species_cnn_model.h5"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                model = load_model(model_file)
                st.success(f"✅ Model loaded: {model_file}")
                return model
            except Exception as e:
                st.error(f"❌ Error loading {model_file}: {e}")
                continue
    
    st.error("❌ No trained model found! Please train a model first.")
    return None

def preprocess_image(image):
    """Preprocess image for prediction"""
    try:
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize(IMG_SIZE)
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict_fish(model, image, top_k=5):
    """Predict fish species from image"""
    # Preprocess image
    img_array = preprocess_image(image)
    if img_array is None:
        return None
    
    try:
        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            confidence = float(predictions[idx])
            species = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Unknown_{idx}"
            
            results.append({
                'rank': i + 1,
                'species': species,
                'confidence': confidence,
                'percentage': confidence * 100
            })
        
        return results
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

def create_prediction_chart(results):
    """Create prediction results chart"""
    if not results:
        return None
    
    # Extract data
    species = [r['species'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    # Create chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Horizontal bar chart
    colors = plt.cm.viridis(np.linspace(0, 1, len(species)))
    bars = ax.barh(range(len(species)), confidences, color=colors)
    
    # Customize
    ax.set_yticks(range(len(species)))
    ax.set_yticklabels(species)
    ax.set_xlabel('Confidence Score')
    ax.set_title('Fish Species Prediction Results', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Add values on bars
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{conf:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    """Main UI application"""
    st.title("🐟 Fish Species Classifier")
    st.markdown("Upload an image to identify fish species using AI")
    
    # Initialize model
    global MODEL
    
    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Information")
        
        if st.button("🔄 Load Model"):
            with st.spinner("Loading model..."):
                MODEL = load_fish_model()
        
        # Display model status
        if MODEL is not None:
            st.success("✅ Model Ready")
            st.info(f"Classes: {len(CLASS_NAMES)}")
            st.info(f"Input Size: {IMG_SIZE}")
        else:
            st.warning("⚠️ Model not loaded")
            if TF_AVAILABLE:
                st.info("Click 'Load Model' to load the trained model")
            else:
                st.error("TensorFlow not available")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📷 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a fish image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a fish for species identification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.info(f"Image size: {image.size}")
            st.info(f"Image mode: {image.mode}")
    
    with col2:
        st.header("🔬 Prediction Results")
        
        if uploaded_file is not None and MODEL is not None:
            if st.button("🚀 Classify Fish", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Make prediction
                    prediction_results = predict_fish(MODEL, image)
                    
                    if prediction_results:
                        # Top prediction
                        top_pred = prediction_results[0]
                        st.markdown(f"## **{top_pred['species']}**")
                        st.metric("Confidence", f"{top_pred['percentage']:.1f}%")
                        
                        # All predictions table
                        st.markdown("### 📊 All Predictions")
                        predictions_df = pd.DataFrame(prediction_results)
                        predictions_df['percentage'] = predictions_df['percentage'].round(1)
                        st.dataframe(predictions_df, hide_index=True)
                        
                        # Chart
                        chart = create_prediction_chart(prediction_results)
                        if chart:
                            st.pyplot(chart)
                        
                        # Technical details
                        with st.expander("🔧 Technical Details"):
                            st.write(f"Max Confidence: {top_pred['confidence']:.4f}")
                            confidence_values = [r['confidence'] for r in prediction_results]
                            entropy = -sum(c * np.log(c + 1e-10) for c in confidence_values)
                            st.write(f"Prediction Entropy: {entropy:.4f}")
                    else:
                        st.error("Failed to make prediction")
        
        elif uploaded_file is not None and MODEL is None:
            st.warning("Please load a model first")
        else:
            st.info("Upload an image to see predictions")
    
    # Footer
    st.markdown("---")
    st.markdown("*Powered by TensorFlow and Streamlit*")

if __name__ == "__main__":
    main()