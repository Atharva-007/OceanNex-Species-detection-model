"""
Streamlit Application Module
===========================

Enhanced Streamlit UI for fish species classification with modular design,
improved error handling, and better user experience.
"""

import streamlit as st
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from src.core.prediction import FishPredictor
    from src.core.model_manager import ModelManager
    from src.data.dataset_manager import DatasetManager
    from src.utils.logging_utils import get_logger
    from src.utils.visualization import create_prediction_chart, create_confidence_plot
    from config.settings import Settings
    from config.model_configs import ModelConfig
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please ensure the project structure is correct and all dependencies are installed.")
    st.stop()

# Configure page
st.set_page_config(
    page_title="🐟 Fish Species Classifier",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class FishClassifierUI:
    """Enhanced Streamlit UI for fish species classification"""
    
    def __init__(self):
        self.settings = Settings()
        self.model_manager = None
        self.predictor = None
        
        # Initialize dataset manager with available dataset
        self.dataset_manager = self._initialize_dataset_manager()
        
        # Initialize session state
        self._init_session_state()
    
    def _initialize_dataset_manager(self) -> Optional['DatasetManager']:
        """Initialize dataset manager with the best available dataset"""
        try:
            # List of datasets to check in order of preference
            dataset_candidates = [
                "FishImgDataset",
                "UnifiedFishDataset", 
                "DemoFishDataset"
            ]
            
            for dataset_name in dataset_candidates:
                dataset_path = Path(dataset_name)
                if dataset_path.exists() and dataset_path.is_dir():
                    logger.info(f"Using dataset: {dataset_name}")
                    return DatasetManager(
                        dataset_path=str(dataset_path),
                        class_mapping_file="class_mapping.json" if Path("class_mapping.json").exists() else None
                    )
            
            # If no dataset found, create a dummy manager for basic functionality
            logger.warning("No dataset found, creating minimal dataset manager")
            return None
            
        except Exception as e:
            logger.error(f"Error initializing dataset manager: {e}")
            return None
    
    def _init_session_state(self):
        """Initialize Streamlit session state"""
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'model_info' not in st.session_state:
            st.session_state.model_info = {}
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load the fish classification model"""
        try:
            with st.spinner("🔄 Loading model..."):
                # Create model config from settings
                model_config = ModelConfig(
                    architecture=self.settings.model.architecture,
                    batch_size=self.settings.training.batch_size,
                    num_classes=35  # Default for fish species, will be updated based on dataset
                )
                
                # Update num_classes if dataset is available
                if self.dataset_manager:
                    try:
                        dataset_info = self.dataset_manager.get_dataset_info()
                        if dataset_info and 'total_species' in dataset_info:
                            model_config.num_classes = dataset_info['total_species']
                    except Exception as e:
                        logger.warning(f"Could not get dataset info: {e}")
                
                self.model_manager = ModelManager(model_config)
                
                if model_path:
                    model = self.model_manager.load_model(model_path)
                else:
                    # Auto-detect best available model
                    model_files = [
                        "demo_fish_classifier.keras",
                        "best_fish_classifier_simplified.keras",
                        "unified_fish_classifier_final.keras"
                    ]
                    
                    model = None
                    for model_file in model_files:
                        if os.path.exists(model_file):
                            try:
                                model = self.model_manager.load_model(model_file)
                                break
                            except Exception as e:
                                logger.warning(f"Failed to load {model_file}: {e}")
                                continue
                
                if model is None:
                    st.error("❌ No trained model found!")
                    return False
                
                # Initialize predictor with the loaded model
                self.predictor = FishPredictor(model=model)
                
                # Update session state
                st.session_state.model_loaded = True
                st.session_state.model_info = {
                    'parameters': model.count_params() if hasattr(model, 'count_params') else 0,
                    'input_shape': model.input_shape if hasattr(model, 'input_shape') else None,
                    'classes': len(self.predictor.class_names)
                }
                
                logger.info("Model loaded successfully")
                logger.info(f"Session state model_loaded: {st.session_state.model_loaded}")
                
                # Force UI refresh
                st.success("✅ Model loaded successfully!")
                return True
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            st.error(f"❌ Error loading model: {e}")
            return False
    
    def render_sidebar(self):
        """Render the sidebar with model information and controls"""
        with st.sidebar:
            st.header("📊 Model Information")
            
            # Model loading section
            if st.button("🔄 Load/Reload Model"):
                success = self.load_model()
                if success:
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
            
            # Check if model was loaded by auto-load but predictor not initialized in UI
            if st.session_state.model_loaded and not self.predictor:
                logger.info("Model marked as loaded but predictor not initialized, attempting to reload...")
                self.load_model()
            
            # Display model status
            logger.info(f"Checking model status - model_loaded: {st.session_state.model_loaded}, predictor exists: {self.predictor is not None}")
            if st.session_state.model_loaded and self.predictor:
                st.success("✅ Model Ready")
                
                # Model details
                info = st.session_state.model_info
                st.metric("Parameters", f"{info.get('parameters', 0):,}")
                st.metric("Classes", info.get('classes', 0))
                
                if info.get('input_shape'):
                    st.info(f"Input Shape: {info['input_shape']}")
                
                # Species list
                with st.expander("🐟 Supported Species"):
                    if hasattr(self.predictor, 'class_names'):
                        for i, species in enumerate(self.predictor.class_names):
                            st.write(f"{i+1}. {species}")
                
                # Settings
                with st.expander("⚙️ Settings"):
                    confidence_threshold = st.slider(
                        "Confidence Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.1,
                        step=0.05
                    )
                    
                    show_top_k = st.selectbox(
                        "Show Top Predictions",
                        options=[3, 5, 10],
                        index=1
                    )
                    
                    return confidence_threshold, show_top_k
            else:
                st.warning("⚠️ Model not loaded")
                st.info("Click 'Load/Reload Model' to start")
                return 0.1, 5
    
    def render_upload_section(self):
        """Render the image upload section"""
        st.header("📷 Upload Fish Image")
        
        uploaded_file = st.file_uploader(
            "Choose a fish image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a fish for species identification"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                
                # Display image with improved layout
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(image, caption="Uploaded Image", width='stretch')
                
                with col2:
                    st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
                    st.metric("Dimensions", f"{image.size[0]} × {image.size[1]}")
                    st.metric("Mode", image.mode)
                    st.metric("Format", image.format or "Unknown")
                
                return image
                
            except Exception as e:
                st.error(f"❌ Error loading image: {e}")
                return None
        
        return None
    
    def render_prediction_section(self, image: Image.Image, confidence_threshold: float, top_k: int):
        """Render the prediction results section"""
        st.header("🔬 Prediction Results")
        
        if not st.session_state.model_loaded or not self.predictor:
            st.error("❌ Model not loaded. Please load a model first.")
            return
        
        try:
            with st.spinner("🤖 Analyzing image..."):
                # Make prediction
                results = self.predictor.predict_single(image, top_k=top_k)
                
                if not results:
                    st.error("❌ Failed to make prediction")
                    return
                
                # Filter by confidence threshold
                filtered_results = [
                    r for r in results 
                    if r['confidence'] >= confidence_threshold
                ]
                
                if not filtered_results:
                    st.warning(f"⚠️ No predictions above {confidence_threshold:.1%} confidence")
                    filtered_results = results[:3]  # Show top 3 anyway
                
                # Display top prediction
                top_pred = filtered_results[0]
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"## **{top_pred['species']}**")
                
                with col2:
                    st.metric("Confidence", f"{top_pred['confidence']:.1%}")
                
                with col3:
                    certainty = "High" if top_pred['confidence'] > 0.7 else "Medium" if top_pred['confidence'] > 0.3 else "Low"
                    st.metric("Certainty", certainty)
                
                # Predictions table
                st.markdown("### 📊 All Predictions")
                
                df = pd.DataFrame(filtered_results)
                df['confidence_pct'] = (df['confidence'] * 100).round(1)
                df['rank'] = range(1, len(df) + 1)
                
                # Reorder columns
                display_df = df[['rank', 'species', 'confidence_pct']].copy()
                display_df.columns = ['Rank', 'Species', 'Confidence (%)']
                
                st.dataframe(
                    display_df,
                    hide_index=True,
                    width='stretch'
                )
                
                # Visualization
                self._render_prediction_charts(filtered_results)
                
                # Technical details
                with st.expander("🔧 Technical Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Max Confidence", f"{top_pred['confidence']:.4f}")
                        
                        # Calculate entropy
                        confidences = [r['confidence'] for r in results]
                        entropy = -sum(c * np.log(c + 1e-10) for c in confidences if c > 0)
                        st.metric("Prediction Entropy", f"{entropy:.4f}")
                    
                    with col2:
                        st.metric("Total Classes", len(self.predictor.class_names))
                        st.metric("Predictions Shown", len(filtered_results))
                
                # Add to history
                self._add_to_history(image, top_pred, filtered_results)
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            st.error(f"❌ Prediction failed: {e}")
    
    def _render_prediction_charts(self, results: List[Dict[str, Any]]):
        """Render prediction visualization charts"""
        try:
            # Confidence bar chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Bar chart
            species = [r['species'] for r in results[:5]]
            confidences = [r['confidence'] for r in results[:5]]
            
            bars = ax1.barh(range(len(species)), confidences, color='skyblue')
            ax1.set_yticks(range(len(species)))
            ax1.set_yticklabels(species)
            ax1.set_xlabel('Confidence')
            ax1.set_title('Top 5 Predictions')
            ax1.set_xlim(0, 1)
            
            # Add value labels
            for i, (bar, conf) in enumerate(zip(bars, confidences)):
                ax1.text(conf + 0.01, i, f'{conf:.1%}', va='center')
            
            # Pie chart for top 3
            top_3 = results[:3]
            other_conf = 1 - sum(r['confidence'] for r in top_3)
            
            pie_labels = [r['species'] for r in top_3] + ['Others']
            pie_values = [r['confidence'] for r in top_3] + [other_conf]
            
            ax2.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Confidence Distribution')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Chart rendering error: {e}")
    
    def _add_to_history(self, image: Image.Image, top_pred: Dict[str, Any], all_results: List[Dict[str, Any]]):
        """Add prediction to history"""
        try:
            history_entry = {
                'timestamp': pd.Timestamp.now(),
                'top_species': top_pred['species'],
                'top_confidence': top_pred['confidence'],
                'num_predictions': len(all_results),
                'image_size': image.size
            }
            
            st.session_state.prediction_history.append(history_entry)
            
            # Keep only last 10 predictions
            if len(st.session_state.prediction_history) > 10:
                st.session_state.prediction_history = st.session_state.prediction_history[-10:]
                
        except Exception as e:
            logger.warning(f"History update error: {e}")
    
    def render_history_section(self):
        """Render prediction history"""
        if st.session_state.prediction_history:
            st.header("📈 Prediction History")
            
            history_df = pd.DataFrame(st.session_state.prediction_history)
            history_df['confidence_pct'] = (history_df['top_confidence'] * 100).round(1)
            
            display_df = history_df[[
                'timestamp', 'top_species', 'confidence_pct', 'num_predictions'
            ]].copy()
            display_df.columns = ['Time', 'Predicted Species', 'Confidence (%)', 'Total Predictions']
            
            st.dataframe(display_df, hide_index=True, width='stretch')
            
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
    
    def render_dataset_info(self):
        """Render dataset information"""
        with st.expander("📊 Dataset Information"):
            try:
                if self.dataset_manager is None:
                    st.warning("No dataset loaded. Some features may be limited.")
                    st.info("Available datasets: FishImgDataset, UnifiedFishDataset, DemoFishDataset")
                    return
                
                # Load dataset info
                dataset_info = self.dataset_manager.get_dataset_info()
                
                if dataset_info:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Images", dataset_info.get('total_images', 0))
                    
                    with col2:
                        st.metric("Total Species", dataset_info.get('total_species', 0))
                    
                    with col3:
                        st.metric("Datasets", dataset_info.get('num_datasets', 0))
                    
                    # Species distribution
                    if 'species_distribution' in dataset_info:
                        st.markdown("**Species Distribution:**")
                        species_df = pd.DataFrame.from_dict(
                            dataset_info['species_distribution'], 
                            orient='index', 
                            columns=['Count']
                        )
                        species_df = species_df.sort_values('Count', ascending=False)
                        st.dataframe(species_df, width='stretch')
                
            except Exception as e:
                st.warning(f"Could not load dataset info: {e}")
    
    def run(self):
        """Run the Streamlit application"""
        # Custom CSS
        st.markdown("""
        <style>
        .main {
            padding-top: 1rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 0.5rem;
            border-radius: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.title("🐟 Fish Species Classifier")
        st.markdown("*Powered by Deep Learning and Computer Vision*")
        
        # Auto-load model on startup
        if not st.session_state.model_loaded:
            self.load_model()
        
        # Sidebar
        confidence_threshold, top_k = self.render_sidebar()
        
        # Main content
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = self.render_upload_section()
        
        with col2:
            if image is not None:
                self.render_prediction_section(image, confidence_threshold, top_k)
            else:
                st.info("📤 Upload a fish image to see AI predictions")
        
        # Additional sections
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self.render_history_section()
        
        with col2:
            self.render_dataset_info()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
        🌊 OceanNex Fish Species Detection Model | Built with Streamlit & TensorFlow
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application entry point"""
    try:
        app = FishClassifierUI()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check the console for detailed error information.")

if __name__ == "__main__":
    main()