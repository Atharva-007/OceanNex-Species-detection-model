"""
Quick Streamlit Launcher
========================
Launch the fish classifier UI directly
"""

import subprocess
import os

def launch_streamlit():
    """Launch Streamlit app with no email prompt"""
    os.environ['STREAMLIT_SERVER_COLLECT_USAGE_STATS'] = 'false'
    
    # Set working directory
    os.chdir(r"d:\Future Projects\OceanNex-Species-detection-model")
    
    # Launch streamlit
    cmd = [
        r"C:\Users\athar\anaconda3\python.exe",
        "-m", "streamlit", "run",
        "fish_classifier_ui_fixed.py",
        "--server.headless=true",
        "--server.enableCORS=false"
    ]
    
    print("🚀 Launching Fish Classifier UI...")
    print("🌐 Opening browser at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    
    subprocess.run(cmd)

if __name__ == "__main__":
    launch_streamlit()