#!/usr/bin/env python3
"""
Fish Species Classifier - Main Application Launcher
==================================================

This script launches the Streamlit web application for fish species classification.
It provides a user-friendly interface for uploading images and getting predictions.

Usage:
    python run_streamlit.py

Requirements:
    - All dependencies installed (run: pip install -r requirements.txt)
    - TensorFlow model files in place
    - Dataset available (optional, for training)
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit application"""
    
    # Get the directory containing this script
    project_root = Path(__file__).parent
    
    # Path to the main Streamlit app
    app_path = project_root / "src" / "ui" / "streamlit_app.py"
    
    # Check if the app file exists
    if not app_path.exists():
        print(f"❌ Error: Streamlit app not found at {app_path}")
        print("Please ensure the project structure is correct.")
        sys.exit(1)
    
    # Print startup message
    print("🐟 Starting Fish Species Classifier Web Application...")
    print("=" * 60)
    print(f"📂 Project root: {project_root}")
    print(f"🖥️  App location: {app_path}")
    print("=" * 60)
    
    # Launch Streamlit
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.headless", "false",
            "--server.runOnSave", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        print("🚀 Launching Streamlit application...")
        print("💡 The app will open in your default web browser")
        print("⏹️  Press Ctrl+C to stop the application")
        print("=" * 60)
        
        # Change to project directory
        os.chdir(project_root)
        
        # Run the command
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")
        print("💡 Make sure Streamlit is installed: pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
