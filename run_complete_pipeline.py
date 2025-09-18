"""
Fish Species Classification - Complete Pipeline
==============================================

This script runs the complete fish species classification pipeline:
1. Dataset analysis and visualization
2. CNN model training
3. Model evaluation
4. Prediction demonstration

Run this script to execute the entire pipeline automatically.
"""

import os
import time
import sys
from datetime import datetime

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"🐟 {title}")
    print("="*60)

def print_step(step_num, title, description):
    """Print step information"""
    print(f"\n📍 STEP {step_num}: {title}")
    print(f"   {description}")
    print("-" * 50)

def run_pipeline():
    """Run the complete fish species classification pipeline"""
    
    print_header("FISH SPECIES CLASSIFICATION PIPELINE")
    print("This pipeline will analyze your fish dataset and train a CNN model")
    print("for species classification.")
    print()
    print("⏱️  Estimated time: 2-4 hours (depending on hardware)")
    print("💾 Required space: ~5GB for models and results")
    print("🖥️  Hardware: GPU recommended for faster training")
    print()
    
    # Check dataset exists
    dataset_path = "FishImgDataset"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please ensure the FishImgDataset folder exists in the current directory.")
        return False
    
    print(f"✅ Dataset found: {dataset_path}")
    
    # Pipeline steps
    steps = [
        {
            "script": "dataset_analysis.py",
            "title": "Dataset Analysis",
            "description": "Analyze dataset structure, distribution, and create visualizations",
            "estimated_time": "2-5 minutes"
        },
        {
            "script": "train_fish_cnn.py", 
            "title": "Model Training",
            "description": "Train CNN model with data augmentation and callbacks",
            "estimated_time": "1-3 hours"
        },
        {
            "script": "predict_fish_species.py",
            "title": "Prediction Demo",
            "description": "Demonstrate prediction capabilities with sample images",
            "estimated_time": "1-2 minutes"
        }
    ]
    
    start_time = datetime.now()
    completed_steps = []
    
    try:
        for i, step in enumerate(steps, 1):
            print_step(i, step["title"], step["description"])
            print(f"⏱️  Estimated time: {step['estimated_time']}")
            print(f"🔄 Running: {step['script']}")
            
            step_start = time.time()
            
            # Run the script
            exit_code = os.system(f'python {step["script"]}')
            
            step_duration = time.time() - step_start
            
            if exit_code == 0:
                print(f"✅ Step {i} completed successfully")
                print(f"⏱️  Duration: {step_duration/60:.1f} minutes")
                completed_steps.append(step["title"])
            else:
                print(f"❌ Step {i} failed with exit code: {exit_code}")
                print(f"💡 Try running '{step['script']}' manually to see detailed error messages")
                break
        
        # Final summary
        total_duration = datetime.now() - start_time
        
        print_header("PIPELINE COMPLETION SUMMARY")
        print(f"🕐 Total time: {total_duration}")
        print(f"✅ Completed steps: {len(completed_steps)}/{len(steps)}")
        
        for i, step_title in enumerate(completed_steps, 1):
            print(f"   {i}. {step_title}")
        
        if len(completed_steps) == len(steps):
            print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
            print("\n📁 Generated Files:")
            
            expected_files = [
                "fish_dataset_comprehensive_analysis.png",
                "fish_dataset_analysis_report.json",
                "fish_species_cnn_final.h5",
                "fish_cnn_training_history.png", 
                "fish_cnn_confusion_matrix.png",
                "fish_cnn_training_results.json",
                "training_log.csv"
            ]
            
            for file in expected_files:
                if os.path.exists(file):
                    print(f"   ✅ {file}")
                else:
                    print(f"   ❌ {file} (missing)")
            
            print("\n🚀 Your fish species classifier is ready!")
            print("\n📋 Next Steps:")
            print("   • Review the training results and visualizations")
            print("   • Test the model with your own fish images")
            print("   • Use the prediction API for integration")
            print("   • Consider further model improvements")
            
        else:
            print(f"\n⚠️  Pipeline incomplete. {len(steps) - len(completed_steps)} steps failed.")
            print("Please check error messages and try running individual scripts.")
        
        return len(completed_steps) == len(steps)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        print(f"✅ Completed steps: {len(completed_steps)}")
        return False
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        return False

def check_requirements():
    """Check if required packages are installed"""
    print_header("CHECKING REQUIREMENTS")
    
    required_packages = [
        'tensorflow',
        'numpy', 
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages:")
        if 'PIL' in missing_packages:
            missing_packages = [p if p != 'PIL' else 'pillow' for p in missing_packages]
        print(f"   python -m pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✅ All required packages are installed!")
    return True

def main():
    """Main function"""
    print("🚀 FISH SPECIES CLASSIFICATION - COMPLETE PIPELINE")
    print("=" * 60)
    print("Welcome to the automated fish species classification pipeline!")
    print()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please install missing packages.")
        return
    
    # Confirm before starting
    print("\n🤔 Ready to start the complete pipeline?")
    print("This will:")
    print("   • Analyze your fish dataset")
    print("   • Train a CNN model (may take 1-3 hours)")
    print("   • Generate comprehensive results and visualizations")
    print()
    
    response = input("Continue? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        success = run_pipeline()
        
        if success:
            print("\n🎊 Congratulations! Your fish species classifier is ready to use.")
        else:
            print("\n📞 If you need help, check the individual script outputs for detailed error messages.")
    else:
        print("\n👋 Pipeline cancelled. You can run individual scripts manually:")
        print("   • python dataset_analysis.py")
        print("   • python train_fish_cnn.py") 
        print("   • python predict_fish_species.py")

if __name__ == "__main__":
    main()