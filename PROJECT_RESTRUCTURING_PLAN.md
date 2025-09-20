# Fish Species Classifier - Repository Restructuring Plan

## Current State Analysis

### Core Application Files (KEEP)
- `src/` - Main source code directory
- `config/` - Configuration files
- `demo_fish_classifier.keras` - Working trained model
- `class_mapping.json` - Essential class mappings
- `run_streamlit.py` - Main application launcher
- `requirements.txt` - Core dependencies
- `README.md` - Documentation

### Dataset Analysis (SELECTIVE KEEP)
- **FishImgDataset/** (31 species, ~13K images) - PRIMARY DATASET - KEEP
- **DemoFishDataset/** (35 species, 729 images) - Used for demo model - KEEP
- **UnifiedFishDataset/** (8,967 images) - Large unified dataset - KEEP (Optional)
- **archive/** - Old dataset - REMOVE
- **fishesdataser2/** - Empty directory - REMOVE

### Files/Directories to REMOVE

#### Redundant Analysis Files
- `comprehensive_dataset_analysis.png`
- `comprehensive_dataset_analysis_report.json`
- `comprehensive_dataset_analysis_report.txt`
- `comprehensive_dataset_summary.json`
- `comprehensive_model_evaluation.png`
- `comprehensive_model_evaluation_report.json`
- `comprehensive_model_evaluation_report.txt`
- `fish_dataset_analysis_report.json`
- `fish_dataset_comprehensive_analysis.png`
- `unified_dataset_analysis.png`
- `unified_dataset_info.json`
- `unified_dataset_report.txt`

#### Redundant Planning/Documentation Files
- `CLEANUP_ANALYSIS.md`
- `CLEANUP_COMPLETED.md`
- `REFACTORING_PLAN.md`
- `REMOVAL_PLAN.md`

#### Empty/Unused Directories
- `performance_analysis/` (empty)
- `model_comparison_results/` (empty)
- `fishesdataser2/` (empty)
- `archive/` (old data)

#### Temporary/Log Files
- `demo_training_log.csv`
- `demo_training_results.png`
- `training_log_simplified.csv`
- `training_progress_monitor.png`
- `demo_model_results.json`
- `validation_results.json`

#### Development Files (Optional Keep)
- `requirements-dev.txt` - Development dependencies (OPTIONAL KEEP)
- `validate_functionality.py` - Validation script (OPTIONAL KEEP)
- `train_demo_model.py` - Training script (OPTIONAL KEEP)

#### Test/Experiment Directories
- `experiments/` - Contains test experiments (REVIEW)
- `tests/` - Unit tests (KEEP for development)
- `scripts/` - Utility scripts (KEEP if needed)

### Recommended Final Structure

```
OceanNex-Species-detection-model/
├── src/                          # Core application code
├── config/                       # Configuration files
├── DemoFishDataset/             # Demo dataset (small)
├── FishImgDataset/              # Primary dataset
├── tests/                       # Unit tests
├── scripts/                     # Utility scripts
├── demo_fish_classifier.keras   # Trained model
├── class_mapping.json          # Class mappings
├── run_streamlit.py            # Main launcher
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

### Size Reduction Estimate
- Remove ~20+ analysis/report files
- Remove 3 empty directories
- Remove multiple log/temporary files
- Estimated reduction: 60-70% fewer files while maintaining functionality

### Actions Required
1. Backup important datasets (FishImgDataset, DemoFishDataset)
2. Remove redundant analysis files
3. Remove empty directories
4. Remove old logs and temporary files
5. Update README.md to reflect new structure
6. Test application functionality after cleanup