"""
Test Suite for Fish Species Classification Project
================================================

Comprehensive testing framework for the refactored fish species classification system.
Includes unit tests, integration tests, and validation tests.
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
config_path = project_root / 'config'

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    
if str(config_path) not in sys.path:
    sys.path.insert(0, str(config_path))

# Test configuration
TEST_DATA_DIR = project_root / 'tests' / 'fixtures'
TEST_RESULTS_DIR = project_root / 'tests' / 'results'

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_RESULTS_DIR.mkdir(exist_ok=True)