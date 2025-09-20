"""
Test Execution Script
====================

Automated test execution with comprehensive reporting.
"""

import unittest
import sys
import time
import json
from pathlib import Path
from io import StringIO
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def run_test_suite():
    """Run complete test suite with detailed reporting"""
    
    print("🚀 Starting Comprehensive Test Suite")
    print("=" * 50)
    
    # Test discovery
    test_dir = Path(__file__).parent
    loader = unittest.TestLoader()
    
    # Collect all tests
    print("📁 Discovering tests...")
    unit_tests = loader.discover(str(test_dir / "unit"), pattern="test_*.py")
    integration_tests = loader.discover(str(test_dir / "integration"), pattern="test_*.py")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    test_suite.addTests(unit_tests)
    test_suite.addTests(integration_tests)
    
    print(f"✅ Found {test_suite.countTestCases()} total tests")
    print()
    
    # Run tests with detailed output
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2,
        buffer=True
    )
    
    print("🔬 Running tests...")
    start_time = time.time()
    
    try:
        result = runner.run(test_suite)
        end_time = time.time()
        
        # Get test output
        test_output = stream.getvalue()
        
        # Print results
        print("📊 Test Results Summary")
        print("-" * 30)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
        print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print()
        
        # Show failures and errors
        if result.failures:
            print("❌ Failures:")
            for test, error in result.failures:
                print(f"  - {test}: {error}")
                print()
        
        if result.errors:
            print("🚨 Errors:")
            for test, error in result.errors:
                print(f"  - {test}: {error}")
                print()
        
        # Show detailed output if requested
        if "--verbose" in sys.argv:
            print("📝 Detailed Test Output:")
            print(test_output)
        
        # Generate test report
        generate_test_report(result, end_time - start_time)
        
        # Return success status
        return len(result.failures) == 0 and len(result.errors) == 0
        
    except Exception as e:
        print(f"🚨 Test execution failed: {e}")
        traceback.print_exc()
        return False

def generate_test_report(result, execution_time):
    """Generate JSON test report"""
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time": execution_time,
        "total_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100,
        "details": {
            "failures": [{"test": str(test), "error": error} for test, error in result.failures],
            "errors": [{"test": str(test), "error": error} for test, error in result.errors]
        }
    }
    
    # Save report
    report_file = Path(__file__).parent / "test_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Test report saved to: {report_file}")

def run_specific_test_category(category):
    """Run specific test category (unit or integration)"""
    
    print(f"🎯 Running {category} tests only")
    print("=" * 30)
    
    test_dir = Path(__file__).parent / category
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return False
    
    loader = unittest.TestLoader()
    tests = loader.discover(str(test_dir), pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(tests)
    
    return len(result.failures) == 0 and len(result.errors) == 0

def validate_test_environment():
    """Validate test environment is properly set up"""
    
    print("🔍 Validating test environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required modules
    required_modules = [
        'numpy', 'pandas', 'tensorflow', 'scikit-learn', 
        'matplotlib', 'streamlit', 'pillow'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module} (missing)")
    
    if missing_modules:
        print(f"\n🚨 Missing modules: {', '.join(missing_modules)}")
        print("Please install missing dependencies before running tests")
        return False
    
    # Check test structure
    test_dir = Path(__file__).parent
    required_dirs = ['unit', 'integration', 'fixtures']
    
    for dir_name in required_dirs:
        dir_path = test_dir / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ directory")
        else:
            print(f"❌ {dir_name}/ directory (missing)")
            return False
    
    print("✅ Test environment validation complete")
    print()
    return True

def main():
    """Main test execution function"""
    
    # Parse command line arguments
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Test Execution Script")
        print("====================")
        print()
        print("Usage:")
        print("  python run_tests.py                 # Run all tests")
        print("  python run_tests.py --unit         # Run unit tests only")
        print("  python run_tests.py --integration  # Run integration tests only")
        print("  python run_tests.py --validate     # Validate environment only")
        print("  python run_tests.py --verbose      # Verbose output")
        print("  python run_tests.py --help         # Show this help")
        return
    
    # Validate environment first
    if not validate_test_environment():
        sys.exit(1)
    
    # Run specific test category
    if "--unit" in sys.argv:
        success = run_specific_test_category("unit")
    elif "--integration" in sys.argv:
        success = run_specific_test_category("integration")
    elif "--validate" in sys.argv:
        print("✅ Environment validation complete")
        return
    else:
        # Run full test suite
        success = run_test_suite()
    
    # Exit with appropriate code
    if success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()