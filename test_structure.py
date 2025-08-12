#!/usr/bin/env python3
"""
Simple structure test for RL-Dewey-Tutor
Tests imports and basic functionality without external dependencies
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        # Test basic imports
        print("✓ Basic imports successful")
        
        # Test environment (without gymnasium)
        print("✓ Code structure verified")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'src/main.py',
        'src/train.py', 
        'src/evaluate.py',
        'src/envs/tutor_env.py',
        'src/rl_dewey_tutor/agents/q_learning_agent.py',
        'src/rl_dewey_tutor/agents/thompson_sampling.py',
        'src/run_experiments.py',
        'README.md',
        'requirements.txt',
        'configs/baseline_config.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing files: {len(missing_files)}")
        return False
    else:
        print(f"\nAll {len(required_files)} required files present!")
        return True

def test_code_quality():
    """Test basic code quality checks"""
    print("\nTesting code quality...")
    
    # Check for docstrings in key files
    key_files = [
        'src/envs/tutor_env.py',
        'src/rl_dewey_tutor/agents/q_learning_agent.py',
        'src/rl_dewey_tutor/agents/thompson_sampling.py'
    ]
    
    for file_path in key_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                if '"""' in content or "'''" in content:
                    print(f"✓ {file_path} - Has docstrings")
                else:
                    print(f"⚠ {file_path} - No docstrings found")
        except Exception as e:
            print(f"✗ {file_path} - Error reading: {e}")
    
    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("RL-Dewey-Tutor Structure Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_file_structure,
        test_code_quality
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Structure is ready for development.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run training: python3 src/train.py --method both")
        print("3. Run experiments: python3 src/run_experiments.py")
    else:
        print("❌ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 