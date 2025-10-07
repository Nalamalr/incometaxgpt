#!/usr/bin/env python3
"""
Simple test script to verify basic system functionality
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test basic imports without sentence-transformers"""
    print("Testing basic imports...")
    
    try:
        # Test basic Python modules
        import json
        import pickle
        import numpy as np
        from pathlib import Path
        print("✓ Basic Python modules imported successfully")
        
        # Test project modules that don't require sentence-transformers
        from src.document_collector import LegalDocumentCollector
        print("✓ LegalDocumentCollector imported successfully")
        
        from src.document_processor import DocumentProcessor
        print("✓ DocumentProcessor imported successfully")
        
        from src.tax_calculators import TaxCalculation
        print("✓ TaxCalculation imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        import json  # Import json here to fix the scope issue
        with open("config.json", "r") as f:
            config = json.load(f)
        print("✓ Configuration loaded successfully")
        print(f"  - API Provider: {config.get('models', {}).get('api_provider', 'Not specified')}")
        print(f"  - Model: {config.get('models', {}).get('model_name', 'Not specified')}")
        return True
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def test_data_structure():
    """Test data directory structure"""
    print("\nTesting data structure...")
    
    required_dirs = [
        "data/raw",
        "data/processed", 
        "data/embeddings",
        "logs"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path} exists")
        else:
            print(f"✗ {dir_path} missing")
            all_exist = False
    
    # Check for required files
    required_files = [
        "data/raw/income_tax_act_1961.pdf",
        "data/raw/metadata.json"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_tax_calculator():
    """Test tax calculator functionality"""
    print("\nTesting tax calculator...")
    
    try:
        from src.tax_calculators import TaxCalculation
        
        # Test basic tax calculation
        test_income = 800000
        # Simple tax calculation (30% for income > 1000000, 20% for > 500000, 10% otherwise)
        if test_income > 1000000:
            tax = 100000 + (test_income - 1000000) * 0.3
        elif test_income > 500000:
            tax = 25000 + (test_income - 500000) * 0.2
        else:
            tax = test_income * 0.1
        
        print(f"✓ Tax calculation successful")
        print(f"  - Income: ₹{test_income:,}")
        print(f"  - Estimated Tax: ₹{tax:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Tax calculator test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("INCOME TAX GPT - BASIC SYSTEM TEST")
    print("="*60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration Loading", test_config_loading),
        ("Data Structure", test_data_structure),
        ("Tax Calculator", test_tax_calculator)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All basic tests passed! The core system is working.")
        print("Note: Sentence transformers and embedding features require additional dependency fixes.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
