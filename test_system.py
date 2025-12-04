#!/usr/bin/env python3
"""
Test script for the Skin Disease Detection System

This script validates key components of the system without running
the full training pipeline.

Author: Elizabeth Chacko
"""

import sys
import json
import os


def test_notebook_structure():
    """Test if the notebook has valid structure."""
    print("Testing notebook structure...")
    
    notebook_path = "Skin_Disease_Detection_System.ipynb"
    
    if not os.path.exists(notebook_path):
        print("  ❌ Notebook file not found")
        return False
    
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        # Check basic structure
        assert 'cells' in notebook, "No cells found in notebook"
        assert 'metadata' in notebook, "No metadata found in notebook"
        assert len(notebook['cells']) > 0, "Notebook has no cells"
        
        print(f"  ✓ Notebook structure valid")
        print(f"  ✓ Found {len(notebook['cells'])} cells")
        
        # Count cell types
        code_cells = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'code')
        markdown_cells = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'markdown')
        
        print(f"  ✓ Code cells: {code_cells}")
        print(f"  ✓ Markdown cells: {markdown_cells}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_healthcare_module():
    """Test the healthcare recommendations module."""
    print("\nTesting healthcare_recommendations module...")
    
    try:
        import healthcare_recommendations as hr
        
        # Test disease list
        diseases = hr.get_all_diseases()
        assert len(diseases) > 0, "No diseases found"
        print(f"  ✓ Found {len(diseases)} disease classes")
        
        # Test each disease has recommendations
        for disease in diseases:
            rec = hr.get_recommendation(disease)
            assert rec is not None, f"No recommendations for {disease}"
            assert 'description' in rec, f"No description for {disease}"
            assert 'recommendations' in rec, f"No recommendations for {disease}"
            assert 'severity' in rec, f"No severity for {disease}"
        
        print(f"  ✓ All diseases have valid recommendations")
        
        # Test severity levels
        for disease in diseases:
            severity = hr.get_severity_level(disease)
            assert severity is not None, f"No severity level for {disease}"
        
        print(f"  ✓ All severity levels valid")
        
        # Test emergency detection
        melanoma_emergency = hr.is_emergency('Melanoma')
        assert melanoma_emergency == True, "Melanoma should be marked as emergency"
        
        normal_emergency = hr.is_emergency('Normal')
        assert normal_emergency == False, "Normal should not be emergency"
        
        print(f"  ✓ Emergency detection working")
        
        # Test formatting
        formatted = hr.format_recommendation('Acne')
        assert len(formatted) > 0, "Formatted recommendation is empty"
        assert 'HEALTHCARE RECOMMENDATIONS' in formatted
        
        print(f"  ✓ Recommendation formatting working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_requirements_file():
    """Test if requirements.txt exists and is valid."""
    print("\nTesting requirements.txt...")
    
    if not os.path.exists('requirements.txt'):
        print("  ❌ requirements.txt not found")
        return False
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.readlines()
        
        # Check for essential packages
        required_packages = ['tensorflow', 'numpy', 'matplotlib', 'opencv', 'pillow', 'scikit-learn']
        
        req_text = '\n'.join(requirements).lower()
        
        for package in required_packages:
            if package not in req_text:
                print(f"  ⚠️  Warning: {package} not found in requirements")
        
        print(f"  ✓ requirements.txt exists with {len(requirements)} packages")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_documentation():
    """Test if documentation files exist."""
    print("\nTesting documentation files...")
    
    docs = {
        'README.md': 'Main documentation',
        'QUICKSTART.md': 'Quick start guide',
        'DEPLOYMENT.md': 'Deployment guide'
    }
    
    all_exist = True
    for doc_file, description in docs.items():
        if os.path.exists(doc_file):
            size = os.path.getsize(doc_file)
            print(f"  ✓ {doc_file} ({description}): {size} bytes")
        else:
            print(f"  ❌ {doc_file} not found")
            all_exist = False
    
    return all_exist


def test_disease_classes():
    """Test disease class consistency."""
    print("\nTesting disease class consistency...")
    
    try:
        import healthcare_recommendations as hr
        
        # Expected classes
        expected_classes = ['Acne', 'Dermatitis', 'Eczema', 'Melanoma', 
                          'Normal', 'Psoriasis', 'Warts']
        
        actual_classes = hr.get_all_diseases()
        
        # Check if classes match
        if set(expected_classes) == set(actual_classes):
            print(f"  ✓ All expected disease classes present")
            return True
        else:
            missing = set(expected_classes) - set(actual_classes)
            extra = set(actual_classes) - set(expected_classes)
            if missing:
                print(f"  ⚠️  Missing classes: {missing}")
            if extra:
                print(f"  ⚠️  Extra classes: {extra}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("SKIN DISEASE DETECTION SYSTEM - TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Notebook Structure", test_notebook_structure),
        ("Healthcare Module", test_healthcare_module),
        ("Requirements File", test_requirements_file),
        ("Documentation", test_documentation),
        ("Disease Classes", test_disease_classes),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
