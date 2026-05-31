#!/usr/bin/env python3
"""
Test script for Fake Reviews Detection Project
This script tests all major components to ensure they work correctly.
"""

import os
import sys
import subprocess
import requests
import time
import json

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    try:
        import torch
        import transformers
        import pandas as pd
        import numpy as np
        import sklearn
        import matplotlib
        import seaborn
        import flask
        import joblib
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_model_loading():
    """Test if the model can be loaded"""
    print("\nTesting model loading...")
    try:
        from new import DeepReviewClassifier
        classifier = DeepReviewClassifier()
        print("✓ Model initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Model loading error: {e}")
        return False

def test_text_processing():
    """Test text preprocessing functionality"""
    print("\nTesting text processing...")
    try:
        from new import clean_text
        test_text = "This is a TEST review with some CAPS and punctuation!!!"
        cleaned = clean_text(test_text)
        print(f"✓ Text cleaning works: '{test_text}' -> '{cleaned}'")
        return True
    except Exception as e:
        print(f"✗ Text processing error: {e}")
        return False

def test_prediction():
    """Test model prediction functionality"""
    print("\nTesting prediction functionality...")
    try:
        from new import DeepReviewClassifier
        classifier = DeepReviewClassifier()
        
        test_reviews = [
            "This product is amazing! I love it so much!",
            "The quality is good and it works as expected."
        ]
        
        predictions = classifier.predict(test_reviews)
        probabilities = classifier.predict_proba(test_reviews)
        
        print(f"✓ Predictions: {predictions}")
        print(f"✓ Probabilities shape: {probabilities.shape}")
        return True
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        return False

def test_web_server():
    """Test if the web server can start and respond"""
    print("\nTesting web server...")
    try:
        # Start the server in a subprocess
        process = subprocess.Popen([sys.executable, 'web2.py'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Test if server is responding
        try:
            response = requests.get('http://localhost:5000', timeout=5)
            if response.status_code == 200:
                print("✓ Web server is responding")
                process.terminate()
                return True
            else:
                print(f"✗ Web server returned status code: {response.status_code}")
                process.terminate()
                return False
        except requests.exceptions.RequestException:
            print("✗ Web server is not responding")
            process.terminate()
            return False
            
    except Exception as e:
        print(f"✗ Web server error: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("\nTesting API endpoints...")
    try:
        # Test analyze endpoint
        test_review = {"review": "This is a test review for API testing."}
        response = requests.post('http://localhost:5000/analyze', 
                               json=test_review, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Analyze endpoint works: {result.get('sentiment', 'Unknown')}")
        else:
            print(f"✗ Analyze endpoint failed: {response.status_code}")
            return False
        
        # Test metrics endpoint
        response = requests.get('http://localhost:5000/metrics', timeout=5)
        if response.status_code == 200:
            print("✓ Metrics endpoint works")
        else:
            print(f"✗ Metrics endpoint failed: {response.status_code}")
            return False
        
        # Test feature importance endpoint
        response = requests.get('http://localhost:5000/feature_importance', timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Feature importance endpoint works: {len(result.get('features', []))} features")
        else:
            print(f"✗ Feature importance endpoint failed: {response.status_code}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ API testing error: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    required_files = [
        'train.py',
        'web2.py', 
        'new.py',
        'evaluate.py',
        'requirements.txt',
        'static/index.html'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files exist")
        return True

def test_directories():
    """Test if required directories exist or can be created"""
    print("\nTesting directories...")
    required_dirs = ['static', 'uploads']
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
                print(f"✓ Created directory: {dir_name}")
            except Exception as e:
                print(f"✗ Failed to create directory {dir_name}: {e}")
                return False
        else:
            print(f"✓ Directory exists: {dir_name}")
    
    return True

def main():
    """Run all tests"""
    print("="*60)
    print("FAKE REVIEWS DETECTION PROJECT - TEST SUITE")
    print("="*60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Directories", test_directories),
        ("Imports", test_imports),
        ("Model Loading", test_model_loading),
        ("Text Processing", test_text_processing),
        ("Prediction", test_prediction),
        ("Web Server", test_web_server),
        ("API Endpoints", test_api_endpoints)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total} tests")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your project is ready to run.")
        print("\nTo start the application:")
        print("1. python train.py (if you haven't trained the model)")
        print("2. python web2.py")
        print("3. Open http://localhost:5000 in your browser")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
