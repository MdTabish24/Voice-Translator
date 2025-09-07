#!/usr/bin/env python3
"""
Simple test script for Real-Time Translator API
Run this to verify the backend is working correctly
"""

import requests
import json
import sys

def test_backend():
    """Test all backend endpoints"""
    base_url = "http://127.0.0.1:5000"
    
    print("🧪 Testing Real-Time Translator Backend")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("   Make sure the backend is running: python app.py")
        return False
    
    # Test 2: Languages endpoint
    print("\n2. Testing languages endpoint...")
    try:
        response = requests.get(f"{base_url}/languages", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Languages endpoint passed")
            print(f"   Found {data['count']} supported languages")
        else:
            print(f"❌ Languages endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Languages endpoint failed: {e}")
    
    # Test 3: Simple translation
    print("\n3. Testing simple translation...")
    try:
        test_data = {
            "text": "Hello world",
            "target_language": "es"
        }
        response = requests.post(
            f"{base_url}/translate", 
            json=test_data, 
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print("✅ Simple translation passed")
            print(f"   '{data['original_text']}' -> '{data['translated_text']}'")
            print(f"   Detected language: {data['source_language']}")
        else:
            print(f"❌ Simple translation failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Simple translation failed: {e}")
    
    # Test 4: Long text translation
    print("\n4. Testing long text translation...")
    try:
        long_text = "This is a longer text that should be split into chunks. " * 20
        test_data = {
            "text": long_text,
            "target_language": "fr"
        }
        response = requests.post(
            f"{base_url}/translate-long", 
            json=test_data, 
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            print("✅ Long text translation passed")
            print(f"   Processed {data['chunks_processed']} chunks")
            print(f"   Success rate: {data.get('success_rate', 1.0):.2%}")
        else:
            print(f"❌ Long text translation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Long text translation failed: {e}")
    
    # Test 5: Error handling
    print("\n5. Testing error handling...")
    try:
        # Test with invalid data
        response = requests.post(f"{base_url}/translate", json={}, timeout=5)
        if response.status_code == 400:
            print("✅ Error handling passed (correctly rejected invalid data)")
        else:
            print(f"⚠️  Error handling unexpected: {response.status_code}")
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Backend testing completed!")
    print("\nNext steps:")
    print("1. Open your browser")
    print("2. Navigate to: http://127.0.0.1:5000/")
    print("3. Test Voice Mode and Camera Mode")
    print("4. Grant microphone and camera permissions when prompted")
    
    return True

if __name__ == "__main__":
    success = test_backend()
    sys.exit(0 if success else 1)