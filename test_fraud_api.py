import requests
import json
import os

# Define API endpoint (fallback to default)
API_URL = os.getenv("API_URL", "https://fraud-api-716102832289.us-central1.run.app/predict")

# Test cases: valid and invalid data
TEST_CASES = [
    {"features": [0.5, 1.2, 3.4, 4.7, 2.3, 0.9, 5.1, 1.8]},  # Valid case
    {"features": [1.1, 0.4, 2.9, 3.6, 4.5, 1.2, 3.7, 2.8]},  # Another valid case
    {"features": "invalid"},  # Invalid format (string instead of list)
    {"features": [0.5, 1.2]},  # Too few features
    {},  # Missing 'features' key
    {"wrong_key": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}  # Wrong key
]

def test_api():
    """Function to send test cases to the fraud detection API and log results."""
    print(f"🔍 Running API tests on: {API_URL}\n")
    
    for i, test_case in enumerate(TEST_CASES):
        print(f"📌 Test Case {i+1}: {test_case}")
        
        try:
            response = requests.post(API_URL, json=test_case, headers={"Content-Type": "application/json"})
            result = response.json()
            
            print("✅ Response:", json.dumps(result, indent=4))
            print("🔹 Status Code:", response.status_code)
            
            if response.status_code != 200:
                print("🚨 Warning: Non-200 response received.")

        except requests.exceptions.RequestException as e:
            print(f"❌ Request Error: {e}")
        except json.JSONDecodeError:
            print("❌ Invalid JSON response received.")

        print("-" * 50)

if __name__ == "__main__":
    test_api()
