import os
import requests

def test_kaggle_token():
    token = "KGAT_be80a85e16188167b396e8f05a84c682"
    url = "https://www.kaggle.com/api/v1/hello"
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    print(f"🔍 Testing Kaggle token validity at {url}...")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        print("✅ Token is valid!")
        print(f"Response: {response.json()}")
    else:
        print(f"❌ Token validation failed with status {response.status_code}")
        print(f"Reason: {response.text}")

if __name__ == "__main__":
    test_kaggle_token()
