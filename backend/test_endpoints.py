import requests
import base64
import json
from google.cloud import billing
from google.cloud import resourcemanager_v3

def verify_billing():
    """Verify billing is enabled for the project."""
    print("\nVerifying Billing Setup:")
    try:
        # Get project billing info
        billing_client = billing.CloudBillingClient()
        project_client = resourcemanager_v3.ProjectsClient()
        
        project = project_client.get_project(name=f"projects/ethica-vision-2024")
        project_billing_info = billing_client.get_project_billing_info(
            name=f"projects/{project.project_id}"
        )
        
        if project_billing_info.billing_enabled:
            print("✅ Billing is enabled")
        else:
            print("❌ Billing is not enabled")
            print("Please visit: https://console.cloud.google.com/billing/enable?project=830766747753")
    except Exception as e:
        print(f"❌ Billing verification error: {str(e)}")

def test_endpoints():
    base_url = "http://127.0.0.1:5000"
    
    # Verify billing first
    verify_billing()
    
    # Test CORS
    print("\n1. Testing CORS:")
    try:
        response = requests.get(f"{base_url}/test-cors")
        print(f"Status: {response.status_code}")
        try:
            print(f"Response: {response.json()}")
        except requests.exceptions.JSONDecodeError:
            print(f"Raw Response: {response.text}")
    except Exception as e:
        print(f"❌ CORS Test Error: {str(e)}")
    
    # Test ingredient extraction
    print("\n2. Testing Ingredient Extraction:")
    try:
        with open("test_image.jpg", "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
        response = requests.post(
            f"{base_url}/extract-ingredients",
            json={"imageBase64": image_base64}
        )
        print(f"Status: {response.status_code}")
        try:
            print(f"Response: {response.json()}")
        except requests.exceptions.JSONDecodeError:
            print(f"Raw Response: {response.text}")
    except FileNotFoundError:
        print("❌ Please add a test_image.jpg file to test ingredient extraction")
    except Exception as e:
        print(f"❌ Ingredient Test Error: {str(e)}")

if __name__ == "__main__":
    test_endpoints()