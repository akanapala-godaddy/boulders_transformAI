import requests
import json
import os
import base64
from PIL import Image
import io

# Test image path - make sure this exists in your backend directory
TEST_IMAGE_PATH = 'test_image.jpg'

def test_remove_object():
    """Test the remove object endpoint directly"""
    print(f"Testing with image: {TEST_IMAGE_PATH}")
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image not found at {TEST_IMAGE_PATH}")
        return
    
    # Point in the middle of the image
    with Image.open(TEST_IMAGE_PATH) as img:
        width, height = img.size
        # Simple point in the center of the image
        point_coords = [[width // 2, height // 2]]
        point_labels = [1]  # 1 = object to be removed
    
    # Prepare the request
    url = 'http://localhost:8001/remove'
    
    # Create form data
    files = {'file': open(TEST_IMAGE_PATH, 'rb')}
    data = {
        'coords': json.dumps(point_coords),
        'labels': json.dumps(point_labels),
        'dilate_size': '15'
    }
    
    print(f"Sending request with coords: {point_coords}")
    
    # Send the request
    response = requests.post(url, files=files, data=data)
    
    print(f"Response status code: {response.status_code}")
    
    # Check if successful
    if response.status_code == 200:
        result = response.json()
        print("Success!")
        
        # Save results if available
        if 'inpainted_image' in result and 'mask_visualization' in result:
            # Save inpainted image
            inpainted_b64 = result['inpainted_image'].split(',')[1]
            inpainted_bytes = base64.b64decode(inpainted_b64)
            
            with open('test_result_inpainted.png', 'wb') as f:
                f.write(inpainted_bytes)
            print("Saved inpainted image to test_result_inpainted.png")
            
            # Save mask visualization
            mask_b64 = result['mask_visualization'].split(',')[1]
            mask_bytes = base64.b64decode(mask_b64)
            
            with open('test_result_mask.png', 'wb') as f:
                f.write(mask_bytes)
            print("Saved mask visualization to test_result_mask.png")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_remove_object() 