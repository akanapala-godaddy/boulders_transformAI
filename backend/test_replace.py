import requests
import os
import base64
from PIL import Image, ImageDraw
import io
import numpy as np

# Test image path - make sure this exists in your backend directory
TEST_IMAGE_PATH = 'test_image.jpg'

def test_replace_with_mask():
    """Test the replace_with_mask endpoint"""
    print(f"Testing with image: {TEST_IMAGE_PATH}")
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image not found at {TEST_IMAGE_PATH}")
        return
    
    # Load the test image
    with Image.open(TEST_IMAGE_PATH) as img:
        width, height = img.size
        
        # Create a simple mask (circle in the middle)
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        radius = min(width, height) // 4
        center_x, center_y = width // 2, height // 2
        draw.ellipse((center_x - radius, center_y - radius, 
                      center_x + radius, center_y + radius), fill=255)
        
        # Save mask for debugging
        mask.save('test_replace_mask.png')
        print(f"Created test mask with dimensions: {mask.size}")
        
        # Convert both to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        mask_bytes = io.BytesIO()
        mask.save(mask_bytes, format='PNG')
        mask_bytes.seek(0)
    
    # Prepare the request
    url = 'http://localhost:8001/replace_with_mask'
    
    # Create form data
    files = {
        'image': ('image.png', img_bytes, 'image/png'),
        'mask': ('mask.png', mask_bytes, 'image/png')
    }
    data = {
        'prompt': 'red bowtie',
        'dilate_size': '5'
    }
    
    print(f"Sending request to {url} with prompt: '{data['prompt']}'")
    
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
            
            with open('test_replace_result.png', 'wb') as f:
                f.write(inpainted_bytes)
            print("Saved replaced image to test_replace_result.png")
            
            # Save mask visualization
            mask_b64 = result['mask_visualization'].split(',')[1]
            mask_bytes = base64.b64decode(mask_b64)
            
            with open('test_replace_mask_vis.png', 'wb') as f:
                f.write(mask_bytes)
            print("Saved mask visualization to test_replace_mask_vis.png")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_replace_with_mask() 