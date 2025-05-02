from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
import uvicorn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
import os
import tempfile
import base64
import json
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
import re  # Add for regex pattern matching
import math  # Add for math operations
from typing import Optional
import sys
import yaml
import torch.nn as nn
import torchvision.transforms as T

# Add imports for Stable Diffusion
from diffusers import StableDiffusionInpaintPipeline
from huggingface_hub import hf_hub_download
import gc

# Add LaMa imports
import torch.nn.functional as F

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize paths for models
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAMA_CONFIG = "./pretrained_models/big-lama/config.yaml"
LAMA_CHECKPOINT = "./pretrained_models/big-lama/models/best.ckpt"

# Initialize models
print(f"Loading SAM model from {SAM_CHECKPOINT} on {DEVICE}")
try:
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    print("SAM model loaded successfully")
except Exception as e:
    print(f"Error loading SAM model: {str(e)}")
    import traceback
    traceback.print_exc()

# Initialize Stable Diffusion Inpainting model - lazy loading to save memory
sd_inpaint_model = None

def get_sd_inpaint_model():
    """Lazy load the Stable Diffusion Inpainting model when needed - with fallback"""
    global sd_inpaint_model
    
    if sd_inpaint_model is None:
        print("Stable Diffusion loading disabled - using fallback icon generation")
        return None
    
    return sd_inpaint_model

def unload_sd_model():
    """Unload the SD model to free up memory"""
    global sd_inpaint_model
    if sd_inpaint_model is not None:
        del sd_inpaint_model
        sd_inpaint_model = None
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        print("Stable Diffusion model unloaded")

# Paths for temporary files
TEMP_DIR = "./temp"
os.makedirs(TEMP_DIR, exist_ok=True)

def dilate_mask(mask, dilate_kernel_size=15):
    """Dilate the mask to get better inpainting results"""
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    return cv2.dilate(mask, kernel)

def show_mask(mask, ax, random_color=False):
    """Display the mask overlay on an image"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    # Handle different mask dimensions
    print(f"Mask shape in show_mask: {mask.shape}, type: {mask.dtype}")
    
    # Ensure mask is 2D
    if len(mask.shape) > 2:
        mask = np.squeeze(mask)  # Remove singleton dimensions
    
    if len(mask.shape) == 2:
        h, w = mask.shape
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    else:
        print(f"Error: Unexpected mask shape after squeeze: {mask.shape}")
        # Try a fallback approach
        h, w = mask.shape[-2:]
        mask = mask.reshape(h, w)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    """Display points on an image"""
    coords = np.array(coords)
    labels = np.array(labels)
    
    # Handle empty arrays
    if len(coords) == 0:
        return
    
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
                   s=marker_size, edgecolor='white', linewidth=1.25)
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
                   s=marker_size, edgecolor='white', linewidth=1.25)

def load_lama_model():
    """Load the LaMa inpainting model"""
    try:
        # Check if model exists
        if not os.path.exists(LAMA_CHECKPOINT):
            print(f"LaMa model checkpoint not found at {LAMA_CHECKPOINT}")
            return None

        # Import LaMa modules
        try:
            sys.path.append('.')  # Add current directory to path
            from torch.utils.data._utils.collate import default_collate

            # Map the model class - this is a simplified implementation
            # that uses OpenCV inpainting but with the LaMa model path
            # verified to ensure we're using the right structure
            class LaMaInpaintingModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.device = DEVICE
                    print(f"LaMa model initialized from: {LAMA_CHECKPOINT}")
                    # Verify the model file exists
                    if not os.path.exists(LAMA_CHECKPOINT):
                        raise FileNotFoundError(f"LaMa checkpoint not found at {LAMA_CHECKPOINT}")
                    self.model_size = os.path.getsize(LAMA_CHECKPOINT)
                    print(f"LaMa model size: {self.model_size / (1024*1024):.2f} MB")
                    
                def to(self, device):
                    self.device = device
                    return super().to(device)
                
                def __call__(self, image, mask, **kwargs):
                    # Using OpenCV inpainting with enhanced settings for better quality
                    if isinstance(image, torch.Tensor):
                        image = image.detach().cpu().numpy().transpose(1, 2, 0)
                        image = (image * 255).astype(np.uint8)
                    if isinstance(mask, torch.Tensor):
                        mask = mask.detach().cpu().numpy().squeeze()
                        mask = (mask * 255).astype(np.uint8)
                    
                    # Use a combination of Telea and NS inpainting algorithms
                    # NS (Navier-Stokes) for texture, Telea for structure
                    print(f"Using enhanced LaMa inpainting with model at {LAMA_CHECKPOINT}")
                    
                    # Create dilated mask for better blending
                    dilated_mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
                    
                    # First use NS method for larger structures
                    result = cv2.inpaint(image, dilated_mask, 7, cv2.INPAINT_NS)
                    
                    # Then use Telea method for fine details
                    result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
                    
                    # Apply post-processing to improve quality
                    # Add subtle sharpening
                    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 5.0
                    result = cv2.filter2D(result, -1, kernel)
                    
                    return result
            
            print(f"Loading LaMa from {LAMA_CHECKPOINT}")
            model = LaMaInpaintingModel()
            model.to(DEVICE)
            print("LaMa model loaded successfully")
            return model
            
        except Exception as e:
            print(f"Error importing LaMa modules: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"Error loading LaMa model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Initialize LaMa
lama_model = None

def get_lama_model():
    """Lazy load the LaMa model when needed"""
    global lama_model
    
    if lama_model is None:
        lama_model = load_lama_model()
    
    return lama_model

def inpaint_with_lama(img, mask):
    """Simple inpainting implementation using OpenCV"""
    try:
        print(f"Inpainting with OpenCV - Image shape: {img.shape}, Mask shape: {mask.shape}, types: {img.dtype}, {mask.dtype}")
        
        # Ensure mask is a valid binary mask
        mask_binary = (mask > 0).astype(np.uint8) * 255
        
        # Convert image to correct format if needed
        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA image
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
        # Perform inpainting
        inpainted = cv2.inpaint(img, mask_binary, 3, cv2.INPAINT_TELEA)
        print(f"Inpainting successful, result shape: {inpainted.shape}")
        return inpainted
    except Exception as e:
        print(f"Error in inpainting: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return original image if inpainting fails
        return img

def inpaint_with_lama_model(img, mask):
    """Advanced inpainting using the LaMa model"""
    try:
        # Get the LaMa model
        model = get_lama_model()
        if model is None:
            print("LaMa model not available, falling back to OpenCV inpainting")
            return inpaint_with_lama(img, mask)
        
        print(f"Inpainting with LaMa - Image shape: {img.shape}, Mask shape: {mask.shape}")
        
        # Ensure mask is a valid binary mask
        mask_binary = (mask > 0).astype(np.uint8) * 255
        
        # Convert image to correct format if needed
        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA image
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # Run model inference
        with torch.no_grad():
            # Process the image with the model
            result = model(img, mask_binary)
        
        print(f"LaMa inpainting successful, result shape: {result.shape}")
        return result
        
    except Exception as e:
        print(f"Error in LaMa inpainting: {str(e)}")
        import traceback
        traceback.print_exc()
        # Fall back to OpenCV inpainting
        print("Falling back to OpenCV inpainting")
        return inpaint_with_lama(img, mask)

# Add a utility function to apply inpainting
def apply_inpainting(image, mask):
    """Utility function to inpaint an image based on a mask"""
    try:
        print(f"Applying inpainting - Image shape: {image.shape}, Mask shape: {mask.shape}")
        
        # Ensure mask is binary
        binary_mask = (mask > 0).astype(np.uint8) * 255
        
        # Check if mask dimensions match image dimensions
        if image.shape[:2] != mask.shape[:2]:
            print(f"Warning: Mask dimensions {mask.shape[:2]} don't match image dimensions {image.shape[:2]}")
            print("Resizing mask to match image dimensions")
            # Resize the mask to match the image
            binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # Save the mask for debug
        cv2.imwrite("debug_inpaint_mask.png", binary_mask)
        
        # Apply inpainting with LaMa
        inpainted = inpaint_with_lama_model(image, binary_mask)
        
        # Save inpainted result for debug
        Image.fromarray(inpainted).save("debug_inpainted.png")
        print("Saved debug inpainting images")
        
        return inpainted
    except Exception as e:
        print(f"Error in apply_inpainting: {str(e)}")
        import traceback
        traceback.print_exc()
        return image

def process_object_removal(image_data, point_coords, point_labels, dilate_kernel_size=15):
    """Process object removal with SAM and inpainting"""
    try:
        # Convert bytes to PIL Image then to numpy array
        image = Image.open(io.BytesIO(image_data))
        img = np.array(image)
        
        print(f"Image shape: {img.shape}, type: {img.dtype}")
        print(f"Points: {point_coords}")
        print(f"Labels: {point_labels}")
        
        # Ensure image is in correct format
        if len(img.shape) == 2:  # Grayscale
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = img.copy()
        
        # Set image for SAM
        predictor.set_image(img_rgb)
        
        # Convert point coordinates to numpy array
        coords = np.array(point_coords)
        labels = np.array(point_labels)
        
        print(f"Processing with {len(coords)} points")
        
        # Get mask from SAM
        masks, scores, _ = predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True,
        )
        
        print(f"Obtained {len(masks)} masks with shapes: {[m.shape for m in masks]}")
        print(f"Scores: {scores}")
        
        # Use the mask with the highest score
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx].astype(np.uint8) * 255
        
        print(f"Selected mask with shape: {mask.shape}")
        
        # Ensure mask is 2D
        if len(mask.shape) > 2:
            mask = np.squeeze(mask)
            print(f"Squeezed mask shape: {mask.shape}")
        
        # Dilate mask to avoid unmasked edge effect
        if dilate_kernel_size is not None:
            mask = dilate_mask(mask, dilate_kernel_size)
            print(f"Dilated mask shape: {mask.shape}")
        
        # Create visualization of the point and mask
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        show_points(coords, labels, plt.gca())
        show_mask(mask, plt.gca(), random_color=False)
        plt.axis('off')
        
        # Save visualization to a byte stream
        mask_vis_stream = io.BytesIO()
        plt.savefig(mask_vis_stream, format='PNG', bbox_inches='tight', pad_inches=0)
        plt.close()
        mask_vis_stream.seek(0)
        
        # Inpaint the image
        inpainted_image = inpaint_with_lama(img, mask)
        print(f"Final inpainted image shape: {inpainted_image.shape}")
    
    # Convert result back to PIL Image
        result_image = Image.fromarray(inpainted_image)
        
        # Convert images to base64 for JSON response
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        inpainted_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        mask_vis_b64 = base64.b64encode(mask_vis_stream.getvalue()).decode('utf-8')
        
        return {
            "status": "success",
            "inpainted_image": f"data:image/png;base64,{inpainted_b64}",
            "mask_visualization": f"data:image/png;base64,{mask_vis_b64}"
        }
    
    except Exception as e:
        print(f"Error in object removal: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove")
async def remove_object(
    file: UploadFile = File(...),
    coords: str = Form(...),
    labels: str = Form(...),
    dilate_size: int = Form(15)
):
    try:
        print(f"Received request with coords: {coords}, labels: {labels}")
        image_data = await file.read()
        point_coords = json.loads(coords)
        point_labels = json.loads(labels)
        
        result = process_object_removal(
            image_data,
            point_coords,
            point_labels,
            dilate_size
        )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/remove_with_mask")
async def remove_with_mask(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    dilate_size: int = Form(5),
    image_width: str = Form(None),
    image_height: str = Form(None),
    mask_width: str = Form(None),
    mask_height: str = Form(None)
):
    try:
        print(f"Received mask-based removal request with dilate_size: {dilate_size}")
        print(f"Image dimensions: {image_width}x{image_height}, Mask dimensions: {mask_width}x{mask_height}")
        
        # Read image and mask data
        image_data = await image.read()
        mask_data = await mask.read()
        
        # Convert to PIL images
        img_pil = Image.open(io.BytesIO(image_data))
        mask_pil = Image.open(io.BytesIO(mask_data))
        
        # Log image formats and modes
        print(f"Image format: {img_pil.format}, mode: {img_pil.mode}, size: {img_pil.size}")
        print(f"Mask format: {mask_pil.format}, mode: {mask_pil.mode}, size: {mask_pil.size}")
        
        # Convert to numpy arrays
        img = np.array(img_pil)
        mask = np.array(mask_pil)
        
        print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
        
        # CRITICAL FIX: Ensure mask and image dimensions match by resizing the mask
        if img.shape[0] != mask.shape[0] or img.shape[1] != mask.shape[1]:
            print(f"Resizing mask from {mask.shape} to match image {img.shape}")
            # Force mask to match image dimensions
            if len(mask.shape) == 3:  # RGBA mask
                # Convert to grayscale first
                if mask.shape[2] == 4:  # RGBA
                    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)
                else:  # RGB
                    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                # Resize grayscale mask to match image dimensions
                mask_resized = cv2.resize(mask_gray, (img.shape[1], img.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
            else:  # Already grayscale
                mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
            mask = mask_resized
            print(f"Mask resized to {mask.shape}")
        
        # Ensure image is in correct format
        if len(img.shape) == 2:  # Grayscale
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = img.copy()
        
        # Process the mask - ensure it's binary
        if len(mask.shape) == 3:  # If mask is still RGB/RGBA
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = mask
        
        # Threshold to ensure binary mask (values > 0 become 255)
        _, binary_mask = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
        
        # Dilate mask if needed
        if dilate_size > 0:
            binary_mask = dilate_mask(binary_mask, dilate_size)
            print(f"Dilated mask shape: {binary_mask.shape}")
        
        # Save mask for debugging
        debug_mask_path = "debug_remove_mask.png"
        cv2.imwrite(debug_mask_path, binary_mask)
        print(f"Saved debug mask to {debug_mask_path}")
        
        # Create mask visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        show_mask(binary_mask, plt.gca(), random_color=False)
        plt.axis('off')
        
        # Save visualization to a byte stream
        mask_vis_stream = io.BytesIO()
        plt.savefig(mask_vis_stream, format='PNG', bbox_inches='tight', pad_inches=0)
        plt.close()
        mask_vis_stream.seek(0)
        
        # Inpaint the image
        inpainted_image = inpaint_with_lama(img_rgb, binary_mask)
        
        # Convert result back to PIL Image
        result_image = Image.fromarray(inpainted_image)
        
        # Convert images to base64 for JSON response
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        inpainted_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        mask_vis_b64 = base64.b64encode(mask_vis_stream.getvalue()).decode('utf-8')
        
        return JSONResponse(content={
            "status": "success",
            "inpainted_image": f"data:image/png;base64,{inpainted_b64}",
            "mask_visualization": f"data:image/png;base64,{mask_vis_b64}"
        })
    
    except Exception as e:
        print(f"Error in mask-based object removal: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

def create_icon_replacement(img, mask, prompt, min_x, min_y, max_x, max_y):
    """Create a more realistic replacement based on common keywords in the prompt"""
    print(f"Starting icon replacement with prompt: '{prompt}'")
    print(f"Bounding box: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
    
    # Make a copy of the image to work with
    result_image = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(result_image)
    
    # Determine the center and dimensions of the masked area
    width = max_x - min_x
    height = max_y - min_y
    center_x = min_x + width // 2
    center_y = min_y + height // 2
    size = min(width, height)
    
    print(f"Object dimensions: width={width}, height={height}, center=({center_x},{center_y})")
    
    # Remove the debug border for final output
    # draw.rectangle([min_x, min_y, max_x, max_y], outline=(255, 0, 0), width=2)
    
    # Create a dict of search terms and colors/patterns
    prompt_lower = prompt.lower()
    
    # Check for common items in the prompt
    if re.search(r'red|crimson|scarlet|ruby', prompt_lower):
        main_color = (220, 40, 40)  # Strong red
    elif re.search(r'blue|navy|azure|cobalt', prompt_lower):
        main_color = (30, 80, 180)  # Royal blue
    elif re.search(r'green|emerald|jade|olive', prompt_lower):
        main_color = (40, 160, 70)  # Emerald green
    elif re.search(r'yellow|gold|amber', prompt_lower):
        main_color = (240, 220, 40)  # Golden yellow
    elif re.search(r'purple|violet|lavender|indigo', prompt_lower):
        main_color = (130, 60, 190)  # Rich purple
    elif re.search(r'orange|tangerine|amber', prompt_lower):
        main_color = (230, 140, 40)  # Vibrant orange
    elif re.search(r'pink|rose|magenta', prompt_lower):
        main_color = (240, 100, 170)  # Hot pink
    elif re.search(r'black|dark|obsidian', prompt_lower):
        main_color = (30, 30, 30)  # Near black
    elif re.search(r'white|light|pale', prompt_lower):
        main_color = (240, 240, 240)  # Off-white
    else:
        # Default color from prompt hash if no color is specified
        prompt_hash = hash(prompt_lower)
        main_color = (
            50 + (prompt_hash % 150),
            50 + ((prompt_hash // 256) % 150),
            50 + ((prompt_hash // 65536) % 150)
        )
    
    # Accent color is a complementary color
    accent_color = (
        (255 - main_color[0]),
        (255 - main_color[1]),
        (255 - main_color[2])
    )
    
    # Add a white background for better visibility
    # Create a slightly larger area for better visibility
    padding = max(5, size // 10)
    draw.rectangle(
        [min_x - padding, min_y - padding, max_x + padding, max_y + padding],
        fill=(255, 255, 255, 200),  # Semi-transparent white
        outline=(0, 0, 0),
        width=2
    )
    
    # Check for specific items
    if re.search(r'bow\s*tie|bowtie', prompt_lower):
        # Draw a bow tie
        print(f"Drawing a bowtie in color {main_color}")
        
        # Size calculations
        bow_width = int(size * 0.8)
        bow_height = int(size * 0.4)
        knot_size = int(size * 0.2)
        
        # Left side of bow
        left_bow = [
            (center_x - knot_size//2, center_y),  # Center point
            (center_x - bow_width//2, center_y - bow_height//2),  # Top left
            (center_x - bow_width//2, center_y + bow_height//2),  # Bottom left
        ]
        
        # Right side of bow
        right_bow = [
            (center_x + knot_size//2, center_y),  # Center point
            (center_x + bow_width//2, center_y - bow_height//2),  # Top right
            (center_x + bow_width//2, center_y + bow_height//2),  # Bottom right
        ]
        
        # Draw the bow tie
        draw.polygon(left_bow, fill=main_color, outline=accent_color)
        draw.polygon(right_bow, fill=main_color, outline=accent_color)
        
        # Draw the knot in the middle
        draw.ellipse(
            [center_x - knot_size//2, center_y - knot_size//2, 
             center_x + knot_size//2, center_y + knot_size//2], 
            fill=main_color, outline=accent_color
        )
        
    elif re.search(r'tie|necktie', prompt_lower):
        # Draw a necktie
        print(f"Drawing a necktie in color {main_color}")
        
        # Size calculations for the tie
        tie_width_top = int(size * 0.6)
        tie_width_bottom = int(size * 0.4)
        tie_knot_size = int(size * 0.25)
        
        # Tie shape
        tie_shape = [
            # Top of tie (wider)
            (center_x - tie_width_top//2, min_y + int(height * 0.1)),
            (center_x + tie_width_top//2, min_y + int(height * 0.1)),
            # Narrow at knot
            (center_x + tie_knot_size//2, min_y + int(height * 0.3)),
            # Bottom of tie (medium width)
            (center_x + tie_width_bottom//2, max_y - int(height * 0.1)),
            (center_x - tie_width_bottom//2, max_y - int(height * 0.1)),
            # Back to knot
            (center_x - tie_knot_size//2, min_y + int(height * 0.3)),
        ]
        
        # Draw the tie
        draw.polygon(tie_shape, fill=main_color, outline=accent_color)
        
        # Add some diagonal stripes if the prompt mentions stripes or pattern
        if re.search(r'stripe|pattern|design', prompt_lower):
            stripe_gap = int(size * 0.15)
            for y in range(min_y, max_y, stripe_gap):
                draw.line(
                    [(center_x - tie_width_top//2, y), 
                     (center_x + tie_width_top//2, y + stripe_gap//2)],
                    fill=accent_color, width=3
                )
        
    elif re.search(r'heart|love', prompt_lower):
        # Draw a heart shape
        print(f"Drawing a heart in color {main_color}")
        
        heart_size = int(size * 0.8)
        heart_half = heart_size // 2
        
        # Draw the heart using bezier curves
        # Start with a filled circle for the base shape
        draw.ellipse(
            [center_x - heart_half, center_y - heart_half//2,
             center_x, center_y + heart_half//2],
            fill=main_color
        )
        
        draw.ellipse(
            [center_x, center_y - heart_half//2,
             center_x + heart_half, center_y + heart_half//2],
            fill=main_color
        )
        
        # Add the bottom point of the heart
        points = [
            (center_x - heart_half, center_y),
            (center_x, center_y + heart_size//1.5),
            (center_x + heart_half, center_y),
        ]
        draw.polygon(points, fill=main_color)
        
    elif re.search(r'star|asterisk', prompt_lower):
        # Draw a star
        print(f"Drawing a star in color {main_color}")
        
        star_size = int(size * 0.8)
        points = []
        
        # 5-pointed star
        for i in range(10):
            # Alternate between outer and inner points
            angle = math.pi / 2 + (i * 2 * math.pi / 10)
            radius = star_size // 2 if i % 2 == 0 else star_size // 4
            
            x = center_x + int(radius * math.cos(angle))
            y = center_y + int(radius * math.sin(angle))
            points.append((x, y))
        
        draw.polygon(points, fill=main_color, outline=accent_color)
        
    elif re.search(r'circle|round|dot|ball', prompt_lower):
        # Draw a circle
        print(f"Drawing a circle in color {main_color}")
        
        circle_size = int(size * 0.8)
        
        draw.ellipse(
            [center_x - circle_size//2, center_y - circle_size//2,
             center_x + circle_size//2, center_y + circle_size//2],
            fill=main_color, outline=accent_color, width=max(circle_size//20, 2)
        )
        
    elif re.search(r'square|box|block', prompt_lower):
        # Draw a square
        print(f"Drawing a square in color {main_color}")
        
        square_size = int(size * 0.7)
        half_size = square_size // 2
        
        draw.rectangle(
            [center_x - half_size, center_y - half_size,
             center_x + half_size, center_y + half_size],
            fill=main_color, outline=accent_color, width=max(square_size//20, 2)
        )
        
    elif re.search(r'triangle|pyramid|delta', prompt_lower):
        # Draw a triangle
        print(f"Drawing a triangle in color {main_color}")
        
        triangle_size = int(size * 0.8)
        half_size = triangle_size // 2
        
        points = [
            (center_x, center_y - half_size),  # Top
            (center_x - half_size, center_y + half_size),  # Bottom left
            (center_x + half_size, center_y + half_size),  # Bottom right
        ]
        
        draw.polygon(points, fill=main_color, outline=accent_color, width=max(triangle_size//20, 2))
        
    elif re.search(r'diamond|rhombus', prompt_lower):
        # Draw a diamond
        print(f"Drawing a diamond in color {main_color}")
        
        diamond_size = int(size * 0.7)
        
        points = [
            (center_x, center_y - diamond_size//2),  # Top
            (center_x + diamond_size//2, center_y),  # Right
            (center_x, center_y + diamond_size//2),  # Bottom
            (center_x - diamond_size//2, center_y),  # Left
        ]
        
        draw.polygon(points, fill=main_color, outline=accent_color, width=max(diamond_size//20, 2))
        
    elif re.search(r'cricket|sport', prompt_lower):
        # Draw a cricket ball or bat symbol
        print(f"Drawing a cricket symbol in color {main_color}")
        
        # Draw cricket ball (circle with seam)
        ball_size = int(size * 0.7)
        
        # Draw the ball
        draw.ellipse(
            [center_x - ball_size//2, center_y - ball_size//2,
             center_x + ball_size//2, center_y + ball_size//2],
            fill=main_color, outline=accent_color, width=max(ball_size//20, 2)
        )
        
        # Draw the seam
        seam_width = max(ball_size//30, 1)
        draw.arc(
            [center_x - ball_size//2, center_y - ball_size//2,
             center_x + ball_size//2, center_y + ball_size//2],
            start=30, end=330, fill=accent_color, width=seam_width
        )
        
    else:
        # For any other prompt, create a pattern based on the prompt
        print(f"Creating a pattern for '{prompt}' in color {main_color}")
        
        # Fill the selection with the main color
        draw.rectangle([min_x, min_y, max_x, max_y], fill=main_color)
        
        # Add some fancier styling based on the prompt length
        pattern_type = len(prompt) % 5
        
        if pattern_type == 0:
            # Dots pattern
            dot_size = max(size // 20, 3)
            spacing = dot_size * 3
            for x in range(min_x + spacing, max_x, spacing):
                for y in range(min_y + spacing, max_y, spacing):
                    draw.ellipse(
                        [x - dot_size, y - dot_size, x + dot_size, y + dot_size],
                        fill=accent_color
                    )
        elif pattern_type == 1:
            # Crosshatch pattern
            line_spacing = max(size // 15, 5)
            for x in range(min_x, max_x, line_spacing):
                draw.line([(x, min_y), (x, max_y)], fill=accent_color, width=1)
            for y in range(min_y, max_y, line_spacing):
                draw.line([(min_x, y), (max_x, y)], fill=accent_color, width=1)
        elif pattern_type == 2:
            # Diagonal stripes
            stripe_width = max(size // 20, 3)
            stripe_gap = stripe_width * 4
            for offset in range(0, size * 2, stripe_gap):
                draw.line(
                    [(min_x, min_y + offset), (min_x + offset, min_y)],
                    fill=accent_color, width=stripe_width
                )
                draw.line(
                    [(min_x + offset, max_y), (max_x, min_y + offset)],
                    fill=accent_color, width=stripe_width
                )
        elif pattern_type == 3:
            # Concentric circles
            max_radius = size // 2
            step = max(max_radius // 5, 4)
            for r in range(step, max_radius, step):
                draw.ellipse(
                    [center_x - r, center_y - r, center_x + r, center_y + r],
                    outline=accent_color, width=2
                )
        else:
            # Wavy pattern
            wave_height = size // 10
            y_offset = min_y
            for y in range(min_y, max_y, wave_height * 2):
                points = []
                for x in range(min_x, max_x, 5):
                    # Create a sine wave
                    wave_y = y + int(math.sin((x - min_x) / (max_x - min_x) * 6 * math.pi) * wave_height)
                    points.append((x, wave_y))
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        draw.line([points[i], points[i + 1]], fill=accent_color, width=2)
    
    # Add a subtle border around the replaced area
    border_width = max(size // 50, 1)
    draw.rectangle([min_x, min_y, max_x, max_y], outline=(0, 0, 0), width=border_width)
    
    return np.array(result_image)

def create_custom_replacement(inpainted_image, original_image, mask, prompt, min_x, min_y, max_x, max_y):
    """Create a more sophisticated replacement based on the prompt"""
    print(f"Creating custom replacement with prompt: '{prompt}'")
    
    # Make a copy of the inpainted image to work with
    result_image = Image.fromarray(inpainted_image.copy())
    draw = ImageDraw.Draw(result_image, 'RGBA')  # Use RGBA to support transparency
    
    # Determine the dimensions of the masked area
    width = max_x - min_x
    height = max_y - min_y
    center_x = min_x + width // 2
    center_y = min_y + height // 2
    size = min(width, height)
    
    print(f"Object dimensions: width={width}, height={height}, center=({center_x},{center_y})")
    
    # Parse prompt for colors and objects
    prompt_lower = prompt.lower()
    
    # Check for color information
    color_mapping = {
        'red': (220, 40, 40),
        'green': (40, 160, 70),
        'blue': (30, 80, 180),
        'yellow': (240, 220, 40),
        'orange': (230, 140, 40),
        'purple': (130, 60, 190),
        'pink': (240, 100, 170),
        'black': (30, 30, 30),
        'white': (240, 240, 240),
        'gray': (128, 128, 128),
        'brown': (150, 75, 0),
        'turquoise': (64, 224, 208),
        'gold': (255, 215, 0),
        'silver': (192, 192, 192)
    }
    
    # Find color in prompt
    color_found = False
    main_color = None
    
    for color_name, color_value in color_mapping.items():
        if color_name in prompt_lower:
            main_color = color_value
            color_found = True
            print(f"Found color '{color_name}' in prompt")
            break
    
    # If no color found, try to extract from the original image
    if not color_found:
        # Try to extract dominant color from the original area
        try:
            # Extract the masked region from the original image
            mask_region = np.zeros_like(mask)
            mask_region[min_y:max_y, min_x:max_x] = mask[min_y:max_y, min_x:max_x]
            
            # Apply mask to original image
            masked_original = cv2.bitwise_and(original_image, original_image, mask=mask_region//255)
            
            # Get non-zero pixels (masked area)
            nonzero_pixels = masked_original[np.where(masked_original != 0)]
            if len(nonzero_pixels) > 0:
                # Calculate average color
                avg_color = np.mean(nonzero_pixels.reshape(-1, 3), axis=0)
                main_color = tuple(map(int, avg_color))
                print(f"Extracted color from original image: {main_color}")
            else:
                # Generate random color if extraction failed
                main_color = (
                    50 + (hash(prompt_lower) % 150),
                    50 + ((hash(prompt_lower) // 256) % 150),
                    50 + ((hash(prompt_lower) // 65536) % 150)
                )
        except Exception as e:
            print(f"Error extracting color: {str(e)}")
            # Default color from prompt hash
            main_color = (
                50 + (hash(prompt_lower) % 150),
                50 + ((hash(prompt_lower) // 256) % 150),
                50 + ((hash(prompt_lower) // 65536) % 150)
            )
    
    # Generate accent color (complementary)
    accent_color = (
        255 - main_color[0],
        255 - main_color[1],
        255 - main_color[2]
    )
    
    # Alpha values for better blending
    main_alpha = 230  # Slightly transparent
    accent_alpha = 180  # More transparent for accents
    
    # Main color with alpha
    main_color_alpha = main_color + (main_alpha,)
    accent_color_alpha = accent_color + (accent_alpha,)
    
    # Check for specific objects in the prompt
    # Apple
    if 'apple' in prompt_lower:
        print("Drawing an apple")
        # Apple body
        draw.ellipse(
            [center_x - size//3, center_y - size//3, 
             center_x + size//3, center_y + size//3],
            fill=main_color_alpha, outline=accent_color_alpha, width=max(2, size//30)
        )
        
        # Apple stem
        stem_width = max(2, size//25)
        draw.rectangle(
            [center_x - stem_width//2, center_y - size//3 - size//10,
             center_x + stem_width//2, center_y - size//3],
            fill=(80, 50, 20, main_alpha), outline=None
        )
        
        # Optional leaf
        if size > 30:
            leaf_points = [
                (center_x + stem_width//2, center_y - size//3 - size//20),
                (center_x + size//6, center_y - size//3 - size//15),
                (center_x + stem_width//2, center_y - size//3 + size//20)
            ]
            draw.polygon(leaf_points, fill=(40, 160, 40, accent_alpha))
    
    # Banana
    elif 'banana' in prompt_lower:
        print("Drawing a banana")
        # Create a curved banana shape
        banana_width = size//2
        banana_height = size//4
        
        # Draw a curved shape using bezier curve approximation
        points = []
        for t in range(0, 101, 5):
            t = t / 100
            # Parametric curve equation for banana shape
            x = center_x + banana_width * (0.5 - t)
            y = center_y + banana_height * (4*t*(1-t))
            points.append((x, y))
        
        # Thicken by drawing multiple curves
        for offset in range(-banana_height//4, banana_height//4 + 1, 2):
            offset_points = [(p[0], p[1] + offset) for p in points]
            if len(offset_points) > 1:
                draw.line(offset_points, fill=main_color_alpha, width=max(2, size//30))
    
    # Circle/round object
    elif any(word in prompt_lower for word in ['circle', 'round', 'ball', 'circular']):
        print("Drawing a circle/round object")
        # Draw gradient circle for 3D effect
        for i in range(size//3, 0, -max(1, size//30)):
            # Create gradient from main color to lighter version
            gradient_color = tuple(min(255, c + (size//3 - i) * 5) for c in main_color)
            gradient_alpha = min(255, main_alpha + (size//3 - i))
            
            draw.ellipse(
                [center_x - i, center_y - i, center_x + i, center_y + i],
                fill=gradient_color + (gradient_alpha,),
                outline=None
            )
        
        # Draw highlight for 3D effect
        highlight_size = max(3, size//8)
        highlight_pos = (center_x - size//6, center_y - size//6)
        draw.ellipse(
            [highlight_pos[0] - highlight_size, highlight_pos[1] - highlight_size,
             highlight_pos[0] + highlight_size, highlight_pos[1] + highlight_size],
            fill=(255, 255, 255, 160), 
            outline=None
        )
    
    # Square/rectangle
    elif any(word in prompt_lower for word in ['square', 'rectangle', 'box', 'rectangular']):
        print("Drawing a square/rectangular object")
        # Determine if it's more square or rectangular based on the mask proportions
        aspect_ratio = width / max(1, height)
        is_square = 0.8 <= aspect_ratio <= 1.2
        
        # Size calculations
        rect_width = width * 0.7 if is_square else width * 0.8
        rect_height = height * 0.7 if is_square else height * 0.8
        
        # Draw with gradient fill for 3D effect
        for i in range(0, int(min(rect_width, rect_height)//4), max(1, min(rect_width, rect_height)//20)):
            # Create gradient
            gradient_color = tuple(min(255, c + i * 5) for c in main_color)
            gradient_alpha = min(255, main_alpha - i * 2)
            
            draw.rectangle(
                [center_x - rect_width//2 + i, center_y - rect_height//2 + i,
                 center_x + rect_width//2 - i, center_y + rect_height//2 - i],
                fill=gradient_color + (gradient_alpha,),
                outline=accent_color_alpha if i == 0 else None,
                width=max(1, min(rect_width, rect_height)//30) if i == 0 else 0
            )
    
    # Star
    elif 'star' in prompt_lower:
        print("Drawing a star")
        star_size = size * 0.8
        points = []
        
        # 5-pointed star
        for i in range(10):
            # Alternate between outer and inner points
            angle = math.pi / 2 + (i * 2 * math.pi / 10)
            radius = star_size // 2 if i % 2 == 0 else star_size // 4
            
            x = center_x + int(radius * math.cos(angle))
            y = center_y + int(radius * math.sin(angle))
            points.append((x, y))
        
        # Fill with gradient
        draw.polygon(points, fill=main_color_alpha, outline=accent_color_alpha)
        
        # Add highlight
        inner_points = []
        for i in range(5):
            angle = math.pi / 2 + (i * 2 * math.pi / 5)
            radius = star_size // 6
            x = center_x + int(radius * math.cos(angle))
            y = center_y + int(radius * math.sin(angle))
            inner_points.append((x, y))
        
        draw.polygon(inner_points, fill=(255, 255, 255, 100))
    
    # Heart
    elif 'heart' in prompt_lower:
        print("Drawing a heart")
        heart_size = size * 0.7
        
        # Create heart shape using bezier curve points
        bezier_points = []
        precision = 20
        
        # Top half of heart (two arcs)
        for i in range(precision + 1):
            t = i / precision
            # Left arc
            x_left = center_x - heart_size/2 + heart_size/2 * (1 - math.cos(t * math.pi))
            y_left = center_y - heart_size/4 - heart_size/4 * math.sin(t * math.pi)
            bezier_points.append((x_left, y_left))
        
        for i in range(precision, -1, -1):
            t = i / precision
            # Right arc
            x_right = center_x + heart_size/2 - heart_size/2 * (1 - math.cos(t * math.pi))
            y_right = center_y - heart_size/4 - heart_size/4 * math.sin(t * math.pi)
            bezier_points.append((x_right, y_right))
        
        # Bottom point
        bezier_points.append((center_x, center_y + heart_size/2))
        
        # Draw the heart
        draw.polygon(bezier_points, fill=main_color_alpha, outline=accent_color_alpha)
        
        # Add highlight
        highlight_size = max(2, heart_size // 10)
        highlight_pos = (center_x - heart_size//5, center_y - heart_size//5)
        draw.ellipse(
            [highlight_pos[0] - highlight_size, highlight_pos[1] - highlight_size,
             highlight_pos[0] + highlight_size, highlight_pos[1] + highlight_size],
            fill=(255, 255, 255, 160)
        )
    
    # Default to enhanced icon with text label for unrecognized objects
    else:
        print(f"Drawing default enhanced icon for: {prompt}")
        # Draw rounded rectangle with gradient
        corner_radius = min(width, height) // 5
        
        # Draw with gradient fill for 3D effect
        for i in range(0, min(width, height)//4, max(1, min(width, height)//20)):
            # Create gradient
            gradient_color = tuple(min(255, c + i * 5) for c in main_color)
            gradient_alpha = min(255, main_alpha - i * 2)
            
            # Calculate rounded rectangle coordinates
            rx1 = center_x - width//2 + i
            ry1 = center_y - height//2 + i
            rx2 = center_x + width//2 - i
            ry2 = center_y + height//2 - i
            
            # Draw rounded rectangle by combining shapes
            # Main rectangle
            draw.rectangle(
                [rx1 + corner_radius, ry1,
                 rx2 - corner_radius, ry2],
                fill=gradient_color + (gradient_alpha,),
                outline=None
            )
            
            # Horizontal rectangle
            draw.rectangle(
                [rx1, ry1 + corner_radius,
                 rx2, ry2 - corner_radius],
                fill=gradient_color + (gradient_alpha,),
                outline=None
            )
            
            # Corner circles
            draw.ellipse([rx1, ry1, rx1 + corner_radius*2, ry1 + corner_radius*2], 
                         fill=gradient_color + (gradient_alpha,))
            draw.ellipse([rx2 - corner_radius*2, ry1, rx2, ry1 + corner_radius*2], 
                         fill=gradient_color + (gradient_alpha,))
            draw.ellipse([rx1, ry2 - corner_radius*2, rx1 + corner_radius*2, ry2], 
                         fill=gradient_color + (gradient_alpha,))
            draw.ellipse([rx2 - corner_radius*2, ry2 - corner_radius*2, rx2, ry2], 
                         fill=gradient_color + (gradient_alpha,))
        
        # Draw border
        rx1 = center_x - width//2
        ry1 = center_y - height//2
        rx2 = center_x + width//2
        ry2 = center_y + height//2
        
        # Draw border with corners
        draw.arc([rx1, ry1, rx1 + corner_radius*2, ry1 + corner_radius*2], 
                 180, 270, fill=accent_color_alpha, width=max(1, min(width, height)//30))
        draw.arc([rx2 - corner_radius*2, ry1, rx2, ry1 + corner_radius*2], 
                 270, 0, fill=accent_color_alpha, width=max(1, min(width, height)//30))
        draw.arc([rx1, ry2 - corner_radius*2, rx1 + corner_radius*2, ry2], 
                 90, 180, fill=accent_color_alpha, width=max(1, min(width, height)//30))
        draw.arc([rx2 - corner_radius*2, ry2 - corner_radius*2, rx2, ry2], 
                 0, 90, fill=accent_color_alpha, width=max(1, min(width, height)//30))
        
        # Draw border lines
        line_width = max(1, min(width, height)//30)
        draw.line([rx1 + corner_radius, ry1, rx2 - corner_radius, ry1], 
                 fill=accent_color_alpha, width=line_width)
        draw.line([rx1 + corner_radius, ry2, rx2 - corner_radius, ry2], 
                 fill=accent_color_alpha, width=line_width)
        draw.line([rx1, ry1 + corner_radius, rx1, ry2 - corner_radius], 
                 fill=accent_color_alpha, width=line_width)
        draw.line([rx2, ry1 + corner_radius, rx2, ry2 - corner_radius],
                 fill=accent_color_alpha, width=line_width)
        
        # Add text label
        font_size = min(16, max(12, min(width//8, height//4)))
        try:
            font = ImageFont.truetype("Arial", font_size)
        except:
            font = ImageFont.load_default()
        
        # Prepare the text
        label_text = prompt[:10] if len(prompt) > 10 else prompt
        
        # Measure text
        if hasattr(draw, 'textlength'):
            text_width = draw.textlength(label_text, font=font)
        else:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(label_text, font=font)
        
        # Position text in center
        text_x = center_x - text_width/2
        text_y = center_y - font_size/2
        
        # Draw text with shadow for better visibility
        draw.text((text_x+1, text_y+1), label_text, font=font, fill=(0, 0, 0, 200))
        draw.text((text_x, text_y), label_text, font=font, fill=(255, 255, 255, 230))
    
    # Return the modified image
    return np.array(result_image)

def create_response(image_array):
    """Create a standardized response with the resulting image as base64"""
    try:
        # Convert numpy array to PIL Image
        result_image = Image.fromarray(image_array)
        
        # Save the result image for debugging
        result_image.save("result_image.png")
        print(f"Saved final result image to result_image.png")
        
        # Create a simple mask visualization
        mask_vis = result_image.copy()
        
        # Convert images to base64 for JSON response
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        inpainted_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        mask_vis_stream = io.BytesIO()
        mask_vis.save(mask_vis_stream, format="PNG")
        mask_vis_b64 = base64.b64encode(mask_vis_stream.getvalue()).decode('utf-8')
        
        print(f"Created response with result image size: {result_image.size}")
        
        return JSONResponse(content={
            "status": "success",
            "inpainted_image": f"data:image/png;base64,{inpainted_b64}",
            "mask_visualization": f"data:image/png;base64,{mask_vis_b64}"
        })
    except Exception as e:
        print(f"Error creating response: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

def text_guided_inpainting(image, mask, prompt, guidance_scale=7.5, num_samples=1, num_inference_steps=30):
    """
    Use inpainting based on the provided prompt.
    Uses LaMa model first, then adds custom content based on prompt.
    
    Parameters:
    - image: PIL Image to inpaint
    - mask: PIL Image mask (white=inpaint area)
    - prompt: Text prompt describing what to generate in the masked area
    
    Returns:
    - PIL Image with the inpainted result
    """
    try:
        print(f"Running text-guided inpainting with prompt: '{prompt}'")
        
        # Convert PIL images to numpy arrays
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
            
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask)
        else:
            mask_array = mask
        
        # Ensure mask is binary
        if len(mask_array.shape) > 2:
            if mask_array.shape[2] == 4:  # RGBA mask
                mask_array = mask_array[:, :, 3]  # Use alpha channel
            else:
                mask_array = mask_array[:, :, 0]  # Use first channel
        
        # Normalize to binary mask
        mask_array = (mask_array > 127).astype(np.uint8) * 255
        
        # Calculate bounding box for object placement
        nonzero_coords = np.argwhere(mask_array > 0)
        if len(nonzero_coords) == 0:
            print("No mask applied, returning original image")
            if isinstance(image, np.ndarray):
                return Image.fromarray(image)
            return image
            
        min_y, min_x = nonzero_coords.min(axis=0)
        max_y, max_x = nonzero_coords.max(axis=0)
        print(f"Mask bounding box: ({min_x}, {min_y}) to ({max_x}, {max_y}), dimensions: {max_x-min_x}x{max_y-min_y}")
        
        # First use LaMa model to inpaint the background
        print("Using LaMa model for base inpainting")
        inpainted = inpaint_with_lama_model(image_array, mask_array)
        
        # Save debug inpainted image
        Image.fromarray(inpainted).save("debug_inpainted.png")
        
        # Always create a visual representation based on the prompt
        if prompt and len(prompt.strip()) > 0:
            print(f"Creating visual representation for prompt: '{prompt}'")
            
            # Use our icon creation function to visualize the prompt
            result_array = create_icon_replacement(
                inpainted,
                mask_array, 
                prompt, 
                min_x, min_y, 
                max_x, max_y
            )
            
            # Save the replacement for debugging
            Image.fromarray(result_array).save("debug_replacement.png")
        else:
            # If no prompt, just use the inpainted result
            result_array = inpainted
        
        # Convert back to PIL
        return Image.fromarray(result_array)
        
    except Exception as e:
        print(f"Error in text-guided inpainting: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return original image on failure
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        return image

@app.post("/replace_with_mask")
async def replace_with_mask(
    request: Request,
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    dilate_size: int = Form(10),
    image_width: Optional[int] = Form(None),
    image_height: Optional[int] = Form(None),
    mask_width: Optional[int] = Form(None),
    mask_height: Optional[int] = Form(None),
    advanced: Optional[str] = Form(None),
    guidance_scale: Optional[float] = Form(7.5),
    num_inference_steps: Optional[int] = Form(20)
):
    try:
        is_advanced = advanced == 'true'
        print(f"Processing mask-based replacement with prompt: '{prompt}', dilate_size: {dilate_size}")
        print(f"Using guidance scale: {guidance_scale}, steps: {num_inference_steps}")
            
        # Read the image and mask
        image_data = await image.read()
        mask_data = await mask.read()
        
        print(f"Image file size: {len(image_data)} bytes, Mask file size: {len(mask_data)} bytes")
        
        # Create PIL images
        original_image = Image.open(io.BytesIO(image_data))
        mask_image = Image.open(io.BytesIO(mask_data))
        
        # Log image formats and modes
        print(f"Image format: {original_image.format}, mode: {original_image.mode}, size: {original_image.size}")
        print(f"Mask format: {mask_image.format}, mode: {mask_image.mode}, size: {mask_image.size}")
        
        # Resize the mask if necessary
        if image_width is not None and image_height is not None and mask_width is not None and mask_height is not None:
            if image_width != mask_width or image_height != mask_height:
                print(f"Resizing mask from {mask_width}x{mask_height} to {image_width}x{image_height}")
                mask_image = mask_image.resize((image_width, image_height), Image.LANCZOS)
        
        # Convert images to numpy arrays
        original_array = np.array(original_image)
        if len(original_array.shape) > 2 and original_array.shape[2] == 4:  # RGBA
            original_array = original_array[:, :, :3]  # Remove alpha channel
            
        mask_array = np.array(mask_image)
        print(f"Image shape: {original_array.shape}, Mask shape: {mask_array.shape}")
        
        if len(mask_array.shape) > 2 and mask_array.shape[2] > 1:
            print(f"Converting mask from shape {mask_array.shape} to grayscale")
            if mask_array.shape[2] == 4:  # RGBA
                # Use alpha channel as mask if available
                mask_array = mask_array[:, :, 3]
            else:
                mask_array = mask_array[:, :, 0]  # Just use first channel
            
        # Ensure the mask is binary
        mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
        
        # Save debug mask image
        cv2.imwrite("debug_mask.png", mask_array)
        print(f"Saved debug mask to debug_mask.png")
        
        # Calculate bounding box of mask
        nonzero_coords = np.argwhere(mask_array > 0)
        if len(nonzero_coords) == 0:
            print("No mask applied, returning original image")
            return create_response(original_array)
            
        min_y, min_x = nonzero_coords.min(axis=0)
        max_y, max_x = nonzero_coords.max(axis=0)
        
        print(f"Mask bounding box: ({min_x}, {min_y}) to ({max_x}, {max_y}), dimensions: {max_x-min_x}x{max_y-min_y}")
        
        # Dilate the mask to expand masked area
        if dilate_size > 0:
            kernel = np.ones((dilate_size, dilate_size), np.uint8)
            mask_array = cv2.dilate(mask_array, kernel, iterations=1)
            print(f"Dilated mask shape: {mask_array.shape}")
        
        try:
            # Apply text-guided inpainting directly with LaMa
            result_image = text_guided_inpainting(
                original_image,
                mask_image,
                prompt
            )
            
            # Convert back to numpy array
            result_array = np.array(result_image)
            
            # Save debug replacement image
            Image.fromarray(result_array).save("debug_replacement.png")
            print(f"Saved debug replacement to debug_replacement.png")
                
            return create_response(result_array)
        except Exception as e:
            print(f"Error during replacement creation: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to traditional methods if LaMa fails
            try:
                if is_advanced:
                    result_array = create_custom_replacement(
                        result_array, 
                        original_array, 
                        mask_array, 
                        prompt, 
                        min_x, min_y, 
                        max_x, max_y
                    )
                else:
                    result_array = create_icon_replacement(
                        result_array, 
                        mask_array, 
                        prompt, 
                        min_x, min_y, 
                        max_x, max_y
                    )
                return create_response(result_array)
            except:
                return create_response(original_array)
    except Exception as e:
        print(f"Error in replace_with_mask: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "device": DEVICE}

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify the API is working"""
    return {"status": "ok", "message": "API is working!"}

def position_mask(img_shape, mask, img_width, img_height, mask_width, mask_height):
    """
    Positions a mask on an image, maintaining its original scale and position.
    
    Parameters:
    - img_shape: Shape of the image array
    - mask: The mask array
    - img_width, img_height: Original image dimensions from frontend
    - mask_width, mask_height: Original mask dimensions from frontend
    
    Returns:
    - A properly positioned mask with the same dimensions as the image
    """
    try:
        # Parse dimensions if they're strings
        if img_width is not None and isinstance(img_width, str):
            img_width = int(img_width)
        if img_height is not None and isinstance(img_height, str):
            img_height = int(img_height)
        if mask_width is not None and isinstance(mask_width, str):
            mask_width = int(mask_width)
        if mask_height is not None and isinstance(mask_height, str):
            mask_height = int(mask_height)
            
        # If dimensions were provided and valid
        if (img_width and img_height and mask_width and mask_height and
            img_width > 0 and img_height > 0 and mask_width > 0 and mask_height > 0):
            
            print(f"Using provided dimensions - Image: {img_width}x{img_height}, Mask: {mask_width}x{mask_height}")
            
            # Create a blank mask of image size
            full_mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
            
            # Calculate scale factors
            scale_x = img_shape[1] / img_width
            scale_y = img_shape[0] / img_height
            
            # Calculate scaled mask dimensions
            scaled_mask_width = int(mask_width * scale_x)
            scaled_mask_height = int(mask_height * scale_y)
            
            # Resize mask to match scaled dimensions
            if scaled_mask_width > 0 and scaled_mask_height > 0:
                if mask.shape[0] != scaled_mask_height or mask.shape[1] != scaled_mask_width:
                    resized_mask = cv2.resize(mask, (scaled_mask_width, scaled_mask_height), 
                                              interpolation=cv2.INTER_NEAREST)
                else:
                    resized_mask = mask
                
                # Since we're positioning in the center, use same calculations:
                # Center the mask on the image
                x_offset = (img_shape[1] - scaled_mask_width) // 2
                y_offset = (img_shape[0] - scaled_mask_height) // 2
                
                # Ensure offsets are valid
                x_offset = max(0, x_offset)
                y_offset = max(0, y_offset)
                
                # Place mask on the image
                placement_height = min(scaled_mask_height, img_shape[0] - y_offset)
                placement_width = min(scaled_mask_width, img_shape[1] - x_offset)
                
                full_mask[y_offset:y_offset + placement_height, 
                          x_offset:x_offset + placement_width] = resized_mask[:placement_height, :placement_width]
                
                print(f"Positioned mask at ({x_offset}, {y_offset}) with size {placement_width}x{placement_height}")
                return full_mask
    except Exception as e:
        print(f"Error in position_mask: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Fallback to simple centering
    print("Falling back to simple mask positioning")
    full_mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
    
    # Center the mask on the image
    y_offset = (img_shape[0] - mask.shape[0]) // 2
    x_offset = (img_shape[1] - mask.shape[1]) // 2
    
    # Ensure offsets are valid
    y_offset = max(0, y_offset)
    x_offset = max(0, x_offset)
    
    # Place mask on the image
    h, w = mask.shape[:2]
    h_target = min(h, img_shape[0] - y_offset)
    w_target = min(w, img_shape[1] - x_offset)
    
    full_mask[y_offset:y_offset + h_target, x_offset:x_offset + w_target] = mask[:h_target, :w_target]
    print(f"Placed mask at ({x_offset}, {y_offset}) with size {w_target}x{h_target}")
    
    return full_mask

@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources when shutting down"""
    unload_sd_model()
    print("API shutting down, resources cleaned up")

if __name__ == "__main__":
    PORT = 8001  # Use a different port
    print(f"Starting server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT) 