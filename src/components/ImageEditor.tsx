'use client';

import { useState, useRef, useEffect } from 'react';
import ReactCrop, { Crop } from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';
import { ArrowUturnLeftIcon, XMarkIcon } from '@heroicons/react/24/outline';

interface ImageEditorProps {
  image: string;
  activeTool: 'customReplace' | 'remove' | 'replace' | null;
  onReset: () => void;
}

export default function ImageEditor({ image, activeTool, onReset }: ImageEditorProps) {
  const [crop, setCrop] = useState<Crop>();
  const [editedImage, setEditedImage] = useState<string | null>(null);
  const [maskImage, setMaskImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [prompt, setPrompt] = useState('');
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushSize, setBrushSize] = useState(20);
  const [eraserMode, setEraserMode] = useState(false);

  // Set up the canvas when the image loads
  useEffect(() => {
    if ((activeTool === 'remove' || activeTool === 'replace' || activeTool === 'customReplace') && imageRef.current) {
      const canvas = canvasRef.current;
      const maskCanvas = maskCanvasRef.current;
      if (!canvas || !maskCanvas) return;

      // Reset state when tool or image changes
      setEditedImage(null);
      setMaskImage(null);

      // Set canvas dimensions to match the image
      const imgWidth = imageRef.current.width;
      const imgHeight = imageRef.current.height;
      
      canvas.width = imgWidth;
      canvas.height = imgHeight;
      
      maskCanvas.width = imgWidth;
      maskCanvas.height = imgHeight;
      
      // Initialize mask canvas with transparent black
      const maskCtx = maskCanvas.getContext('2d');
      if (maskCtx) {
        maskCtx.fillStyle = 'rgba(0, 0, 0, 0)';
        maskCtx.fillRect(0, 0, imgWidth, imgHeight);
      }
    }
  }, [activeTool, image]);

  const getCoordinates = (e: React.MouseEvent<HTMLCanvasElement>): [number, number] => {
    const canvas = canvasRef.current;
    if (!canvas) return [0, 0];
    
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    
    return [x, y];
  };

  const drawOnMask = (x: number, y: number) => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) return;
    
    const ctx = maskCanvas.getContext('2d');
    if (!ctx) return;
    
    ctx.globalCompositeOperation = eraserMode ? 'destination-out' : 'source-over';
    ctx.fillStyle = 'rgba(255, 255, 255, 1)';
    ctx.beginPath();
    ctx.arc(x, y, brushSize, 0, Math.PI * 2);
    ctx.fill();
    
    // Display the mask on the main canvas
    updateMainCanvas();
  };
  
  const updateMainCanvas = () => {
    const canvas = canvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    if (!canvas || !maskCanvas || !imageRef.current) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear and draw the image
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height);
    
    // Overlay the mask
    const maskCtx = maskCanvas.getContext('2d');
    if (!maskCtx) return;
    
    // Get mask data and draw it with a semi-transparent blue overlay
    const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const overlayCanvas = document.createElement('canvas');
    overlayCanvas.width = canvas.width;
    overlayCanvas.height = canvas.height;
    const overlayCtx = overlayCanvas.getContext('2d');
    
    if (overlayCtx) {
      overlayCtx.fillStyle = 'rgba(30, 144, 255, 0.5)';
      overlayCtx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Use the mask as alpha channel
      const overlayData = overlayCtx.getImageData(0, 0, canvas.width, canvas.height);
      
      for (let i = 0; i < maskData.data.length; i += 4) {
        // Use the mask's alpha channel to determine the overlay's alpha
        overlayData.data[i + 3] = maskData.data[i + 3];
      }
      
      overlayCtx.putImageData(overlayData, 0, 0);
      ctx.drawImage(overlayCanvas, 0, 0);
    }
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (loading) return;
    setIsDrawing(true);
    const [x, y] = getCoordinates(e);
    drawOnMask(x, y);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const [x, y] = getCoordinates(e);
    drawOnMask(x, y);
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  const handleClearMask = () => {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) return;
    
    const ctx = maskCanvas.getContext('2d');
    if (!ctx) return;
    
    ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    updateMainCanvas();
  };

  const handleProcessImage = async () => {
    if (activeTool !== 'remove' && activeTool !== 'replace' && activeTool !== 'customReplace') return;
    
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) return;
    
    // Check if mask has content
    const ctx = maskCanvas.getContext('2d');
    if (!ctx) return;
    
    const maskData = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    let hasContent = false;
    for (let i = 3; i < maskData.data.length; i += 4) {
      if (maskData.data[i] > 0) {
        hasContent = true;
        break;
      }
    }
    
    if (!hasContent) {
      alert(activeTool === 'remove' 
        ? 'Please draw a mask on the areas you want to remove.'
        : 'Please draw a mask on the areas you want to replace.');
      return;
    }
    
    if ((activeTool === 'replace' || activeTool === 'customReplace') && !prompt.trim()) {
      alert('Please enter a description of what you want to replace the selected area with.');
      return;
    }
    
    setLoading(true);
    try {
      // Convert image to blob
      const response = await fetch(image);
      const imageBlob = await response.blob();
      
      // Convert mask canvas to blob
      const maskBlob = await new Promise<Blob>((resolve) => {
        maskCanvas.toBlob((blob) => {
          if (blob) resolve(blob);
          else resolve(new Blob([]));
        });
      });
      
      const formData = new FormData();
      formData.append('image', imageBlob);
      formData.append('mask', maskBlob);
      formData.append('dilate_size', '5');
      
      // Add image dimensions to help backend position the mask correctly
      if (imageRef.current) {
        formData.append('image_width', imageRef.current.naturalWidth.toString());
        formData.append('image_height', imageRef.current.naturalHeight.toString());
        formData.append('mask_width', maskCanvas.width.toString());
        formData.append('mask_height', maskCanvas.height.toString());
      }
      
      let result;
      let apiPath = '';
      
      if (activeTool === 'remove') {
        apiPath = '/api/remove';
      } else if (activeTool === 'replace' || activeTool === 'customReplace') {
        formData.append('prompt', prompt);
        // For customReplace, we'll use the same endpoint but add a parameter to indicate
        // it should use more advanced/complex replacement
        if (activeTool === 'customReplace') {
          formData.append('advanced', 'true');
        }
        apiPath = '/api/replace';
      }
      
      console.log(`Sending request to ${apiPath}${(activeTool === 'replace' || activeTool === 'customReplace') ? ` with prompt: "${prompt}"` : ''}`);
      
      result = await fetch(apiPath, {
        method: 'POST',
        body: formData,
      });

      if (!result || !result.ok) {
        const errorText = await result.text();
        throw new Error(`Failed to process image: ${errorText}`);
      }

      const data = await result.json();
      if (data.status === 'success') {
        setEditedImage(data.inpainted_image);
        setMaskImage(data.mask_visualization);
      } else {
        throw new Error(data.message || 'Failed to process image');
      }
    } catch (error) {
      console.error('Error processing image:', error);
      alert(`Failed to process image: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <button
          onClick={onReset}
          className="flex items-center space-x-2 text-gray-600 hover:text-gray-900"
        >
          <ArrowUturnLeftIcon className="h-5 w-5" />
          <span>Upload New Image</span>
        </button>
      </div>

      <div className="relative">
        {loading && (
          <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-20 rounded-lg">
            <div className="text-white">Processing image...</div>
          </div>
        )}

        {(activeTool === 'remove' || activeTool === 'replace' || activeTool === 'customReplace') ? (
          editedImage ? (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h3 className="text-sm font-medium mb-2">Original with Mask</h3>
                  <img
                    src={maskImage || image}
                    alt="Mask"
                    className="max-w-full h-auto rounded-lg"
                  />
                </div>
                <div>
                  <h3 className="text-sm font-medium mb-2">Result</h3>
                  <img
                    src={editedImage}
                    alt="Result"
                    className="max-w-full h-auto rounded-lg"
                  />
                </div>
              </div>
              <button
                onClick={() => {
                  setEditedImage(null);
                  setMaskImage(null);
                  handleClearMask();
                }}
                className="px-4 py-2 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300"
              >
                Try Again
              </button>
              <a
                href={editedImage}
                download={activeTool === 'remove' ? "removed-object.png" : 
                          activeTool === 'replace' ? "replaced-object.png" : 
                          "custom-replaced.png"}
                className="ml-2 inline-block px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
              >
                Download Result
              </a>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="relative" style={{ position: 'relative', display: 'inline-block' }}>
                <img
                  ref={imageRef}
                  src={image}
                  alt="Original"
                  className="max-w-full h-auto rounded-lg pointer-events-none"
                  style={{ display: 'block' }}
                />
                <canvas
                  ref={canvasRef}
                  className="absolute top-0 left-0 w-full h-full cursor-crosshair"
                  style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 10 }}
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onMouseLeave={handleMouseUp}
                />
                <canvas
                  ref={maskCanvasRef}
                  className="absolute top-0 left-0 w-full h-full"
                  style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', display: 'none' }}
                />
              </div>
              
              <div className="space-y-3">
                <div className="flex flex-wrap gap-2 items-center">
                  <div className="mr-2">
                    <label htmlFor="brushSize" className="block text-sm font-medium text-gray-700">
                      Brush Size: {brushSize}px
                    </label>
                    <input
                      id="brushSize"
                      type="range"
                      min="5"
                      max="50"
                      value={brushSize}
                      onChange={(e) => setBrushSize(parseInt(e.target.value))}
                      className="w-32 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                  
                  <button
                    onClick={() => setEraserMode(!eraserMode)}
                    className={`px-3 py-1 rounded-full text-sm ${
                      eraserMode 
                        ? 'bg-blue-100 text-blue-800 border border-blue-300' 
                        : 'bg-white text-gray-700 border border-gray-300'
                    }`}
                  >
                    {eraserMode ? 'Eraser Mode' : 'Draw Mode'}
                  </button>
                  
                  <button
                    onClick={handleClearMask}
                    className="px-3 py-1 rounded-full text-sm bg-gray-100 text-gray-700 border border-gray-300"
                  >
                    Clear Mask
                  </button>
                </div>
                
                <p className="text-sm text-gray-600">
                  {activeTool === 'remove' 
                    ? 'Draw over the areas you want to remove. Toggle to eraser mode to refine your selection.'
                    : activeTool === 'replace'
                    ? 'Draw over the areas you want to replace with a simple pattern. Toggle to eraser mode to refine your selection.'
                    : 'Draw over the object you want to replace or transform. Be precise with your selection for best results.'}
                </p>
              </div>
              
              {(activeTool === 'replace' || activeTool === 'customReplace') && (
                <div className="mt-2">
                  <label htmlFor="promptInput" className="block text-sm font-medium text-gray-700 mb-1">
                    {activeTool === 'customReplace' 
                      ? 'Describe What You Want To Create' 
                      : 'Replacement Description'}
                  </label>
                  <input
                    id="promptInput"
                    type="text"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder={activeTool === 'customReplace' 
                      ? "Example: Move the dog to sit on a swing instead of a bench"
                      : "Describe what you want to replace the selected area with..."}
                    className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  {activeTool === 'customReplace' && (
                    <p className="mt-1 text-xs text-gray-500">
                      For best results, be specific about what you want to change and how. For example: "Replace the dog with a cat", 
                      "Move the person from the bench to a swing", or "Change the tree to a palm tree".
                    </p>
                  )}
                </div>
              )}
              
              <button
                onClick={handleProcessImage}
                className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
              >
                {activeTool === 'remove' 
                  ? 'Remove Selected Areas' 
                  : activeTool === 'replace'
                  ? 'Replace Selected Areas'
                  : 'Transform Selection'}
              </button>
            </div>
          )
        ) : (
          // For other tools, use the original ReactCrop component
          <ReactCrop
            crop={crop}
            onChange={(c) => setCrop(c)}
            aspect={undefined}
          >
            <img
              ref={imageRef}
              src={editedImage || image}
              alt="Original"
              className="max-w-full h-auto rounded-lg"
            />
          </ReactCrop>
        )}
      </div>

      {editedImage && activeTool !== 'remove' && (
        <div className="mt-4">
          <a
            href={editedImage}
            download="edited-image.png"
            className="inline-block px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
          >
            Download Edited Image
          </a>
        </div>
      )}
    </div>
  );
} 