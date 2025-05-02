import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const image = formData.get('image') as File;
    const mask = formData.get('mask') as File;
    const prompt = formData.get('prompt') as string;
    const dilateSize = formData.get('dilate_size') as string;
    const advanced = formData.get('advanced') as string;
    
    if (!image || !mask || !prompt) {
      return NextResponse.json(
        { status: 'error', message: 'Missing required fields' },
        { status: 400 }
      );
    }
    
    // Create form data for the backend
    const backendFormData = new FormData();
    backendFormData.append('image', image);
    backendFormData.append('mask', mask);
    backendFormData.append('prompt', prompt);
    backendFormData.append('dilate_size', dilateSize || '5');
    
    // Pass image dimensions if they were provided
    const imageWidth = formData.get('image_width');
    const imageHeight = formData.get('image_height');
    const maskWidth = formData.get('mask_width');
    const maskHeight = formData.get('mask_height');
    
    if (imageWidth) backendFormData.append('image_width', imageWidth as string);
    if (imageHeight) backendFormData.append('image_height', imageHeight as string);
    if (maskWidth) backendFormData.append('mask_width', maskWidth as string);
    if (maskHeight) backendFormData.append('mask_height', maskHeight as string);
    
    // If advanced flag is present, pass it along
    if (advanced === 'true') {
      backendFormData.append('advanced', 'true');
      console.log(`Sending advanced custom replace request with prompt: "${prompt}"`);
    } else {
      console.log(`Sending mask-based replace request to backend with prompt: "${prompt}"`);
    }
    
    // Send request to Python backend
    const response = await fetch('http://localhost:8001/replace_with_mask', {
      method: 'POST',
      body: backendFormData,
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to replace object: ${errorText}`);
    }
    
    const result = await response.json();
    return NextResponse.json(result);
  } catch (error) {
    console.error('Error replacing object:', error);
    return NextResponse.json(
      { status: 'error', message: error instanceof Error ? error.message : 'Failed to replace object' },
      { status: 500 }
    );
  }
} 