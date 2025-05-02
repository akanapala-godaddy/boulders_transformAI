import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const image = formData.get('image') as File;
    const mask = formData.get('mask') as File;
    const dilateSize = formData.get('dilate_size') as string;
    
    if (!image || !mask) {
      return NextResponse.json(
        { status: 'error', message: 'Missing required fields' },
        { status: 400 }
      );
    }
    
    // Create form data for the backend
    const backendFormData = new FormData();
    backendFormData.append('image', image);
    backendFormData.append('mask', mask);
    backendFormData.append('dilate_size', dilateSize || '5');
    
    console.log(`Sending mask-based remove request to backend`);
    
    // Send request to Python backend
    const response = await fetch('http://localhost:8001/remove_with_mask', {
      method: 'POST',
      body: backendFormData,
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to remove object: ${errorText}`);
    }
    
    const result = await response.json();
    return NextResponse.json(result);
  } catch (error) {
    console.error('Error removing object:', error);
    return NextResponse.json(
      { status: 'error', message: error instanceof Error ? error.message : 'Failed to remove object' },
      { status: 500 }
    );
  }
} 