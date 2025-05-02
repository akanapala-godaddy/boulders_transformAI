import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const image = formData.get('image') as string;
    
    // Convert base64 to blob
    const base64Data = image.split(',')[1];
    const imageBlob = await fetch(`data:image/png;base64,${base64Data}`).then(res => res.blob());
    
    // Create form data for the backend
    const backendFormData = new FormData();
    backendFormData.append('file', imageBlob);
    
    // Send request to Python backend
    const response = await fetch('http://localhost:8000/enhance', {
      method: 'POST',
      body: backendFormData,
    });
    
    if (!response.ok) {
      throw new Error('Failed to enhance image');
    }
    
    const result = await response.json();
    return NextResponse.json(result);
  } catch (error) {
    console.error('Error enhancing image:', error);
    return NextResponse.json(
      { error: 'Failed to enhance image' },
      { status: 500 }
    );
  }
} 