'use client';

import { useState } from 'react';
import ImageEditor from '@/components/ImageEditor';
import { TrashIcon, SparklesIcon, ArrowPathIcon, PencilSquareIcon } from '@heroicons/react/24/outline';

export default function Home() {
  const [image, setImage] = useState<string | null>(null);
  const [activeTool, setActiveTool] = useState<'customReplace' | 'remove' | 'replace' | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleReset = () => {
    setImage(null);
    setActiveTool(null);
  };

  return (
    <main className="container mx-auto px-4 py-8">
      <header className="text-center mb-10">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">AI Image Editor</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Transform your images with AI. Remove objects, replace elements, or create custom edits with ease.
        </p>
      </header>

      {!image ? (
        <div 
          className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
            isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => document.getElementById('fileInput')?.click()}
        >
          <input
            type="file"
            id="fileInput"
            className="hidden"
            accept="image/*"
            onChange={handleFileChange}
          />
          <svg
            className="mx-auto h-12 w-12 text-gray-400"
            stroke="currentColor"
            fill="none"
            viewBox="0 0 48 48"
            aria-hidden="true"
          >
            <path
              d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <p className="mt-2 text-sm text-gray-500">
            Click to upload an image or drag and drop
          </p>
          <p className="mt-1 text-xs text-gray-400">
            PNG, JPG, GIF up to 10MB
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="md:col-span-1 space-y-4">
            <h2 className="text-xl font-medium text-gray-800 mb-4">Tools</h2>
            
            <button
              onClick={() => setActiveTool('remove')}
              className={`flex items-center space-x-2 w-full px-4 py-3 rounded-lg ${
                activeTool === 'remove' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-white text-gray-700 hover:bg-gray-100'
              }`}
            >
              <TrashIcon className="h-5 w-5" />
              <span>Remove Object</span>
            </button>

           

            <button
              onClick={() => setActiveTool('replace')}
              className={`flex items-center space-x-2 w-full px-4 py-3 rounded-lg ${
                activeTool === 'replace' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-white text-gray-700 hover:bg-gray-100'
              }`}
            >
              <ArrowPathIcon className="h-5 w-5" />
              <span>Replace Object</span>
            </button>

            <button
              onClick={() => setActiveTool('customReplace')}
              className={`flex items-center space-x-2 w-full px-4 py-3 rounded-lg ${
                activeTool === 'customReplace' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-white text-gray-700 hover:bg-gray-100'
              }`}
            >
              <PencilSquareIcon className="h-5 w-5" />
              <span>Enhance Image</span>
            </button>
          </div>
          
          <div className="md:col-span-3">
            <ImageEditor 
              image={image} 
              activeTool={activeTool} 
              onReset={handleReset} 
            />
          </div>
        </div>
      )}
    </main>
  );
}
