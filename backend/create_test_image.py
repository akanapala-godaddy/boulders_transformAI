from PIL import Image, ImageDraw
import numpy as np

# Create a 512x512 image with a gradient
width, height = 512, 512
image = Image.new('RGB', (width, height))
draw = ImageDraw.Draw(image)

# Draw a gradient
for y in range(height):
    for x in range(width):
        r = int((x / width) * 255)
        g = int((y / height) * 255)
        b = 128
        draw.point((x, y), fill=(r, g, b))

# Draw some shapes
draw.rectangle([(100, 100), (200, 200)], fill='red')
draw.ellipse([(300, 300), (400, 400)], fill='blue')

# Save the image
image.save('test_image.jpg') 