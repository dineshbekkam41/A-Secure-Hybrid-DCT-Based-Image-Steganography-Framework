import urllib.request
import cv2
import numpy as np

# Create a simple test image if you can't download
def create_test_image(size=512):
    """Create a colorful test image"""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create gradient patterns
    for i in range(size):
        for j in range(size):
            image[i, j, 0] = (i * 255) // size  # Red gradient
            image[i, j, 1] = (j * 255) // size  # Green gradient
            image[i, j, 2] = ((i + j) * 255) // (2 * size)  # Blue gradient
    
    # Add some texture
    noise = np.random.randint(0, 50, (size, size, 3), dtype=np.uint8)
    image = cv2.add(image, noise)
    
    return image

# Create test images
print("Creating test images...")

for name, size in [('test1', 512), ('test2', 512), ('test3', 1024)]:
    img = create_test_image(size)
    path = f"dataset/cover_images/{name}.png"
    cv2.imwrite(path, img)
    print(f"[OK] Created {path}")

print("\n[OK] Test images created!")

