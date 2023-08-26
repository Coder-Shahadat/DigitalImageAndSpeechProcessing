import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'new.jpg'  # Replace with your image file path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Image Enhancement (Histogram Equalization)
enhanced_img = cv2.equalizeHist(image)

# Smoothing (Averaging)
kernel_size = 5  # Adjust the kernel size for desired smoothing
smoothed_img = cv2.blur(image, (kernel_size, kernel_size))

# Sharpening (Laplacian)
laplacian = cv2.Laplacian(image, cv2.CV_64F)
sharpened_img = cv2.convertScaleAbs(image - laplacian)

# Display the images using matplotlib
plt.figure(figsize=(12, 10))

plt.subplot(221)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(222)
plt.imshow(enhanced_img, cmap='gray')
plt.title('Enhanced Image')
plt.axis('off')

plt.subplot(223)
plt.imshow(smoothed_img, cmap='gray')
plt.title('Smoothed Image')
plt.axis('off')

plt.subplot(224)
plt.imshow(sharpened_img, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')
plt.tight_layout()
plt.show()