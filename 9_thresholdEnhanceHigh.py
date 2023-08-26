import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'new.jpg'  # Replace with your image file path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Threshold Image
_, threshold_img = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# Power Enhance Contrast Image
gamma = 1.5  # Adjust the gamma value for desired contrast enhancement
power_enhanced_img = np.power(image / 255.0, gamma)
power_enhanced_img = np.uint8(power_enhanced_img * 255)

# High Pass Image (using Laplacian)
laplacian = cv2.Laplacian(image, cv2.CV_64F)
high_pass_img = cv2.convertScaleAbs(laplacian)

# Display the images using matplotlib
plt.figure(figsize=(12, 10))

plt.subplot(131)
plt.imshow(threshold_img, cmap='gray')
plt.title('Threshold Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(power_enhanced_img, cmap='gray')
plt.title('Power Enhance Contrast Image')
plt.axis('off')

plt.subplot(133)
plt.imshow(high_pass_img, cmap='gray')
plt.title('High Pass Image')
plt.axis('off')
plt.tight_layout()
plt.show()