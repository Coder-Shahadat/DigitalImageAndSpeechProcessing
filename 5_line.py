import cv2
import matplotlib.pyplot as plt
import numpy as np

image=cv2.imread('iamgeline.png',cv2.IMREAD_GRAYSCALE)
verticalKernal=np.ones((15,1),np.uint8)
horizontalKernal=np.ones((1,15),np.uint8)

verticalLines=cv2.erode(image,verticalKernal,iterations=1)
horizontalLines=cv2.erode(image,horizontalKernal,iterations=1)

plt.figure(figsize=(12,8))

plt.subplot(131)
plt.imshow(image,cmap='gray')
plt.axis('off')
plt.title('Orginal')

plt.subplot(132)
plt.imshow(verticalLines,cmap='gray')
plt.axis('off')
plt.title('Vertical Lines')

plt.subplot(133)
plt.imshow(horizontalLines,cmap='gray')
plt.title('Horizontal Lines')
plt.axis('off')
plt.tight_layout()
plt.show()