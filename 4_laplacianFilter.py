import cv2
import matplotlib.pyplot as plt

image=cv2.imread('new.jpg',cv2.IMREAD_GRAYSCALE)

laplacianImage=cv2.Laplacian(image,cv2.CV_64F)
laplacianImage=cv2.convertScaleAbs(laplacianImage)

plt.figure(figsize=(12,10))
plt.subplot(121)
plt.imshow(image,cmap='gray')
plt.axis('off')

plt.subplot(122)
plt.imshow(laplacianImage,cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()