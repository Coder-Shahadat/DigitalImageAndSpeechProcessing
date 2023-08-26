import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path='new.jpg'
image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

kernal=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
convolvedImage=cv2.filter2D(image,-1,kernal)

plt.figure(figsize=(12,10))
plt.subplot(121)
plt.imshow(image,cmap='gray')
plt.axis('off')

plt.subplot(122)
plt.imshow(convolvedImage,cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()