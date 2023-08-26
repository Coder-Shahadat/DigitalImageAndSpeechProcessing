import cv2
import matplotlib.pyplot as plt
import numpy as np

image='new.jpg'
inputImage=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
hist=cv2.calcHist([inputImage],[0],None,[256],[0,256])


# low pass filter 
lowPass=cv2.GaussianBlur(inputImage,(3,3),0)

#High pass filter
kernal=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
highPass=cv2.filter2D(inputImage,-1,kernal)

plt.figure(figsize=(12,10))

plt.subplot(221)
plt.xlabel('Pixels')
plt.ylabel('Frequency')
plt.plot(hist)

plt.subplot(222)
plt.imshow(inputImage,cmap='gray')
plt.axis('off')
plt.title('Orginal Image')

plt.subplot(223)
plt.imshow(lowPass,cmap='gray')
plt.axis('off')
plt.title('Low Pass Filtered Image')

plt.subplot(224)
plt.imshow(highPass,cmap='gray')
plt.title('High Pass Filtered Image')
plt.axis('off')
plt.tight_layout()
plt.show()