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

plt.xlabel('Pixels')
plt.ylabel('Frequency')
plt.plot(hist)
plt.show()

cv2.imshow('Orginal Image',inputImage)
cv2.imshow('Low pass Filtered Image',lowPass)
cv2.imshow('High pass Filtered Image',highPass)
cv2.waitKey(0)
cv2.destroyAllWindows()