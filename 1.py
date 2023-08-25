import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('flow.jpg', cv2.IMREAD_GRAYSCALE)
# (i)
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
# (ii)
low_pass_image = cv2.GaussianBlur(image, (5, 5), 0)  # Low pass filtered image
# (iii)
kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
high_pass_image = cv2.filter2D(image,-1,kernel)

plt.subplot(311)
plt.plot(histogram)
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(312)
plt.imshow(low_pass_image)
plt.title('Low Pass Filtered Image')

plt.subplot(313)
plt.imshow(high_pass_image)
plt.title('High Pass Image')
plt.tight_layout()
plt.show()