import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

# Load the image
image = cv2.imread('new.jpg', cv2.IMREAD_GRAYSCALE)

# Perform FFT
f_transform = fft2(image)
f_transform_shifted = fftshift(f_transform)

# Define filter functions
def gaussian_filter(shape, sigma):
    x, y = shape
    center_x, center_y = x // 2, y // 2
    u, v = np.meshgrid(np.arange(y) - center_y, np.arange(x) - center_x)
    return np.exp(-(u**2 + v**2) / (2 * sigma**2))

def butterworth_filter(shape, cutoff, n):
    x, y = shape
    center_x, center_y = x // 2, y // 2
    u, v = np.meshgrid(np.arange(y) - center_y, np.arange(x) - center_x)
    d = np.sqrt(u**2 + v**2)
    return 1 / (1 + (d / cutoff)**(2 * n))

# Apply filters
gaussian_sigma = 20
butterworth_cutoff = 30
butterworth_n = 2

gaussian_result = f_transform_shifted * gaussian_filter(image.shape, gaussian_sigma)
butterworth_result = f_transform_shifted * butterworth_filter(image.shape, butterworth_cutoff, butterworth_n)

# Apply Laplacian filter
laplacian_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
laplacian_filter_transformed = fft2(laplacian_filter, s=image.shape)
laplacian_result = f_transform_shifted * laplacian_filter_transformed

# Inverse FFT
gaussian_output = np.abs(ifft2(fftshift(gaussian_result)))
butterworth_output = np.abs(ifft2(fftshift(butterworth_result)))
laplacian_output = np.abs(ifft2(fftshift(laplacian_result)))

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(gaussian_output, cmap='gray')
plt.title('Gaussian Filter')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(butterworth_output, cmap='gray')
plt.title('Butterworth Filter')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(laplacian_output, cmap='gray')
plt.title('Laplacian Filter')
plt.axis('off')
plt.tight_layout()
plt.show()

# Calculate metrics
mse_gaussian = mean_squared_error(image, gaussian_output)
psnr_gaussian = peak_signal_noise_ratio(image, gaussian_output)
ssim_gaussian = structural_similarity(image, gaussian_output, data_range=image.max() - image.min())  # Specify data_range

mse_butterworth = mean_squared_error(image, butterworth_output)
psnr_butterworth = peak_signal_noise_ratio(image, butterworth_output)
ssim_butterworth = structural_similarity(image, butterworth_output, data_range=image.max() - image.min())  # Specify data_range

mse_laplacian = mean_squared_error(image, laplacian_output)
psnr_laplacian = peak_signal_noise_ratio(image, laplacian_output)
ssim_laplacian = structural_similarity(image, laplacian_output, data_range=image.max() - image.min())  # Specify data_range

# Print metrics
print("Metrics for Gaussian Filter:")
print("MSE:", mse_gaussian)
print("PSNR:", psnr_gaussian)
print("SSIM:", ssim_gaussian)
print()

print("Metrics for Butterworth Filter:")
print("MSE:", mse_butterworth)
print("PSNR:", psnr_butterworth)
print("SSIM:", ssim_butterworth)
print()

print("Metrics for Laplacian Filter:")
print("MSE:", mse_laplacian)
print("PSNR:", psnr_laplacian)
print("SSIM:", ssim_laplacian)