import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
from skimage import img_as_float
from scipy.ndimage import median_filter
from skimage.metrics import structural_similarity as ssim

# -----------------------------
# Function to load your image
# -----------------------------
def load_image(path):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale

# -----------------------------
# Add Gaussian Noise
# -----------------------------
def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# -----------------------------
# Denoising Filters
# -----------------------------
def improved_bilateral_filter(image):
    # Tuning the parameters for better edge preservation
    return cv2.bilateralFilter(image, d=9, sigmaColor=25, sigmaSpace=25)  # Adjusted d, sigmaColor, and sigmaSpace

def wavelet_denoising(image):
    image_float = img_as_float(image)
    return (denoise_wavelet(image_float, method='BayesShrink', mode='soft', wavelet_levels=4, rescale_sigma=True, channel_axis=None) * 255).astype(np.uint8)

def kalman_like_filter(image):
    return cv2.GaussianBlur(image, (3, 3), 1)

# Improved Median Filter
def improved_median_filter(image, size=7):
    # Larger size to reduce graininess but still preserve the general structure
    return median_filter(image, size=size).astype(np.uint8)

def non_local_means(image):
    return cv2.fastNlMeansDenoising(image, h=20, templateWindowSize=7, searchWindowSize=31)

# -----------------------------
# Sharpening Filter
# -----------------------------
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# -----------------------------
# PSNR Calculation
# -----------------------------
def calculate_psnr(original, compared):
    return cv2.PSNR(original, compared)

# -----------------------------
# SSIM Calculation
# -----------------------------
def calculate_ssim(original, compared):
    return ssim(original, compared)

# -----------------------------
# Display function with PSNR and SSIM below each image
# -----------------------------
def plot_all_images(images, titles, psnr_values, ssim_values, rows=2, cols=4):
    plt.figure(figsize=(16, 8))
    for i, (img, title, psnr, ssim_value) in enumerate(zip(images, titles, psnr_values, ssim_values)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(f"{title}\nPSNR: {psnr:.2f} dB\nSSIM: {ssim_value:.4f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main Function
# -----------------------------
def main():
    # Load your image (replace with your file path)
    image = load_image("dip.jpeg")

    # Add noise
    noisy = add_gaussian_noise(image)

    # Apply denoising
    bilateral = improved_bilateral_filter(noisy)
    wavelet = wavelet_denoising(noisy)
    kalman = kalman_like_filter(noisy)
    
    # Apply improved median filter
    median = improved_median_filter(noisy, size=7)
    
    non_local = non_local_means(noisy)

    # Sharpen the denoised images (e.g., non-local means)
    sharpened = sharpen_image(non_local)

    # Define titles for the images
    titles = ['Original', 'Noisy', 'Improved Bilateral Filtered', 'Wavelet Denoised', 'Gaussian Blur',
              'Median Filtered', 'Non-Local Means', 'Sharpened Output']

    # Ensure that titles, psnr_values, and ssim_values all have the same length
    images = [image, noisy, bilateral, wavelet, kalman, median, non_local, sharpened]
    
    # Calculate PSNR and SSIM for each image
    psnr_values = [
        calculate_psnr(image, image),
        calculate_psnr(image, noisy),
        calculate_psnr(image, bilateral),
        calculate_psnr(image, wavelet),
        calculate_psnr(image, kalman),
        calculate_psnr(image, median),
        calculate_psnr(image, non_local),
        calculate_psnr(image, sharpened)
    ]

    ssim_values = [
        calculate_ssim(image, image),
        calculate_ssim(image, noisy),
        calculate_ssim(image, bilateral),
        calculate_ssim(image, wavelet),
        calculate_ssim(image, kalman),
        calculate_ssim(image, median),
        calculate_ssim(image, non_local),
        calculate_ssim(image, sharpened)
    ]

    # Print PSNR and SSIM values
    print("PSNR Values:")
    for title, psnr in zip(titles, psnr_values):
        print(f"{title}: {psnr:.2f} dB")

    print("\nSSIM Values:")
    for title, ssim_value in zip(titles, ssim_values):
        print(f"{title}: {ssim_value:.4f}")

    # Plot all results with PSNR and SSIM next to each image title
    plot_all_images(images, titles, psnr_values, ssim_values)

if __name__ == "__main__":
    main()
