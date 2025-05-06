# image_denoising_processing
image denoising and processing using multiple real-world filters done on jupyter notebook
# image_restoration_and_denoising_project

## Team Members
- Disha R Shetty- 4SO22CD018
- Diya Nagesh – 4SO22CD019
- Mave Lobo - 4SO22CD029

## Problem Statement
CCTV cameras used for monitoring traffic often capture low-quality, noisy images or video feeds due to varying lighting conditions, environmental factors (like fog, rain, or dirt on the lens), or camera sensor limitations. The noise in these images can affect the performance of traffic monitoring systems, such as vehicle detection, license plate recognition, or traffic flow analysis.

## Noise Removal Techniques:
### Bilateral Filtering:
Method: This filter smooths the image while keeping edges sharp by considering both spatial and intensity differences. It’s effective for removing noise while preserving important features like vehicles or road signs in CCTV footage.
Effectiveness: Great for real-time video processing where both noise removal and edge preservation are needed.

### Wavelet Denoising:
Method: Decompose the image into wavelet coefficients and threshold the coefficients to remove noise, then perform an inverse wavelet transform to reconstruct the image. This technique preserves important structures (e.g., vehicles) while reducing noise in the background.
Effectiveness: Effective for noisy frames, especially when the image contains both high and low-frequency information, such as road textures and moving cars.

### Kalman Filtering:
Method: A statistical method used to predict and correct values in noisy data. In traffic monitoring, this can be applied to smooth out noisy video data and track vehicles in real-time.
Effectiveness: Good for dynamic systems, such as tracking moving vehicles in video streams.

### Non-Local Means Denoising:
Method: This method compares similar patches of pixels throughout the image to denoise the image, which is useful when there is spatially repetitive content like road textures in traffic footage.
Effectiveness: Effective in removing noise while preserving fine details of moving objects, like vehicles.

### Deep Learning-Based Denoising:
Method: Use CNNs or autoencoders to learn noise patterns from large datasets of noisy traffic images. These models can then be used to remove noise while preserving relevant details like vehicle license plates and traffic signs.
Effectiveness: Highly effective for complex noise patterns but requires training data and computational power.

## PSNR and SSIM Metrics
To evaluate the performance of different denoising techniques used in this project, we rely on two widely used image quality assessment metrics: PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index Measure).

### PSNR (Peak Signal-to-Noise Ratio)
PSNR is a standard metric used to measure the quality of a restored or compressed image compared to the original image. It is expressed in decibels (dB). Higher PSNR values indicate better restoration quality, meaning the denoised image is more similar to the original.

**Formula:**
-PSNR = 10 * log10(MAX² / MSE)   
where:
* MAX is the maximum possible pixel value of the image (e.g., 255 for 8-bit images)
* MSE is the Mean Squared Error between the original and processed image

**Interpretation:**
* PSNR > 30 dB: Good quality
* PSNR < 20 dB: Poor quality

### SSIM (Structural Similarity Index Measure)
SSIM compares the structural information between the original and denoised images. Unlike PSNR, which considers pixel-wise differences, SSIM evaluates image similarity based on luminance, contrast, and structure.<br>
 **Range:** 0 to 1
* 1 indicates perfect similarity.
* Values closer to 1 are better.<br>
SSIM is especially useful when assessing how well details like edges and textures are preserved after noise removal.

## Example: Noise Removal in Traffic Surveillance Video
Imagine you have a CCTV feed with poor lighting and high noise due to rain. The background noise makes it difficult for vehicle detection algorithms to identify vehicles properly. Using Bilateral Filtering can smooth out the noise in the image, while preserving the edges of vehicles, allowing the system to better detect and track moving vehicles in the video.

Alternatively, Kalman Filtering can be used in a real-time traffic monitoring system to help track vehicles across frames, reducing the impact of noise from frame to frame and improving the accuracy of vehicle detection.

