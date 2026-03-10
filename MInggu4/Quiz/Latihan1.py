import cv2
import numpy as np
import matplotlib.pyplot as plt

print ("Implementasi histogram equalization manual tanpa menggunakan OpenCV\n")

def manual_histogram_equalization(image):
    """
    Manual implementation of histogram equalization
    """

    # 1. Hitung histogram
    hist = np.zeros(256)
    for pixel in image.flatten():
        hist[pixel] += 1

    # 2. Hitung cumulative histogram (CDF)
    cdf = hist.cumsum()

    # Normalisasi CDF
    cdf_normalized = cdf * 255 / cdf[-1]

    # 3. Transformation function
    transform = cdf_normalized.astype(np.uint8)

    # 4. Apply transformation
    equalized = transform[image]

    # 5. Return hasil
    return equalized, transform


# Load image
image = cv2.imread("Minggu4/Quiz/Alam.jpg", cv2.IMREAD_GRAYSCALE)

equalized, transform = manual_histogram_equalization(image)

# Display
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Manual Histogram Equalization")
plt.imshow(equalized, cmap='gray')
plt.axis('off')

plt.show()


print("\n=== PRAKTIKUM SELESAI ===")
