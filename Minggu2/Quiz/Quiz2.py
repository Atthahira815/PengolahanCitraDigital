# ============================================
# PRAKTIKUM 2 (LANJUTAN):
# ANALYSIS OF COLOR MODELS AND IMAGE ALIASING
# ============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=== PRAKTIKUM 2 (LANJUTAN): COLOR MODEL ANALYSIS & IMAGE ALIASING ===")
print("Topics: Color Model Suitability, Image Sampling, Aliasing Effects\n")

# ======================================================
# FUNCTION 1: COLOR MODEL SUITABILITY ANALYSIS
# ======================================================
def analyze_color_model_suitability(image, application):
    """
    Analyze which color model is most suitable for a specific application.

    Parameters:
    image       : Input BGR image
    application : 'skin_detection', 'shadow_removal',
                  'text_extraction', 'object_detection'

    Returns:
    best_model  : Recommended color model
    analysis    : Dictionary of simple statistical indicators
    """

    # Convert image into different color spaces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    analysis = {}

    if application == 'skin_detection':
        # Skin color is more stable in chrominance channels
        analysis['HSV (Saturation mean)'] = np.mean(hsv[:, :, 1])
        analysis['YCbCr (Cr mean)'] = np.mean(ycbcr[:, :, 1])
        best_model = 'YCbCr'

    elif application == 'shadow_removal':
        # Shadows mainly affect luminance
        analysis['Grayscale (Std Dev)'] = np.std(gray)
        analysis['YCbCr (Y Std Dev)'] = np.std(ycbcr[:, :, 0])
        best_model = 'YCbCr'

    elif application == 'text_extraction':
        # Text relies heavily on intensity contrast
        analysis['Grayscale (Std Dev)'] = np.std(gray)
        best_model = 'Grayscale'

    elif application == 'object_detection':
        # Object detection benefits from color separation
        analysis['HSV (Channel Variance)'] = np.mean(np.std(hsv, axis=(0, 1)))
        best_model = 'HSV'

    else:
        best_model = 'Unknown'

    return best_model, analysis


# ======================================================
# FUNCTION 2: IMAGE ALIASING SIMULATION
# ======================================================
def simulate_image_aliasing(image, downsampling_factors):
    """
    Simulate aliasing effects by downsampling and upsampling an image.

    Parameters:
    image                : Input grayscale image
    downsampling_factors : List of downsampling factors (e.g., [2, 4, 8])

    Returns:
    results : Dictionary containing reconstructed images and MSE values
    """

    h, w = image.shape
    results = {}

    for factor in downsampling_factors:
        # Downsample without anti-aliasing
        small = cv2.resize(
            image,
            (w // factor, h // factor),
            interpolation=cv2.INTER_NEAREST
        )

        # Upsample back to original size
        reconstructed = cv2.resize(
            small,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        # Calculate Mean Squared Error (MSE)
        mse = np.mean(
            (image.astype(float) - reconstructed.astype(float)) ** 2
        )

        results[factor] = {
            'reconstructed': reconstructed,
            'mse': mse
        }

    return results


# ======================================================
# MAIN PROGRAM
# ======================================================
if __name__ == "__main__":

    # Load image
    img = cv2.imread('Quiz/Minggu 2/Riski gacor.jpg')  # Replace with your image path
    if img is None:
        # Create synthetic image if file not found
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 255), -1)
        cv2.circle(img, (250, 150), 60, (0, 0, 255), -1)
        print("Using synthetic image (sample_image.jpg not found)\n")

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------------
    # Color Model Analysis
    # -------------------------------
    print("1. COLOR MODEL SUITABILITY ANALYSIS")

    applications = [
        'skin_detection',
        'shadow_removal',
        'text_extraction',
        'object_detection'
    ]

    for app in applications:
        best_model, analysis = analyze_color_model_suitability(img, app)
        print(f"\nApplication: {app}")
        print(f"Recommended color model: {best_model}")
        for k, v in analysis.items():
            print(f"  {k}: {v:.2f}")

    # -------------------------------
    # Image Aliasing Simulation
    # -------------------------------
    print("\n2. IMAGE ALIASING SIMULATION")

    factors = [2, 4, 8]
    aliasing_results = simulate_image_aliasing(gray_img, factors)

    # Visualization
    fig, axes = plt.subplots(1, len(factors) + 1, figsize=(15, 5))

    axes[0].imshow(gray_img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    for idx, factor in enumerate(factors):
        axes[idx + 1].imshow(
            aliasing_results[factor]['reconstructed'],
            cmap='gray'
        )
        axes[idx + 1].set_title(
            f'Downsampling x{factor}\nMSE: {aliasing_results[factor]["mse"]:.1f}'
        )
        axes[idx + 1].axis('off')

    plt.suptitle('Aliasing Effects Due to Downsampling', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\n=== PROGRAM FINISHED ===")