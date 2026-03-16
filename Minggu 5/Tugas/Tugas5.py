import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim


print("EVALUASI SPATIAL FILTERING UNTUK RESTORASI CITRA")
print("=" * 60)


# ================================
# LOAD / CREATE IMAGE
# ================================

img = cv2.imread("Tugas/Week 5/RiskiGacor.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    img = np.zeros((256,256), dtype=np.uint8)
    cv2.rectangle(img,(50,50),(200,200),180,-1)
    cv2.circle(img,(128,128),40,255,-1)

clean = img.copy()


# ================================
# NOISE FUNCTIONS
# ================================

def gaussian_noise(image, sigma=20):
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy,0,255).astype(np.uint8)


def salt_pepper_noise(image, prob=0.05):
    noisy = image.copy()
    rand = np.random.random(image.shape)

    noisy[rand < prob/2] = 0
    noisy[rand > 1 - prob/2] = 255

    return noisy


def speckle_noise(image):
    noise = np.random.randn(*image.shape)
    noisy = image + image * noise * 0.2
    return np.clip(noisy,0,255).astype(np.uint8)


# ================================
# FILTER DEFINITIONS
# ================================

filters = {
    "Mean 3x3": lambda x: cv2.blur(x,(3,3)),
    "Mean 5x5": lambda x: cv2.blur(x,(5,5)),
    "Gaussian σ1": lambda x: cv2.GaussianBlur(x,(5,5),1),
    "Gaussian σ2": lambda x: cv2.GaussianBlur(x,(7,7),2),
    "Median 3x3": lambda x: cv2.medianBlur(x,3),
    "Median 5x5": lambda x: cv2.medianBlur(x,5),
    "Min Filter": lambda x: cv2.erode(x,np.ones((3,3)))
}


# ================================
# METRICS
# ================================

def calculate_metrics(original, filtered):

    mse = np.mean((original.astype(float) - filtered.astype(float))**2)

    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10*np.log10((255**2)/mse)

    ssim_value = ssim(original, filtered)

    return mse, psnr, ssim_value


# ================================
# EVALUATION FUNCTION
# ================================

def evaluate_filters(noisy_img, noise_name):

    print(f"\nPENGUJIAN FILTER UNTUK {noise_name.upper()}")
    print("-" * 60)

    results = []

    fig, axes = plt.subplots(2,4, figsize=(16,8))
    axes = axes.ravel()

    axes[0].imshow(noisy_img, cmap='gray')
    axes[0].set_title(f"Noisy Image\n({noise_name})")
    axes[0].axis('off')

    for i,(name,f) in enumerate(filters.items()):

        start = time.time()

        filtered = f(noisy_img)

        end = time.time()

        runtime = end-start

        mse, psnr, ssim_value = calculate_metrics(clean, filtered)

        results.append([name,mse,psnr,ssim_value,runtime])

        axes[i+1].imshow(filtered,cmap='gray')
        axes[i+1].set_title(f"{name}\nPSNR:{psnr:.2f} SSIM:{ssim_value:.3f}")
        axes[i+1].axis('off')


    plt.tight_layout()
    plt.show()


    print(f"{'Filter':<15} {'MSE':<12} {'PSNR(dB)':<12} {'SSIM':<10} {'Time(s)':<10}")
    print("-"*60)

    for r in results:
        print(f"{r[0]:<15} {r[1]:<12.2f} {r[2]:<12.2f} {r[3]:<10.3f} {r[4]:<10.5f}")

    return results


# ================================
# GENERATE NOISE
# ================================

gaussian_img = gaussian_noise(clean)
sp_img = salt_pepper_noise(clean)
speckle_img = speckle_noise(clean)


# ================================
# RUN TESTS
# ================================

gaussian_results = evaluate_filters(gaussian_img,"Gaussian Noise")
sp_results = evaluate_filters(sp_img,"Salt & Pepper Noise")
speckle_results = evaluate_filters(speckle_img,"Speckle Noise")


print("\nANALISIS SINGKAT")
print("-"*60)
print("1. Gaussian filter biasanya memberikan hasil terbaik untuk Gaussian noise.")
print("2. Median filter paling efektif untuk salt-and-pepper noise.")
print("3. Mean atau Gaussian filter cukup efektif untuk speckle noise.")
print("4. Kernel yang lebih besar mengurangi noise lebih baik tetapi meningkatkan blur.")