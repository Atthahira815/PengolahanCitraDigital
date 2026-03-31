import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim

# =========================
# 1. LOAD / CREATE IMAGE
# =========================
def create_test_image():
    img = np.zeros((256, 256), dtype=np.uint8)

    cv2.rectangle(img, (30, 30), (100, 100), 200, -1)
    cv2.circle(img, (180, 80), 40, 150, -1)
    cv2.putText(img, 'TEST', (80, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 180, 2)

    return img

# =========================
# 2. PSF MOTION BLUR
# =========================
def motion_psf(length=15, angle=30):
    psf = np.zeros((length, length))
    center = length // 2
    angle = np.deg2rad(angle)

    x1 = int(center - (length/2)*np.cos(angle))
    y1 = int(center - (length/2)*np.sin(angle))
    x2 = int(center + (length/2)*np.cos(angle))
    y2 = int(center + (length/2)*np.sin(angle))

    cv2.line(psf, (x1, y1), (x2, y2), 1, 1)
    psf = psf / np.sum(psf)

    return psf

# =========================
# 3. DEGRADATION
# =========================
def add_motion_blur(img, psf):
    return cv2.filter2D(img.astype(float), -1, psf)

def add_gaussian_noise(img, sigma=20):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 255)

def add_salt_pepper(img, prob=0.05):
    noisy = img.copy()
    total = img.size

    num_salt = int(total * prob / 2)
    coords = [np.random.randint(0, i, num_salt) for i in img.shape]
    noisy[coords[0], coords[1]] = 255

    num_pepper = int(total * prob / 2)
    coords = [np.random.randint(0, i, num_pepper) for i in img.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy

# =========================
# 4. RESTORATION METHODS
# =========================
def pad_psf(psf, shape):
    psf_pad = np.zeros(shape)
    center = shape[0]//2

    psf_center = psf.shape[0]//2
    y = center - psf_center
    x = center - psf_center

    psf_pad[y:y+psf.shape[0], x:x+psf.shape[1]] = psf
    return np.fft.ifftshift(psf_pad)

def inverse_filter(img, psf, eps=1e-3):
    G = np.fft.fft2(img)
    H = np.fft.fft2(pad_psf(psf, img.shape))

    F = G / (H + eps)
    result = np.abs(np.fft.ifft2(F))

    return np.clip(result, 0, 255)

def wiener_filter(img, psf, K=0.01):
    G = np.fft.fft2(img)
    H = np.fft.fft2(pad_psf(psf, img.shape))

    H_conj = np.conj(H)
    H2 = np.abs(H)**2

    W = H_conj / (H2 + K)
    F = G * W

    result = np.abs(np.fft.ifft2(F))
    return np.clip(result, 0, 255)

def richardson_lucy(img, psf, iter=15):
    img = img.astype(np.float32)
    psf_flip = np.flip(psf)

    estimate = img.copy()

    for _ in range(iter):
        conv = cv2.filter2D(estimate, -1, psf)
        conv = np.where(conv == 0, 1e-8, conv)

        ratio = img / conv
        estimate *= cv2.filter2D(ratio, -1, psf_flip)

        estimate = np.clip(estimate, 0, 255)

    return estimate

# =========================
# 5. METRICS
# =========================
def compute_metrics(original, restored):
    mse = np.mean((original - restored)**2)
    psnr = 10 * np.log10(255**2 / mse) if mse != 0 else 100
    ssim_val = ssim(original.astype(np.uint8),
                    restored.astype(np.uint8))
    return mse, psnr, ssim_val

def show_spectrum(img, title):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    spectrum = np.log(1 + np.abs(fshift))

    plt.imshow(spectrum, cmap='gray')
    plt.title(title)
    plt.axis('off')

# =========================
# 6. PIPELINE
# =========================
def run_pipeline():
    original = create_test_image()
    psf = motion_psf(15, 30)

    # 3 degradasi
    blur = add_motion_blur(original, psf)

    blur_gauss = add_motion_blur(original, psf)
    blur_gauss = add_gaussian_noise(blur_gauss, 20)

    blur_sp = add_motion_blur(original, psf)
    blur_sp = add_salt_pepper(blur_sp, 0.05)

    datasets = {
        "Motion Blur": blur,
        "Gaussian + Blur": blur_gauss,
        "S&P + Blur": blur_sp
    }

    results = {}

    for name, img in datasets.items():
        results[name] = {}

        # Inverse
        start = time.time()
        inv = inverse_filter(img, psf)
        t_inv = time.time() - start

        # Wiener
        start = time.time()
        wnr = wiener_filter(img, psf, K=0.01)
        t_wnr = time.time() - start

        # RL
        start = time.time()
        rl = richardson_lucy(img, psf, 15)
        t_rl = time.time() - start

        results[name]['Inverse'] = (inv, t_inv)
        results[name]['Wiener'] = (wnr, t_wnr)
        results[name]['RL'] = (rl, t_rl)

    # =====================
    # VISUALIZATION
    # =====================
    for name, img in datasets.items():
        plt.figure(figsize=(12,8))
        plt.suptitle(name)

        # ORIGINAL
        plt.subplot(3,3,1)
        plt.imshow(original, cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # DEGRADED
        plt.subplot(3,3,2)
        plt.imshow(img, cmap='gray')
        plt.title("Degraded")
        plt.axis('off')

        # SPECTRUM DEGRADED
        plt.subplot(3,3,3)
        show_spectrum(img, "Spectrum Degraded")

        i = 4
        for method in results[name]:
            restored, _ = results[name][method]

            # HASIL RESTORASI
            plt.subplot(3,3,i)
            plt.imshow(restored, cmap='gray')
            plt.title(method)
            plt.axis('off')

            # SPECTRUM RESTORASI
            plt.subplot(3,3,i+1)
            show_spectrum(restored, f"{method} Spectrum")

            i += 2

        plt.tight_layout()
        plt.show()

    # =====================
    # METRICS TABLE
    # =====================
    print("\nTUGAS 6 : PIPELINE RESTORASI CITRA")
    print("=" * 70)

    for name in results:
        print(f"\nKASUS: {name}")
        print("-" * 70)
        print(f"{'Metode':<20}{'PSNR (dB)':<12}{'MSE':<12}{'SSIM':<10}{'Time (s)'}")
        print("-" * 70)

        for method in results[name]:
            img, t = results[name][method]
            mse, psnr, s = compute_metrics(original, img)

            print(f"{method:<20}{psnr:<12.2f}{mse:<12.2f}{s:<10.3f}{t:.4f}")

# =========================
# RUN
# =========================
run_pipeline()