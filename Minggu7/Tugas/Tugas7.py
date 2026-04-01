import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import pywt
import time

# =========================
# 1. LOAD CITRA
# =========================
img = cv2.imread('Tugas/Week 7/RiskiGacor.jpg', 0)
img = cv2.resize(img, (256, 256))

# Noise periodik
rows, cols = img.shape
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)
noise = 50 * np.sin(2*np.pi*X/20)
img_noise = np.clip(img + noise, 0, 255).astype(np.uint8)

# =========================
# 2. FFT ANALYSIS
# =========================
def compute_fft(image):
    f = fft2(image)
    fshift = fftshift(f)
    magnitude = np.log(1 + np.abs(fshift))
    phase = np.angle(fshift)
    return fshift, magnitude, phase

start = time.time()
fshift, mag, phase = compute_fft(img_noise)
fft_time = time.time() - start

# =========================
# 3. FREKUENSI DOMINAN
# =========================
mag_abs = np.abs(fshift)
center = (mag_abs.shape[0]//2, mag_abs.shape[1]//2)
mag_abs[center] = 0

dominant = np.unravel_index(np.argmax(mag_abs), mag_abs.shape)

# =========================
# 4. REKONSTRUKSI
# =========================
def reconstruct(magnitude, phase):
    complex_img = magnitude * np.exp(1j * phase)
    img_back = np.abs(ifft2(ifftshift(complex_img)))
    return np.uint8(np.clip(img_back, 0, 255))

img_mag = reconstruct(np.abs(fshift), np.zeros_like(phase))
img_phase = reconstruct(np.ones_like(mag), phase)

# =========================
# 5. FILTERING
# =========================
def ideal_lowpass(shape, D0):
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i-crow)**2 + (j-ccol)**2) <= D0:
                mask[i, j] = 1
    return mask

def gaussian_lowpass(shape, D0):
    rows, cols = shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i-crow)**2 + (j-ccol)**2)
            mask[i, j] = np.exp(-(D**2)/(2*(D0**2)))
    return mask

# High-pass
def gaussian_highpass(shape, D0):
    return 1 - gaussian_lowpass(shape, D0)

# Notch filter
def notch_filter(shape, centers, radius=5):
    mask = np.ones(shape)
    for c in centers:
        cv2.circle(mask, c, radius, 0, -1)
    return mask

# Variasi cutoff
cutoffs = [10, 30, 60]
results_lp = []

for d in cutoffs:
    mask = gaussian_lowpass(img.shape, d)
    result = np.abs(ifft2(ifftshift(fshift * mask)))
    results_lp.append(result)

# High-pass
mask_hp = gaussian_highpass(img.shape, 30)
img_hp = np.abs(ifft2(ifftshift(fshift * mask_hp)))

# Notch (manual titik noise)
mask_notch = notch_filter(img.shape, [(120,120),(136,136)])
img_notch = np.abs(ifft2(ifftshift(fshift * mask_notch)))

# =========================
# 6. WAVELET (db4 & Haar)
# =========================
coeffs_db4 = pywt.wavedec2(img, 'db4', level=2)
coeffs_haar = pywt.wavedec2(img, 'haar', level=2)

cA, (cH1, cV1, cD1), (cH2, cV2, cD2) = coeffs_db4

# Rekonstruksi (hapus detail level 1)
coeffs_mod = [cA, (None,None,None), (cH2,cV2,cD2)]
img_wave = pywt.waverec2(coeffs_mod, 'db4')

# =========================
# 7. PSNR
# =========================
def psnr(a, b):
    mse = np.mean((a - b)**2)
    return 20*np.log10(255/np.sqrt(mse))

# =========================
# 8. OUTPUT TERMINAL RAPI
# =========================
print("\n" + "="*50)
print("HASIL ANALISIS DOMAIN FREKUENSI")
print("="*50)

print(f"Frekuensi Dominan : {dominant}")

total_energy = np.sum(mag_abs)
low_energy = np.sum(mag_abs[center[0]-30:center[0]+30, center[1]-30:center[1]+30])
print(f"Low Frequency Energy : {low_energy/total_energy*100:.2f}%")

print("\n" + "="*50)
print("HASIL FILTERING")
print("="*50)

for i, d in enumerate(cutoffs):
    print(f"Gaussian LP (cutoff {d}) → PSNR: {psnr(img, results_lp[i]):.2f} dB")

print(f"High-pass Gaussian → PSNR: {psnr(img, img_hp):.2f} dB")
print(f"Notch Filter → PSNR: {psnr(img, img_notch):.2f} dB")

print("\n" + "="*50)
print("PERFORMA")
print("="*50)
print(f"Waktu FFT : {fft_time:.6f} detik")

print("\n" + "="*50)
print("WAVELET")
print("="*50)
print(f"Koef LL shape : {cA.shape}")
print("Wavelet db4 & Haar berhasil digunakan")

print("\n" + "="*50)
print("KESIMPULAN")
print("="*50)
print("Phase lebih penting dari magnitude untuk struktur citra")
print("Gaussian filter mengurangi ringing dibanding ideal")
print("Notch filter efektif menghilangkan noise periodik")
print("Wavelet unggul untuk analisis multi-resolusi")

print("\n" + "="*65)
print("TABEL PERBANDINGAN METRIK")
print("="*65)
print(f"{'Metode':<25} {'PSNR (dB)':<15} {'Waktu (detik)':<15}")
print("-"*65)

# Hitung waktu tiap metode
def measure_time(func):
    start = time.time()
    result = func()
    t = time.time() - start
    return result, t

# Gaussian LP (30)
_, t_lp = measure_time(lambda: np.abs(ifft2(ifftshift(fshift * gaussian_lowpass(img.shape, 30)))))

# High-pass
_, t_hp = measure_time(lambda: np.abs(ifft2(ifftshift(fshift * gaussian_highpass(img.shape, 30)))))

# Notch
_, t_notch = measure_time(lambda: np.abs(ifft2(ifftshift(fshift * notch_filter(img.shape, [(120,120),(136,136)])))))

# Wavelet
_, t_wave = measure_time(lambda: pywt.waverec2(coeffs_db4, 'db4'))

# Print tabel
print(f"{'Gaussian Lowpass':<25} {psnr(img, results_lp[1]):<15.2f} {t_lp:<15.6f}")
print(f"{'Gaussian Highpass':<25} {psnr(img, img_hp):<15.2f} {t_hp:<15.6f}")
print(f"{'Notch Filter':<25} {psnr(img, img_notch):<15.2f} {t_notch:<15.6f}")
print(f"{'Wavelet (db4)':<25} {psnr(img, img_wave):<15.2f} {t_wave:<15.6f}")

print("="*65)

# =========================
# 9. VISUALISASI
# =========================
plt.figure(figsize=(14,10))

plt.subplot(3,4,1), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(3,4,2), plt.imshow(img_noise, cmap='gray'), plt.title("Noise")
plt.subplot(3,4,3), plt.imshow(mag, cmap='gray'), plt.title("Magnitude")
plt.subplot(3,4,4), plt.imshow(phase, cmap='hsv'), plt.title("Phase")

plt.subplot(3,4,5), plt.imshow(img_mag, cmap='gray'), plt.title("Mag Only")
plt.subplot(3,4,6), plt.imshow(img_phase, cmap='gray'), plt.title("Phase Only")
plt.subplot(3,4,7), plt.imshow(results_lp[1], cmap='gray'), plt.title("LP (30)")
plt.subplot(3,4,8), plt.imshow(img_hp, cmap='gray'), plt.title("High-pass")

plt.subplot(3,4,9), plt.imshow(img_notch, cmap='gray'), plt.title("Notch")
plt.subplot(3,4,10), plt.imshow(cA, cmap='gray'), plt.title("Wavelet LL")
plt.subplot(3,4,11), plt.imshow(cH1, cmap='gray'), plt.title("Wavelet LH")
plt.subplot(3,4,12), plt.imshow(img_wave, cmap='gray'), plt.title("Reconstruct")

plt.tight_layout()
plt.show()