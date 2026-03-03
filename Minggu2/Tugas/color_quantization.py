import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans

# ==============================
# FUNGSI BANTUAN
# ==============================

def uniform_quantization(img, levels=16):
    step = 256 // levels
    return (img // step) * step

def non_uniform_quantization_gray(img, k=16):
    data = img.reshape((-1, 1))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_.astype("uint8")
    return centers[labels].reshape(img.shape)

def memory_size(img):
    return img.size * img.itemsize

def show_image(title, img, gray=False):
    plt.figure(figsize=(4,4))
    if gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# ==============================
# LOAD GAMBAR
# ==============================

images = {
    "Terang": cv2.imread("Tugas/Week 2/Objek_terang.jpg"),
    "Normal": cv2.imread("Tugas/Week 2/Objek_normal.jpg"),
    "Redup": cv2.imread("Tugas/Week 2/Objek_redup.jpg")
}

for key, img in images.items():
    if img is None:
        print(f"Gambar {key} tidak ditemukan!")
        exit()

# ==============================
# PROSES SETIAP KONDISI
# ==============================

for kondisi, img in images.items():
    print(f"\n=== KONDISI CAHAYA: {kondisi.upper()} ===")

    h, w, c = img.shape
    print(f"Resolusi      : {w} x {h}")
    print(f"Aspect Ratio  : {w/h:.2f}")
    print(f"Channels      : {c}")

    # ==============================
    # KONVERSI WARNA
    # ==============================

    t0 = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t_gray = time.time() - t0

    t0 = time.time()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    t_hsv = time.time() - t0

    t0 = time.time()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    t_lab = time.time() - t0

    print(f"Waktu konversi Gray : {t_gray:.6f} s")
    print(f"Waktu konversi HSV  : {t_hsv:.6f} s")
    print(f"Waktu konversi LAB  : {t_lab:.6f} s")

    # ==============================
    # KUANTISASI
    # ==============================

    gray_uq = uniform_quantization(gray, 16)
    gray_nuq = non_uniform_quantization_gray(gray, 16)

    # ==============================
    # MEMORI & KOMPRESI
    # ==============================

    mem_original = memory_size(gray)
    mem_quant = memory_size(gray_uq)

    print(f"Memori asli        : {mem_original/1024:.2f} KB")
    print(f"Memori kuantisasi  : {mem_quant/1024:.2f} KB")
    print(f"Rasio kompresi     : {mem_original/mem_quant:.2f}")

    # ==============================
    # MATRKS & VEKTOR
    # ==============================

    print("Matriks 5x5:")
    print(gray[:5, :5])

    print("Vektor 25 elemen:")
    print(gray.flatten()[:25])

    # ==============================
    # VISUALISASI
    # ==============================

    show_image(f"Asli - {kondisi}", img)
    show_image(f"Grayscale - {kondisi}", gray, gray=True)
    show_image(f"Uniform Quantization - {kondisi}", gray_uq, gray=True)
    show_image(f"Non-Uniform Quantization - {kondisi}", gray_nuq, gray=True)

    # ==============================
    # HISTOGRAM
    # ==============================

    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.hist(gray.flatten(), bins=256)
    plt.title("Histogram Asli")

    plt.subplot(1,3,2)
    plt.hist(gray_uq.flatten(), bins=16)
    plt.title("Histogram Uniform")

    plt.subplot(1,3,3)
    plt.hist(gray_nuq.flatten(), bins=16)
    plt.title("Histogram Non-Uniform")

    plt.tight_layout()
    plt.show()

print("\n=== PROGRAM SELESAI ===")