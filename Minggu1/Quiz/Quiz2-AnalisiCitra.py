import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_my_image(image_path):
    """Analyze your own image (loaded from local folder)"""

    img = cv2.imread(image_path)

    if img is None:
        print("Gambar tidak ditemukan! Cek path atau nama file.")
        return None

    analysis_results = {}

    # 1. Dimensi dan resolusi
    height, width, channels = img.shape
    resolution = width * height
    aspect_ratio = width / height

    print("=== ANALISIS CITRA ===")
    print(f"Dimensi      : {width} x {height}")
    print(f"Resolusi     : {resolution:,} pixel")
    print(f"Channels     : {channels}")
    print(f"Aspect Ratio : {aspect_ratio:.2f}")

    # 2. Konversi ke grayscale & bandingkan ukuran
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    size_color = img.size * img.dtype.itemsize
    size_gray = gray.size * gray.dtype.itemsize

    print(f"\nUkuran citra warna : {size_color / 1024:.2f} KB")
    print(f"Ukuran grayscale  : {size_gray / 1024:.2f} KB")

    # 3. Statistik grayscale
    print("\nStatistik Grayscale:")
    print(f"Min   : {gray.min()}")
    print(f"Max   : {gray.max()}")
    print(f"Mean  : {gray.mean():.2f}")
    print(f"Std   : {gray.std():.2f}")

    # 4. Histogram semua channel
    plt.figure(figsize=(12,4))
    colors = ('b', 'g', 'r')

    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0,256])
        plt.plot(hist, color=color)

    plt.title("Histogram Citra Warna")
    plt.xlabel("Intensitas")
    plt.ylabel("Frekuensi")
    plt.grid(alpha=0.3)
    plt.show()

    sample = cv2.imread("Quiz/Minggu 1/GGPatrick.jpg")

    print("\n=== PERBANDINGAN DENGAN CITRA SAMPLE ===")
    print(f"Citra pribadi  : {width} x {height}")
    print(f"Citra sample   : {sample.shape[1]} x {sample.shape[0]}")

    return analysis_results


# === PEMANGGILAN FUNGSI ===
analyze_my_image("Quiz/Minggu 1/Benda_Jahat.jpg")
