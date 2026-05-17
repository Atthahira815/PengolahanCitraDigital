import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================================
# FUNGSI EKSTRAKSI FITUR
# =========================================
def extract_features(image_path):

    # Load image
    img = cv2.imread(image_path)

    if img is None:
        return None

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold
    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Cari contour
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    # Ambil contour terbesar
    cnt = max(contours, key=cv2.contourArea)

    # =========================================
    # REGION PROPERTIES
    # =========================================
    area = cv2.contourArea(cnt)

    perimeter = cv2.arcLength(cnt, True)

    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = float(w) / h

    rect_area = w * h
    extent = float(area) / rect_area

    hull = cv2.convexHull(cnt)

    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        solidity = 0
    else:
        solidity = float(area) / hull_area

    # =========================================
    # MOMENTS
    # =========================================
    M = cv2.moments(cnt)

    hu = cv2.HuMoments(M)

    # Log transform biar stabil
    for i in range(7):

        if hu[i] != 0:
            hu[i] = -1 * np.sign(hu[i]) * np.log10(abs(hu[i]))

    # =========================================
    # FOURIER DESCRIPTORS
    # =========================================
    contour_array = cnt[:, 0, :]

    contour_complex = np.empty(contour_array.shape[0], dtype=complex)

    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]

    fourier_result = np.fft.fft(contour_complex)

    # Ambil descriptor frekuensi rendah
    fourier_descriptor = np.abs(fourier_result[:10])

    # Normalisasi Fourier descriptor
    fourier_descriptor = fourier_descriptor / fourier_descriptor[0]

    # =========================================
    # FEATURE VECTOR
    # =========================================
    feature_vector = [
        area,
        perimeter,
        aspect_ratio,
        extent,
        solidity,
        hu[0][0],
        hu[1][0],
        hu[2][0]
    ]

    # Tambah Fourier descriptor
    feature_vector.extend(fourier_descriptor.tolist())

    return feature_vector


# =========================================
# LOAD DATASET
# =========================================
dataset_path = r"c:\College\Semester 4\Pengolahan Citra Digital\Tugas\Week11\dataset"

classes = ["Apel", "Pisang", "Jeruk"]

X = []
y = []

for label in classes:

    folder = os.path.join(dataset_path, label)

    for file in os.listdir(folder):

        image_path = os.path.join(folder, file)

        features = extract_features(image_path)

        if features is not None:

            X.append(features)
            y.append(label)

# Convert ke numpy array
X = np.array(X)
y = np.array(y)

# =========================================
# SPLIT DATA TRAINING & TESTING
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# =========================================
# KNN CLASSIFIER
# =========================================
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

# Prediksi
y_pred = knn.predict(X_test)

# =========================================
# EVALUASI
# =========================================
accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print("=== HASIL KLASIFIKASI ===")
print("Accuracy :", accuracy)

print("\n=== CONFUSION MATRIX ===")
print(cm)

# =========================================
# VISUALISASI SHAPE ANALYSIS
# =========================================
import matplotlib.pyplot as plt

sample_path = os.path.join(
    dataset_path,
    "apel",
    "sample1.jpg"
)

sample = cv2.imread(sample_path)

sample_rgb = cv2.cvtColor(
    sample,
    cv2.COLOR_BGR2RGB
)

gray = cv2.cvtColor(
    sample,
    cv2.COLOR_BGR2GRAY
)

blur = cv2.GaussianBlur(gray, (5,5), 0)

_, thresh = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

cnt = max(contours, key=cv2.contourArea)

# =========================================
# CONTOUR
# =========================================
contour_img = sample_rgb.copy()

cv2.drawContours(
    contour_img,
    [cnt],
    -1,
    (0,255,0),
    2
)

# =========================================
# CONVEX HULL
# =========================================
hull = cv2.convexHull(cnt)

hull_img = sample_rgb.copy()

cv2.drawContours(
    hull_img,
    [hull],
    -1,
    (255,0,0),
    2
)

# =========================================
# POLYGON APPROXIMATION
# =========================================
epsilon = 0.02 * cv2.arcLength(cnt, True)

approx = cv2.approxPolyDP(
    cnt,
    epsilon,
    True
)

poly_img = sample_rgb.copy()

cv2.drawContours(
    poly_img,
    [approx],
    -1,
    (255,255,0),
    2
)

# =========================================
# CENTROID
# =========================================
M = cv2.moments(cnt)

cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])

centroid_img = sample_rgb.copy()

cv2.circle(
    centroid_img,
    (cx, cy),
    8,
    (255,0,0),
    -1
)

# =========================================
# FOURIER BOUNDARY
# =========================================
fourier_img = np.zeros(
    (300,300),
    dtype=np.uint8
)

contour_array = cnt[:,0,:]

for point in contour_array:

    x, y = point

    x = int(x * 300 / sample.shape[1])
    y = int(y * 300 / sample.shape[0])

    cv2.circle(
        fourier_img,
        (x,y),
        1,
        255,
        -1
    )

# =========================================
# DISPLAY SHAPE ANALYSIS
# =========================================
plt.figure(figsize=(15,10))

# ORIGINAL
plt.subplot(2,3,1)

plt.imshow(sample_rgb)

plt.title("Original Image")

plt.axis('off')

# THRESHOLD
plt.subplot(2,3,2)

plt.imshow(thresh, cmap='gray')

plt.title("Threshold")

plt.axis('off')

# CONTOUR
plt.subplot(2,3,3)

plt.imshow(contour_img)

plt.title("Contour Detection")

plt.axis('off')

# CONVEX HULL
plt.subplot(2,3,4)

plt.imshow(hull_img)

plt.title("Convex Hull")

plt.axis('off')

# POLYGON
plt.subplot(2,3,5)

plt.imshow(poly_img)

plt.title("Polygon Approximation")

plt.axis('off')

# CENTROID
plt.subplot(2,3,6)

plt.imshow(centroid_img)

plt.title("Centroid")

plt.axis('off')

plt.tight_layout()

plt.show()

# =========================================
# DISPLAY FOURIER DESCRIPTOR
# =========================================
plt.figure(figsize=(6,6))

plt.imshow(fourier_img, cmap='gray')

plt.title("Fourier Descriptor Boundary")

plt.axis('off')

plt.show()

# =========================================
# DISPLAY CLASSIFICATION RESULT
# =========================================
plt.figure(figsize=(7,4))

plt.axis('off')

plt.text(
    0.05,
    0.85,
    "KNN Classification Result",
    fontsize=16,
    weight='bold'
)

plt.text(
    0.05,
    0.65,
    f"Accuracy : {accuracy:.2f}",
    fontsize=14
)

plt.text(
    0.05,
    0.45,
    "Feature Used:",
    fontsize=13
)

plt.text(
    0.10,
    0.30,
    "- Region Properties",
    fontsize=12
)

plt.text(
    0.10,
    0.20,
    "- Hu Moments",
    fontsize=12
)

plt.text(
    0.10,
    0.10,
    "- Fourier Descriptor",
    fontsize=12
)

plt.show()