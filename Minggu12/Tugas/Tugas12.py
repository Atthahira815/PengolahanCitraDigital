import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# =========================================
# DATASET
# =========================================
dataset_path = r"c:\College\Semester 4\Pengolahan Citra Digital\Tugas\Week12\dataset"

classes = [
    "buku",
    "mug",
    "botol",
    "mainan",
    "remote"
]

# =========================================
# FEATURE DETECTOR
# =========================================
# SIFT
sift = cv2.SIFT_create()

# ORB
orb = cv2.ORB_create(nfeatures=1000)

# =========================================
# LOAD IMAGE
# =========================================
def load_images():

    data = []

    for label in classes:

        folder = os.path.join(dataset_path, label)

        for file in os.listdir(folder):

            path = os.path.join(folder, file)

            img = cv2.imread(path)

            if img is None:
                continue

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            data.append((gray, label, path))

    return data

# =========================================
# EXTRACT FEATURES
# =========================================
def extract_features(detector, image):

    start = time.time()

    keypoints, descriptors = detector.detectAndCompute(
        image,
        None
    )

    elapsed = time.time() - start

    return keypoints, descriptors, elapsed

# =========================================
# FEATURE MATCHING
# =========================================
def brute_force_matching(desc1, desc2, method="SIFT"):

    if method == "ORB":

        bf = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False
        )

    else:

        bf = cv2.BFMatcher(
            cv2.NORM_L2,
            crossCheck=False
        )

    matches = bf.knnMatch(
        desc1,
        desc2,
        k=2
    )

    good_matches = []

    # Lowe Ratio Test
    for m, n in matches:

        if m.distance < 0.75 * n.distance:

            good_matches.append(m)

    return good_matches

# =========================================
# FLANN MATCHING
# =========================================
def flann_matching(desc1, desc2):

    index_params = dict(
        algorithm=1,
        trees=5
    )

    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(
        index_params,
        search_params
    )

    matches = flann.knnMatch(
        np.float32(desc1),
        np.float32(desc2),
        k=2
    )

    good_matches = []

    for m, n in matches:

        if m.distance < 0.75 * n.distance:

            good_matches.append(m)

    return good_matches

# =========================================
# HOMOGRAPHY + RANSAC
# =========================================
def homography_ransac(
    kp1,
    kp2,
    matches
):

    if len(matches) < 4:
        return None, None

    src_pts = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ]).reshape(-1,1,2)

    dst_pts = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ]).reshape(-1,1,2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    return H, mask

# =========================================
# BAG OF VISUAL WORDS
# =========================================
def build_vocabulary(
    descriptor_list,
    k=50
):

    descriptors = np.vstack(descriptor_list)

    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    kmeans.fit(descriptors)

    return kmeans

# =========================================
# HISTOGRAM VISUAL WORDS
# =========================================
def build_histogram(
    descriptors,
    kmeans
):

    prediction = kmeans.predict(descriptors)

    histogram = np.bincount(
        prediction,
        minlength=kmeans.n_clusters
    )

    return histogram

# =========================================
# LOAD DATA
# =========================================
data = load_images()

all_descriptors = []

feature_vectors = []

labels = []

# =========================================
# EXTRACT SIFT FEATURES
# =========================================
print("\n=== FEATURE EXTRACTION ===")

for image, label, path in data:

    kp, desc, elapsed = extract_features(
        sift,
        image
    )

    if desc is None:
        continue

    print(f"\nImage : {path}")
    print(f"Keypoints : {len(kp)}")
    print(f"Descriptor Shape : {desc.shape}")
    print(f"Extraction Time : {elapsed:.5f} s")

    all_descriptors.append(desc)

    labels.append(label)

# =========================================
# BUILD BOVW
# =========================================
print("\n=== BUILD VOCABULARY ===")

kmeans = build_vocabulary(
    all_descriptors,
    k=50
)

# =========================================
# BUILD HISTOGRAM FEATURES
# =========================================
for desc in all_descriptors:

    hist = build_histogram(
        desc,
        kmeans
    )

    feature_vectors.append(hist)

X = np.array(feature_vectors)

y = np.array(labels)

# =========================================
# PCA REDUCTION
# =========================================
print("\n=== PCA REDUCTION ===")

pca = PCA(
    n_components=min(10, X.shape[0], X.shape[1])
)

X_pca = pca.fit_transform(X)

print("Original Shape :", X.shape)
print("Reduced Shape :", X_pca.shape)

# =========================================
# TRAIN TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_pca,
    y,
    test_size=0.3,
    random_state=42
)

# =========================================
# KNN CLASSIFIER
# =========================================
knn = KNeighborsClassifier(
    n_neighbors=3
)

knn.fit(
    X_train,
    y_train
)

y_pred = knn.predict(X_test)

accuracy = np.mean(y_pred == y_test)

print("\n=== CLASSIFICATION RESULT ===")
print("Accuracy :", accuracy)

# =========================================
# CONFUSION MATRIX
# =========================================
cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix")
print(cm)

# =========================================
# FEATURE MATCHING DEMO
# =========================================
img1 = data[0][0]
img2 = data[1][0]

kp1, desc1, _ = extract_features(
    sift,
    img1
)

kp2, desc2, _ = extract_features(
    sift,
    img2
)

matches = brute_force_matching(
    desc1,
    desc2,
    method="SIFT"
)

H, mask = homography_ransac(
    kp1,
    kp2,
    matches
)

# =========================================
# DRAW MATCHES
# =========================================
match_img = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    matches[:30],
    None,
    flags=2
)

# =========================================
# KEYPOINT VISUALIZATION
# =========================================
kp_img = cv2.drawKeypoints(
    img1,
    kp1,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# =========================================
# DISPLAY RESULT
# =========================================
plt.figure(figsize=(14,10))

# KEYPOINTS
plt.subplot(2,2,1)

plt.imshow(kp_img, cmap='gray')

plt.title("SIFT Keypoints")

plt.axis('off')

# MATCHING
plt.subplot(2,2,2)

plt.imshow(match_img)

plt.title("Feature Matching")

plt.axis('off')

# CONFUSION MATRIX
plt.subplot(2,2,3)

plt.imshow(cm, cmap='Blues')

plt.title("Confusion Matrix")

plt.colorbar()

# PCA INFORMATION
plt.subplot(2,2,4)

plt.axis('off')

plt.text(
    0.1,
    0.8,
    f"Vocabulary Size : 50",
    fontsize=12
)

plt.text(
    0.1,
    0.6,
    f"PCA Components : 32",
    fontsize=12
)

plt.text(
    0.1,
    0.4,
    f"Accuracy : {accuracy:.2f}",
    fontsize=12
)

plt.text(
    0.1,
    0.2,
    f"Feature : SIFT + BoVW + PCA",
    fontsize=12
)

plt.tight_layout()

plt.show()