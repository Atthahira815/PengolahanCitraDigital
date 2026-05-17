import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    learning_curve
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize

from skimage.feature import hog, local_binary_pattern

# ==========================================
# LOAD DATASET
# ==========================================
print("Loading Fashion-MNIST Dataset...")

fashion = fetch_openml(
    'Fashion-MNIST',
    version=1,
    as_frame=False
)

X = fashion.data[:1000]
y = fashion.target[:1000].astype(int)

# ==========================================
# FEATURE EXTRACTION
# ==========================================
print("\nExtracting Features...")

hog_features = []
lbp_features = []

for image in X:

    img = image.reshape(28,28).astype(np.uint8)

    # ==========================================
    # HOG FEATURE
    # ==========================================
    hog_feature = hog(
        img,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        visualize=False
    )

    hog_features.append(hog_feature)

    # ==========================================
    # LBP FEATURE
    # ==========================================
    lbp = local_binary_pattern(
        img,
        P=8,
        R=1,
        method='uniform'
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0,11),
        range=(0,10)
    )

    lbp_features.append(hist)

hog_features = np.array(hog_features)
lbp_features = np.array(lbp_features)

# ==========================================
# COMBINE FEATURES
# ==========================================
X_features = np.hstack([
    hog_features,
    lbp_features
])

print("Feature Shape :", X_features.shape)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ==========================================
# KNN EXPERIMENT
# ==========================================
print("\n=== KNN EXPERIMENT ===")

k_values = [1,3,5,7,9,11]

knn_results = []

for k in k_values:

    start = time.time()

    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric='euclidean'
    )

    knn.fit(X_train, y_train)

    train_time = time.time() - start

    start = time.time()

    y_pred = knn.predict(X_test)

    inference_time = time.time() - start

    acc = accuracy_score(y_test, y_pred)

    knn_results.append(acc)

    print(f"\nk = {k}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Training Time : {train_time:.5f}s")
    print(f"Inference Time : {inference_time:.5f}s")

# ==========================================
# SVM EXPERIMENT
# ==========================================
print("\n=== SVM EXPERIMENT ===")

svm = SVC(
    kernel='rbf',
    C=10,
    gamma=0.01,
    probability=True
)

start = time.time()

svm.fit(X_train, y_train)

svm_train_time = time.time() - start

start = time.time()

svm_pred = svm.predict(X_test)

svm_inference_time = time.time() - start

svm_acc = accuracy_score(
    y_test,
    svm_pred
)

print("\nSVM Accuracy :", svm_acc)

print("Training Time :", svm_train_time)

print("Inference Time :", svm_inference_time)

# ==========================================
# METRICS
# ==========================================
precision = precision_score(
    y_test,
    svm_pred,
    average='weighted'
)

recall = recall_score(
    y_test,
    svm_pred,
    average='weighted'
)

f1 = f1_score(
    y_test,
    svm_pred,
    average='weighted'
)

print("\n=== EVALUATION ===")

print("Precision :", precision)

print("Recall :", recall)

print("F1 Score :", f1)

# ==========================================
# CONFUSION MATRIX
# ==========================================
cm = confusion_matrix(
    y_test,
    svm_pred
)

# ==========================================
# PCA FOR DECISION BOUNDARY
# ==========================================
print("\nApplying PCA...")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_features)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca,
    y,
    test_size=0.3,
    random_state=42
)

svm_pca = SVC(
    kernel='rbf',
    gamma=0.01,
    C=10
)

svm_pca.fit(
    X_train_pca,
    y_train_pca
)

# ==========================================
# DECISION BOUNDARY
# ==========================================
x_min, x_max = X_pca[:,0].min()-1, X_pca[:,0].max()+1
y_min, y_max = X_pca[:,1].min()-1, X_pca[:,1].max()+1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.5),
    np.arange(y_min, y_max, 0.5)
)

Z = svm_pca.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

# ==========================================
# CROSS VALIDATION
# ==========================================
print("\n=== CROSS VALIDATION ===")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ==========================================
# GRID SEARCH
# ==========================================
print("\n=== GRID SEARCH ===")

param_grid = {
    'C':[0.1,1,10],
    'gamma':[0.001,0.01,0.1],
    'kernel':['rbf']
}

grid = GridSearchCV(
    SVC(),
    param_grid,
    cv=cv
)

grid.fit(X_train, y_train)

print("Best Parameter :", grid.best_params_)

# ==========================================
# LEARNING CURVE
# ==========================================
train_sizes, train_scores, test_scores = learning_curve(
    svm,
    X_features,
    y,
    cv=5,
    train_sizes=np.linspace(0.1,1.0,5)
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

# ==========================================
# ROC CURVE
# ==========================================
y_bin = label_binarize(
    y_test,
    classes=np.unique(y)
)

y_score = svm.predict_proba(X_test)

fpr, tpr, _ = roc_curve(
    y_bin.ravel(),
    y_score.ravel()
)

roc_auc = auc(fpr, tpr)

# ==========================================
# VISUALIZATION
# ==========================================
plt.figure(figsize=(16,12))

# KNN ACCURACY
plt.subplot(2,2,1)

plt.plot(
    k_values,
    knn_results,
    marker='o'
)

plt.title("KNN Accuracy vs k")

plt.xlabel("k")

plt.ylabel("Accuracy")

# CONFUSION MATRIX
plt.subplot(2,2,2)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(ax=plt.gca())

plt.title("Confusion Matrix")

# LEARNING CURVE
plt.subplot(2,2,3)

plt.plot(
    train_sizes,
    train_mean,
    label='Training Accuracy'
)

plt.plot(
    train_sizes,
    test_mean,
    label='Validation Accuracy'
)

plt.title("Learning Curve")

plt.xlabel("Training Size")

plt.ylabel("Accuracy")

plt.legend()

# ROC CURVE
plt.subplot(2,2,4)

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.2f}"
)

plt.plot([0,1],[0,1],'--')

plt.title("ROC Curve")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend()

plt.tight_layout()

plt.show()

# ==========================================
# DECISION BOUNDARY VISUALIZATION
# ==========================================
plt.figure(figsize=(8,6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3
)

scatter = plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y,
    cmap='tab10'
)

plt.title("SVM Decision Boundary (PCA 2D)")

plt.xlabel("PCA 1")

plt.ylabel("PCA 2")

plt.colorbar(scatter)

plt.show()