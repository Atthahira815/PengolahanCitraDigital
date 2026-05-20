import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    Flatten, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import cv2

# =========================================================
# LOAD DATASET CIFAR-10
# =========================================================

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

num_classes = 10

y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# =========================================================
# VISUALISASI DATASET
# =========================================================

plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis("off")

plt.suptitle("Sample CIFAR-10 Images")
plt.tight_layout()
plt.show()

# =========================================================
# DATA AUGMENTATION
# =========================================================

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

datagen.fit(X_train)

# =========================================================
# VISUALISASI AUGMENTASI
# =========================================================

sample = X_train[0]

sample = np.expand_dims(sample, 0)

plt.figure(figsize=(12, 6))

i = 0

for batch in datagen.flow(sample, batch_size=1):

    plt.subplot(2, 5, i + 1)
    plt.imshow(batch[0])
    plt.axis("off")

    i += 1

    if i >= 10:
        break

plt.suptitle("Data Augmentation Examples")
plt.show()

# =========================================================
# CNN FROM SCRATCH
# =========================================================

def build_cnn(dropout_rate=0.5):

    model = Sequential([

        Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),

        Flatten(),

        Dense(256, activation='relu'),
        Dropout(dropout_rate),

        Dense(num_classes, activation='softmax')

    ])

    return model

# =========================================================
# BUILD MODEL
# =========================================================

cnn_model = build_cnn(dropout_rate=0.5)

cnn_model.summary()

# =========================================================
# COMPILE MODEL
# =========================================================

cnn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================================
# TRAIN CNN
# EPOCH DIKECILIN BIAR CEPAT
# =========================================================

start_time = time.time()

history = cnn_model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=64),
    validation_data=(X_test, y_test_cat),
    epochs=2,
    verbose=1
)

cnn_training_time = time.time() - start_time

print("\nCNN Training Time:", cnn_training_time)

# =========================================================
# LEARNING CURVE
# =========================================================

plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')

plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')

plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# =========================================================
# EVALUASI CNN
# =========================================================

loss, accuracy = cnn_model.evaluate(X_test, y_test_cat)

print("\nCNN Accuracy:", accuracy)

# =========================================================
# PREDIKSI
# =========================================================

y_pred = cnn_model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test.flatten()

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nCLASSIFICATION REPORT")

print(classification_report(
    y_true,
    y_pred_classes,
    target_names=class_names
))

# =========================================================
# FEATURE MAP VISUALIZATION
# =========================================================

feature_model = Model(
    inputs=cnn_model.inputs,
    outputs=cnn_model.layers[0].output
)

sample_image = np.expand_dims(X_test[0], axis=0)

feature_maps = feature_model.predict(sample_image)

plt.figure(figsize=(12,12))

for i in range(16):

    plt.subplot(4,4,i+1)

    plt.imshow(feature_maps[0, :, :, i], cmap='viridis')

    plt.axis('off')

plt.suptitle("Feature Maps")
plt.show()

# =========================================================
# TRANSFER LEARNING - VGG16
# =========================================================

base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32,32,3)
)

# Freeze layers
base_model.trainable = False

transfer_model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(256, activation='relu'),

    Dropout(0.5),

    Dense(num_classes, activation='softmax')

])

transfer_model.compile(
    optimizer=Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================================
# TRAIN TRANSFER LEARNING
# =========================================================

start_time = time.time()

transfer_history = transfer_model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=64),
    validation_data=(X_test, y_test_cat),
    epochs=2,
    verbose=1
)

transfer_training_time = time.time() - start_time

print("\nTransfer Learning Training Time:", transfer_training_time)

# =========================================================
# FINE TUNING
# =========================================================

base_model.trainable = True

for layer in base_model.layers[:-4]:
    layer.trainable = False

transfer_model.compile(
    optimizer=Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_history = transfer_model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=64),
    validation_data=(X_test, y_test_cat),
    epochs=1,
    verbose=1
)

# =========================================================
# EVALUASI TRANSFER LEARNING
# =========================================================

transfer_loss, transfer_acc = transfer_model.evaluate(X_test, y_test_cat)

print("\nTransfer Learning Accuracy:", transfer_acc)

# =========================================================
# PERBANDINGAN MODEL
# =========================================================

models_name = ['CNN Scratch', 'Transfer Learning']
accuracies = [accuracy, transfer_acc]

plt.figure(figsize=(7,5))

bars = plt.bar(models_name, accuracies)

plt.ylabel("Accuracy")
plt.title("Model Comparison")

for bar, acc in zip(bars, accuracies):

    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{acc:.2f}",
        ha='center'
    )

plt.show()

# =========================================================
# PCA VISUALIZATION
# =========================================================

features = cnn_model.predict(X_test[:1000])

pca = PCA(n_components=2)

reduced = pca.fit_transform(features)

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    reduced[:,0],
    reduced[:,1],
    c=y_test[:1000].flatten(),
    cmap='tab10'
)

plt.colorbar(scatter)

plt.title("PCA Feature Embedding")

plt.show()

# =========================================================
# t-SNE VISUALIZATION
# =========================================================

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

tsne_result = tsne.fit_transform(features)

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    tsne_result[:,0],
    tsne_result[:,1],
    c=y_test[:1000].flatten(),
    cmap='tab10'
)

plt.colorbar(scatter)

plt.title("t-SNE Feature Embedding")

plt.show()

# =========================================================
# ROC CURVE
# =========================================================

plt.figure(figsize=(10,8))

for i in range(num_classes):

    fpr, tpr, _ = roc_curve(
        y_test_cat[:, i],
        y_pred[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f'{class_names[i]} AUC={roc_auc:.2f}'
    )

plt.plot([0,1], [0,1], 'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

# =========================================================
# VISUALISASI HASIL PREDIKSI
# =========================================================

plt.figure(figsize=(12,12))

for i in range(16):

    plt.subplot(4,4,i+1)

    plt.imshow(X_test[i])

    pred = class_names[y_pred_classes[i]]
    true = class_names[y_true[i]]

    color = "green" if pred == true else "red"

    plt.title(f"P:{pred}\nT:{true}", color=color)

    plt.axis("off")

plt.tight_layout()

plt.show()

# =========================================================
# KESIMPULAN OTOMATIS
# =========================================================

print("\n==============================")
print("KESIMPULAN")
print("==============================")

print(f"CNN Scratch Accuracy      : {accuracy:.4f}")
print(f"Transfer Learning Accuracy: {transfer_acc:.4f}")

if transfer_acc > accuracy:
    print("Transfer Learning lebih baik.")
else:
    print("CNN Scratch lebih baik.")

print("\nProgram selesai.")