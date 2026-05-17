import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# GENERATE TEXT IMAGE WITH NOISE (OCR)
# ==========================================
def generate_text_image():

    img = np.ones((300, 800), dtype=np.uint8) * 255

    cv2.putText(
        img,
        'OCR MORPHOLOGY TEST',
        (40, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0),
        4
    )

    cv2.putText(
        img,
        'OPENING AND CLOSING',
        (60, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0),
        3
    )

    # Tambah noise titik
    noise = np.random.randint(0, 100, (300, 800))
    img[noise < 3] = 0

    # Tambah goresan
    cv2.line(img, (100, 50), (700, 250), 0, 2)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# ==========================================
# GENERATE OVERLAPPING OBJECTS
# ==========================================
def generate_objects_image():

    img = np.zeros((500, 500, 3), dtype=np.uint8)

    circles = [
        (150, 150),
        (220, 150),
        (290, 170),
        (180, 250),
        (260, 260),
        (340, 240),
        (150, 350),
        (240, 360),
        (330, 350)
    ]

    for center in circles:
        cv2.circle(
            img,
            center,
            60,
            (255, 255, 255),
            -1
        )

    return img

# ==========================================
# STRUCTURING ELEMENT
# ==========================================
def get_kernel(shape, size):

    if shape == 'square':
        return cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (size, size)
        )

    elif shape == 'cross':
        return cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            (size, size)
        )

    elif shape == 'ellipse':
        return cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (size, size)
        )

# ==========================================
# OCR PREPROCESSING PIPELINE
# ==========================================
def ocr_pipeline(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    kernel_open = get_kernel('ellipse', 3)

    opening = cv2.morphologyEx(
        blur,
        cv2.MORPH_OPEN,
        kernel_open
    )

    kernel_close = get_kernel('cross', 3)

    closing = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    _, thresh = cv2.threshold(
        closing,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return {
        'Gray': gray,
        'Blur': blur,
        'Opening': opening,
        'Closing': closing,
        'Threshold': thresh
    }

# ==========================================
# MORPHOLOGICAL OPERATIONS
# ==========================================
def morphology_operations(gray):

    kernel = get_kernel('ellipse', 5)

    start = time.time()
    erosion = cv2.erode(gray, kernel, iterations=1)
    erosion_time = time.time() - start

    start = time.time()
    dilation = cv2.dilate(gray, kernel, iterations=1)
    dilation_time = time.time() - start

    start = time.time()
    opening = cv2.morphologyEx(
        gray,
        cv2.MORPH_OPEN,
        kernel
    )
    opening_time = time.time() - start

    start = time.time()
    closing = cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        kernel
    )
    closing_time = time.time() - start

    gradient = cv2.morphologyEx(
        gray,
        cv2.MORPH_GRADIENT,
        kernel
    )

    top_hat = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        kernel
    )

    black_hat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    print("\nWAKTU KOMPUTASI")
    print(f"Erosi   : {erosion_time:.5f} detik")
    print(f"Dilasi  : {dilation_time:.5f} detik")
    print(f"Opening : {opening_time:.5f} detik")
    print(f"Closing : {closing_time:.5f} detik")

    return {
        'Original': gray,
        'Erosion': erosion,
        'Dilation': dilation,
        'Opening': opening,
        'Closing': closing,
        'Gradient': gradient,
        'Top Hat': top_hat,
        'Black Hat': black_hat
    }

# ==========================================
# OBJECT COUNTING WITH WATERSHED
# ==========================================
def count_objects(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY,
    )

    kernel = get_kernel('ellipse', 5)

    opening = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    sure_bg = cv2.dilate(
        opening,
        kernel,
        iterations=3
    )

    dist_transform = cv2.distanceTransform(
        opening,
        cv2.DIST_L2,
        5
    )

    _, sure_fg = cv2.threshold(
        dist_transform,
        0.4 * dist_transform.max(),
        255,
        0
    )

    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1

    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers)

    result = image.copy()

    result[markers == -1] = [0, 0, 255]

    count = len(np.unique(markers)) - 2

    return result, count

# ==========================================
# GENERATE IMAGES
# ==========================================
img_text = generate_text_image()

img_objects = generate_objects_image()

# ==========================================
# OCR PROCESS
# ==========================================
ocr_results = ocr_pipeline(img_text)

# ==========================================
# MORPHOLOGY OPERATIONS
# ==========================================
gray_text = cv2.cvtColor(
    img_text,
    cv2.COLOR_BGR2GRAY
)

morph_results = morphology_operations(gray_text)

# ==========================================
# OBJECT COUNTING
# ==========================================
watershed_result, total_objects = count_objects(img_objects)

print("\nCOUNTING OBJECT")
print("=" * 40)
print(f"Jumlah objek terdeteksi: {total_objects}")

# ==========================================
# DISPLAY MORPHOLOGY RESULTS
# ==========================================
plt.figure(figsize=(15, 12))

titles = list(morph_results.keys())

for i, title in enumerate(titles):

    plt.subplot(3, 3, i + 1)

    plt.imshow(
        morph_results[title],
        cmap='gray'
    )

    plt.title(title)

    plt.axis('off')

plt.tight_layout()

plt.show()

# ==========================================
# DISPLAY OCR PIPELINE
# ==========================================
plt.figure(figsize=(15, 8))

ocr_titles = ['Gray', 'Blur', 'Opening', 'Closing', 'Threshold']

for i, title in enumerate(ocr_titles):

    plt.subplot(2, 3, i + 1)

    plt.imshow(
        ocr_results[title],
        cmap='gray'
    )

    plt.title(title)

    plt.axis('off')

plt.tight_layout()

plt.show()

# ==========================================
# DISPLAY OBJECT COUNTING RESULT
# ==========================================
plt.figure(figsize=(8, 8))

plt.imshow(
    cv2.cvtColor(
        watershed_result,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(f'Watershed Result - Count: {total_objects}')

plt.axis('off')

plt.show()