import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# ==================================
# MEMBUAT CITRA OTOMATIS
# ==================================

def buat_bimodal():
    img = np.zeros((256,256),dtype=np.uint8)
    cv2.circle(img,(128,128),60,255,-1)
    gt = img.copy()
    return img, gt

def buat_iluminasi():
    img = np.zeros((256,256),dtype=np.uint8)

    for i in range(256):
        img[:,i] = i

    cv2.rectangle(img,(80,80),(180,180),255,-1)

    gt = np.zeros((256,256),dtype=np.uint8)
    cv2.rectangle(gt,(80,80),(180,180),255,-1)

    return img, gt

def buat_overlap():
    img = np.zeros((256,256),dtype=np.uint8)

    cv2.circle(img,(90,130),50,255,-1)
    cv2.circle(img,(150,130),50,255,-1)
    cv2.circle(img,(120,80),50,255,-1)

    gt = img.copy()
    return img, gt

# ==================================
# METRIK
# ==================================

def metric(gt,pred):
    gt = gt > 0
    pred = pred > 0

    TP=np.logical_and(gt,pred).sum()
    TN=np.logical_and(~gt,~pred).sum()
    FP=np.logical_and(~gt,pred).sum()
    FN=np.logical_and(gt,~pred).sum()

    iou=TP/(TP+FP+FN+1e-6)
    dice=2*TP/(2*TP+FP+FN+1e-6)
    acc=(TP+TN)/(TP+TN+FP+FN)

    return iou,dice,acc

# ==================================
# DATASET
# ==================================

dataset = {
    "Bimodal": buat_bimodal(),
    "Iluminasi": buat_iluminasi(),
    "Overlap": buat_overlap()
}

# ==================================
# PROSES
# ==================================

for nama,(img,gt) in dataset.items():

    print("\n====",nama,"====")

    # OTSU
    t=time.time()
    _,otsu=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print("Otsu :",metric(gt,otsu),"time:",time.time()-t)

    # Adaptive
    t=time.time()
    adap=cv2.adaptiveThreshold(img,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,11,2)
    print("Adaptive :",metric(gt,adap),"time:",time.time()-t)

    # Sobel
    sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    sobel=np.sqrt(sobelx**2+sobely**2)

    # Canny
    canny=cv2.Canny(img,50,150)

    # TAMPILKAN
    plt.figure(figsize=(12,8))

    plt.subplot(231)
    plt.imshow(img,cmap='gray')
    plt.title("Original")

    plt.subplot(232)
    plt.imshow(gt,cmap='gray')
    plt.title("Ground Truth")

    plt.subplot(233)
    plt.imshow(otsu,cmap='gray')
    plt.title("Otsu")

    plt.subplot(234)
    plt.imshow(adap,cmap='gray')
    plt.title("Adaptive")

    plt.subplot(235)
    plt.imshow(sobel,cmap='gray')
    plt.title("Sobel")

    plt.subplot(236)
    plt.imshow(canny,cmap='gray')
    plt.title("Canny")

    plt.suptitle(nama)
    plt.tight_layout()
    plt.show()