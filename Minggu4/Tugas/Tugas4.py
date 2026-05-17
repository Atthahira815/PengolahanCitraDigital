import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================================
# METRIC FUNCTIONS
# ================================

def contrast_ratio(image):
    return np.max(image) - np.min(image)

def entropy_calc(image):
    hist,_ = np.histogram(image.flatten(),256,[0,256])
    prob = hist/np.sum(hist)
    prob = prob[prob>0]
    entropy = -np.sum(prob*np.log2(prob))
    return entropy


# ================================
# POINT PROCESSING
# ================================

def negative_transform(img):
    return 255 - img

def log_transform(img):
    img_float = img.astype(np.float32)
    c = 255 / np.log(1 + np.max(img_float))
    log_img = c * np.log(1 + img_float)
    return np.array(log_img, dtype=np.uint8)

def gamma_transform(img,gamma):
    return np.array(255*(img/255)**gamma,dtype='uint8')


# ================================
# CONTRAST STRETCHING
# ================================

def contrast_stretch_manual(img):
    rmin = np.min(img)
    rmax = np.max(img)

    stretched = (img-rmin)/(rmax-rmin)*255
    return np.array(stretched,dtype='uint8')


def contrast_stretch_auto(img):
    return cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)


# ================================
# HISTOGRAM METHODS
# ================================

def hist_equalization(img):
    return cv2.equalizeHist(img)

def clahe_enhance(img):
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    return clahe.apply(img)


# ================================
# VISUALIZATION FUNCTION
# ================================

def show_results(original,results,title):

    plt.figure(figsize=(14,10))

    names = list(results.keys())
    images = list(results.values())

    plt.subplot(3,4,1)
    plt.imshow(original,cmap='gray')
    plt.title("Original")
    plt.axis("off")

    for i,img in enumerate(images):

        plt.subplot(3,4,i+2)
        plt.imshow(img,cmap='gray')
        plt.title(names[i])
        plt.axis("off")

    plt.suptitle(title)
    plt.show()


def show_histograms(original,enhanced,title):

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.hist(original.flatten(), bins=256, range=(0,256))
    plt.title("Histogram Original")

    plt.subplot(1,2,2)
    plt.hist(enhanced.flatten(), bins=256, range=(0,256))
    plt.title("Histogram Enhanced")

    plt.suptitle(title)
    plt.show()


# ================================
# PROCESS IMAGE
# ================================

def process_image(path,title):

    img = cv2.imread(path,0)

    results = {}

    # point processing
    results["Negative"] = negative_transform(img)
    results["Log"] = log_transform(img)
    results["Gamma 0.5"] = gamma_transform(img,0.5)
    results["Gamma 1.0"] = gamma_transform(img,1.0)
    results["Gamma 2.0"] = gamma_transform(img,2.0)

    # histogram methods
    results["Stretch Manual"] = contrast_stretch_manual(img)
    results["Stretch Auto"] = contrast_stretch_auto(img)
    results["Hist Equalization"] = hist_equalization(img)
    results["CLAHE"] = clahe_enhance(img)

    # show results
    show_results(img,results,title)

    # evaluation metrics
    print("\n"+"="*60)
    print(title)
    print("="*60)

    print(f"{'Method':<20}{'Contrast':<15}{'Entropy':<15}")
    print("-"*60)

    # Original
    print(f"{'Original':<20}{contrast_ratio(img):<15.2f}{entropy_calc(img):<15.2f}")

    # Results
    for name,image in results.items():

        contrast = contrast_ratio(image)
        entropy = entropy_calc(image)

        print(f"{name:<20}{contrast:<15.2f}{entropy:<15.2f}")

    # histogram comparison example
    show_histograms(img,results["CLAHE"],title+" CLAHE")


# ================================
# MAIN PROGRAM
# ================================

process_image("Tugas/Week 4/UnderExposed.jpg","Underexposed Image")
process_image("Tugas/Week 4/OverExposed.jpg","Overexposed Image")
process_image("Tugas/Week 4/UnEven.jpg","Uneven Illumination Image")

print("\n=== PROGRAM SELESAI ===")