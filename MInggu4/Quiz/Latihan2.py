import cv2
import numpy as np

print ("Analisis dan implementasi enhancement pipeline untuk citra medis\n")

def medical_image_enhancement(medical_image, modality='X-ray'):
    
    report = {}

    # Convert grayscale jika perlu
    if len(medical_image.shape) == 3:
        medical_image = cv2.cvtColor(medical_image, cv2.COLOR_BGR2GRAY)

    # Enhancement berdasarkan modality
    if modality == 'X-ray':

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(medical_image)

        report['method'] = "CLAHE"
        report['purpose'] = "Increase bone structure contrast"

    elif modality == 'MRI':

        enhanced = cv2.GaussianBlur(medical_image,(5,5),0)
        enhanced = cv2.equalizeHist(enhanced)

        report['method'] = "Gaussian + Histogram Equalization"
        report['purpose'] = "Improve soft tissue visibility"

    elif modality == 'CT':

        enhanced = cv2.normalize(medical_image,None,0,255,cv2.NORM_MINMAX)

        report['method'] = "Intensity Normalization"
        report['purpose'] = "Standardize Hounsfield units"

    elif modality == 'Ultrasound':

        median = cv2.medianBlur(medical_image,5)
        enhanced = cv2.equalizeHist(median)

        report['method'] = "Median Filter + Histogram Equalization"
        report['purpose'] = "Reduce speckle noise"

    else:
        enhanced = medical_image
        report['method'] = "None"

    # Metrics
    report['mean_intensity'] = float(np.mean(enhanced))
    report['std_intensity'] = float(np.std(enhanced))

    return enhanced, report


# Contoh penggunaan
img = cv2.imread("Minggu4/Quiz/XRayKepala.jpg", cv2.IMREAD_GRAYSCALE)

enhanced, report = medical_image_enhancement(img,'X-ray')

print("\n=== Enhancement Report ===")
for key, value in report.items():
    print(f"{key:<15} : {value}")

cv2.imshow("Original", img)
cv2.imshow("Enhanced", enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()


print("\n=== PRAKTIKUM SELESAI ===")
