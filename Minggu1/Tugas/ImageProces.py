import cv2
import numpy as np

img = cv2.imread("Tugas/Week 1/RiskiGacor.jpg")

if img is None:
    print("Image not found!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Tampilkan matrix & vector
print(gray[:5, :5])
print(gray.flatten()[:25])

h, w = gray.shape
print("Resolution:", w, "x", h)
print("Aspect Ratio:", w/h)

memory = gray.size * gray.itemsize
print("Memory:", memory/1024, "KB")

# Manipulasi citra
crop = gray[100:400, 100:400]
resize = cv2.resize(gray, (w//2, h//2))
rotate = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
flip = cv2.flip(gray, 1)

# ===== TAMPILAN GAMBAR =====
cv2.imshow("Original", img)
cv2.imshow("Grayscale", gray)
cv2.imshow("Cropped", crop)
cv2.imshow("Resized", resize)
cv2.imshow("Rotated", rotate)
cv2.imshow("Flipped", flip)

cv2.waitKey(0)
cv2.destroyAllWindows()
