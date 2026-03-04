import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# =========================
# LOAD IMAGE
# =========================

ref = cv2.imread("Tugas/Week 3/Normal.jpeg", 0)    # gambar normal
target = cv2.imread("Tugas/Week 3/Miring.jpeg", 0) # gambar miring

h, w = ref.shape


# =========================
# 1. GEOMETRIC TRANSFORM
# =========================

# Translation
M_trans = np.float32([[1,0,40],
                      [0,1,30]])
translation = cv2.warpAffine(ref, M_trans, (w,h))

# Rotation
center = (w//2, h//2)
M_rot = cv2.getRotationMatrix2D(center, 30, 1)
rotation = cv2.warpAffine(ref, M_rot, (w,h))

# Scaling
scaling = cv2.resize(ref, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

# Affine
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M_aff = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(ref, M_aff, (w,h))

# Perspective
pts1 = np.float32([[0,0],[w,0],[w,h],[0,h]])
pts2 = np.float32([[50,0],[w-50,50],[w,h],[0,h-50]])

M_per = cv2.getPerspectiveTransform(pts1, pts2)
perspective = cv2.warpPerspective(target, M_per, (w,h))


# =========================
# 2. INTERPOLATION (DOWN → UP)
# =========================

# Step 1: Downscale dulu
small = cv2.resize(ref, None, fx=0.25, fy=0.25)

# Step 2: Upscale kembali pakai metode berbeda

start = time.time()
nearest = cv2.resize(small, (w,h), interpolation=cv2.INTER_NEAREST)
t_nearest = time.time() - start

start = time.time()
bilinear = cv2.resize(small, (w,h), interpolation=cv2.INTER_LINEAR)
t_bilinear = time.time() - start

start = time.time()
bicubic = cv2.resize(small, (w,h), interpolation=cv2.INTER_CUBIC)
t_bicubic = time.time() - start


# =========================
# 3. METRICS
# =========================

def mse(img1,img2):
    return np.mean((img1.astype(float)-img2.astype(float))**2)

def psnr(img1,img2):
    m = mse(img1,img2)
    if m == 0:
        return 100
    return 10*np.log10((255**2)/m)


nearest_r = cv2.resize(nearest,(w,h))
bilinear_r = cv2.resize(bilinear,(w,h))
bicubic_r = cv2.resize(bicubic,(w,h))


print("===== EVALUASI =====")

print("\nNearest")
print("MSE:",mse(ref,nearest_r))
print("PSNR:",psnr(ref,nearest_r))
print("Time:",t_nearest)

print("\nBilinear")
print("MSE:",mse(ref,bilinear_r))
print("PSNR:",psnr(ref,bilinear_r))
print("Time:",t_bilinear)

print("\nBicubic")
print("MSE:",mse(ref,bicubic_r))
print("PSNR:",psnr(ref,bicubic_r))
print("Time:",t_bicubic)


# =========================
# FIGURE 1
# GEOMETRIC TRANSFORM
# =========================

plt.figure(figsize=(10,8))

plt.subplot(231)
plt.imshow(ref,cmap="gray")
plt.title("Reference")
plt.axis("off")

plt.subplot(232)
plt.imshow(target,cmap="gray")
plt.title("Target")
plt.axis("off")

plt.subplot(233)
plt.imshow(translation,cmap="gray")
plt.title("Translation")
plt.axis("off")

plt.subplot(234)
plt.imshow(rotation,cmap="gray")
plt.title("Rotation")
plt.axis("off")

plt.subplot(235)
plt.imshow(affine,cmap="gray")
plt.title("Affine")
plt.axis("off")

plt.subplot(236)
plt.imshow(perspective,cmap="gray")
plt.title("Perspective")
plt.axis("off")

plt.suptitle("Geometric Transformations")
plt.show()


# =========================
# FIGURE 2
# INTERPOLATION
# =========================

plt.figure(figsize=(12,5))

plt.subplot(141)
plt.imshow(small, cmap="gray")
plt.title("Downscaled Image")
plt.axis("off")

plt.subplot(142)
plt.imshow(nearest, cmap="gray")
plt.title("Nearest")
plt.axis("off")

plt.subplot(143)
plt.imshow(bilinear, cmap="gray")
plt.title("Bilinear")
plt.axis("off")

plt.subplot(144)
plt.imshow(bicubic, cmap="gray")
plt.title("Bicubic")
plt.axis("off")

plt.suptitle("Interpolation Comparison")
plt.show()


# =========================
# FIGURE 3
# ERROR MAP
# =========================

plt.figure(figsize=(10,5))

plt.subplot(131)
plt.imshow(abs(ref-nearest_r),cmap="hot")
plt.title("Error Nearest")

plt.subplot(132)
plt.imshow(abs(ref-bilinear_r),cmap="hot")
plt.title("Error Bilinear")

plt.subplot(133)
plt.imshow(abs(ref-bicubic_r),cmap="hot")
plt.title("Error Bicubic")

plt.suptitle("Error Map Comparison")
plt.show()
