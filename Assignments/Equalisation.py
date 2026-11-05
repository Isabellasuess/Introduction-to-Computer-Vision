import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from pathlib import Path

# LOAD IMAGE AND CONVERT TO GRAYSCALE
IMAGE_PATH = str(Path(__file__).resolve().parent.parent / "Images" / "IMG20240115134913.jpg")

img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise FileNotFoundError(f"Could not read image at: {IMAGE_PATH}")

# HISTOGRAM EQUALIZATION
img_he = cv2.equalizeHist(img_gray)

# ADAPTIVE HISTOGRAM EQUALIZATION (AHE) -> No separate builtin Function found
# Approximate AHE —> at high values, the clip limit becomes so large it essentially never triggers
ahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
img_ahe = ahe.apply(img_gray)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img_gray)

# PSNR
psnr_eq = psnr(img_gray, img_he, data_range=255)
psnr_ahe = psnr(img_gray, img_ahe, data_range=255)
psnr_clahe = psnr(img_gray, img_clahe, data_range=255)

print(f"PSNR Histogram Equalization: {psnr_eq:.2f}")
print(f"PSNR AHE: {psnr_ahe:.2f}")
print(f"PSNR CLAHE: {psnr_clahe:.2f}")

# Display: images + histograms
fig, ax = plt.subplots(4, 2, figsize=(12, 16))

ax[0, 0].imshow(img_gray, cmap='gray')
ax[0, 0].set_title("Original Image")
ax[0, 0].axis('off')
ax[0, 1].hist(img_gray.ravel(), bins=256, range=(0, 255), color='gray')
ax[0, 1].set_title("Original Histogram")

ax[1, 0].imshow(img_he, cmap='gray')
ax[1, 0].set_title(f"Histogram Equalization\nPSNR = {psnr_eq:.2f} dB")
ax[1, 0].axis('off')
ax[1, 1].hist(img_he.ravel(), bins=256, range=(0, 255), color='gray')
ax[1, 1].set_title("Equalized Histogram")

ax[2, 0].imshow(img_ahe, cmap='gray')
ax[2, 0].set_title(f"AHE \nPSNR = {psnr_ahe:.2f} dB")
ax[2, 0].axis('off')
ax[2, 1].hist(img_ahe.ravel(), bins=256, range=(0, 255), color='gray')
ax[2, 1].set_title("AHE Histogram")

ax[3, 0].imshow(img_clahe, cmap='gray')
ax[3, 0].set_title(f"CLAHE \nPSNR = {psnr_clahe:.2f} dB")
ax[3, 0].axis('off')
ax[3, 1].hist(img_clahe.ravel(), bins=256, range=(0, 255), color='gray')
ax[3, 1].set_title("CLAHE Histogram")

plt.tight_layout()
plt.show()