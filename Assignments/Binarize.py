import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# LOAD IMAGE
IMAGE_PATH = Path(__file__).resolve().parent.parent / "Images" / "IMG20240115134913.jpg"

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Could not read image at: {IMAGE_PATH}")

img_color = Image.open(IMAGE_PATH)
img_gray = np.array(img_color.convert("L"))

# BINARIZE
threshold = 128
binary_img = (img_gray > threshold).astype(np.uint8)

# SHOW ORIGINAL AND BINARIZED
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img_color)
axes[0].set_title("Original")

axes[1].imshow(binary_img, cmap="gray")
axes[1].set_title(f"Binarized (threshold = {threshold})")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("binarization_result.png", dpi=150)
plt.show()