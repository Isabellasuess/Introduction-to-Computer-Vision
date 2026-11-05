import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Load image
IMAGE_PATH = Path(__file__).resolve().parent.parent / "Images" / "ItCV3.jpg"

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Could not read image at: {IMAGE_PATH}")

img_array = np.array(Image.open(IMAGE_PATH).convert("L"))

# Define structuring elements
se_vertical = np.array([
    [1],
    [1],
    [1],
], dtype=np.uint8)

se_horizontal = np.array([
    [1, 1, 1]
], dtype=np.uint8)

se_cross = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
], dtype=np.uint8)

se_square = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
], dtype=np.uint8)

se_diamond = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
], dtype=np.uint8)

structuring_elements = {
    "Vertical SE":     se_vertical,
    "Horizontal SE":   se_horizontal,
    "Cross SE":        se_cross,
    "Square SE":       se_square,
    "Diamond/Disc SE": se_diamond,
}

# Dilate
NUM_ITERATIONS = 7

dilated_results = {}
for name, se in structuring_elements.items():
    dilated_results[name] = cv2.dilate(img_array, se, iterations=NUM_ITERATIONS)

# Show images
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

axes[0].imshow(img_array, cmap="gray")
axes[0].set_title("Original")

for ax, (name, result) in zip(axes[1:], dilated_results.items()):
    ax.imshow(result, cmap="gray")
    ax.set_title(f"Dilation: {name}\n({NUM_ITERATIONS}x)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("dilation_results.png", dpi=150)
plt.show()