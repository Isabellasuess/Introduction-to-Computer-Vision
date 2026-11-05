import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Load image
IMAGE_PATH = Path(__file__).resolve().parent.parent / "Images" / "images_opening_cross_3.png"

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
    "Vertical SE": se_vertical,
    "Horizontal SE": se_horizontal,
    "Cross SE": se_cross,
    "Square SE": se_square,
    "Diamond/Disc SE": se_diamond,
}

NUM_ITERATIONS = 3

# Opening = erosion followed by dilation
opening_results = {}
for name, se in structuring_elements.items():
    eroded = cv2.erode(img_array, se, iterations=NUM_ITERATIONS)
    opening_results[name] = cv2.dilate(eroded, se, iterations=NUM_ITERATIONS)

# Closing = dilation followed by erosion
closing_results = {}
for name, se in structuring_elements.items():
    dilated = cv2.dilate(img_array, se, iterations=NUM_ITERATIONS)
    closing_results[name] = cv2.erode(dilated, se, iterations=NUM_ITERATIONS)

# Show Opening results
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

axes[0].imshow(img_array, cmap="gray")
axes[0].set_title("Original")

for ax, (name, result) in zip(axes[1:], opening_results.items()):
    ax.imshow(result, cmap="gray")
    ax.set_title(f"Opening: {name}\n({NUM_ITERATIONS}x)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("opening_results.png", dpi=150)
plt.show()

# Show Closing results
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

axes[0].imshow(img_array, cmap="gray")
axes[0].set_title("Original")

for ax, (name, result) in zip(axes[1:], closing_results.items()):
    ax.imshow(result, cmap="gray")
    ax.set_title(f"Closing: {name}\n({NUM_ITERATIONS}x)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("closing_results.png", dpi=150)
plt.show()