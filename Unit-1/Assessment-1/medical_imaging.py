import cv2
import matplotlib.pyplot as plt
import os

# =====================================================
# Load Image (Windows/Linux/Mac Compatible)
# =====================================================

# Get the folder where this Python file is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Image file path
image_path = os.path.join(current_dir, "medical_image.png")

# Read image in grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image exists
if image is None:
    print("========================================")
    print("ERROR: medical_image.png not found!")
    print("Expected location:")
    print(image_path)
    print("========================================")
    exit()

print("Image loaded successfully!")
print("Image Path:", image_path)

# =====================================================
# Pixel Resolution Demonstration
# =====================================================

# Reduce image resolution
low_pixel = cv2.resize(
    image,
    (100, 100),
    interpolation=cv2.INTER_AREA
)

# Resize back to original size
low_pixel = cv2.resize(
    low_pixel,
    (image.shape[1], image.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

# =====================================================
# Intensity Resolution Demonstration
# =====================================================

# Reduce grayscale levels from 256 to 16
levels = 16
factor = 256 // levels
low_intensity = (image // factor) * factor

# =====================================================
# Display Images
# =====================================================

plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# Low Pixel Resolution
plt.subplot(1, 3, 2)
plt.imshow(low_pixel, cmap="gray")
plt.title("Low Pixel Resolution")
plt.axis("off")

# Low Intensity Resolution
plt.subplot(1, 3, 3)
plt.imshow(low_intensity, cmap="gray")
plt.title("Low Intensity Resolution")
plt.axis("off")

plt.tight_layout()
plt.show()

# =====================================================
# Explanation
# =====================================================

print("\n========== MEDICAL IMAGING ANALYSIS ==========\n")

print("Pixel Resolution:")
print("- Pixel resolution determines the number of pixels in an image.")
print("- Lower pixel resolution reduces image sharpness.")
print("- Small structures such as tiny fractures, tumors, or blood vessels become difficult to detect.")

print("\nIntensity Resolution:")
print("- Intensity resolution determines the number of gray levels.")
print("- Lower intensity resolution reduces contrast between tissues.")
print("- Different tissues may appear similar, making diagnosis more difficult.")

print("\nConclusion:")
print("- High pixel resolution improves spatial detail.")
print("- High intensity resolution improves contrast between tissues.")
print("- Both are essential for accurate diagnosis in medical imaging.")