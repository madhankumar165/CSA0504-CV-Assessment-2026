import cv2
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load Image
# -----------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "medical_image.png")

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("medical_image.png not found!")
    exit()

# -----------------------------
# Pixel Resolution
# -----------------------------

low_pixel = cv2.resize(image, (100,100), interpolation=cv2.INTER_AREA)

low_pixel = cv2.resize(
    low_pixel,
    (image.shape[1], image.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

# -----------------------------
# Intensity Resolution
# -----------------------------

levels = 16
factor = 256 // levels

low_intensity = (image // factor) * factor

# -----------------------------
# Display
# -----------------------------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(image,cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(low_pixel,cmap="gray")
plt.title("Low Pixel Resolution")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(low_intensity,cmap="gray")
plt.title("Low Intensity Resolution")
plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------
# Output
# -----------------------------

print("\n========== MEDICAL IMAGE ANALYSIS ==========\n")

print("Pixel Resolution:")
print("- Controls image sharpness.")
print("- Lower resolution causes pixelation.")
print("- Small fractures and tumors become difficult to detect.")

print("\nIntensity Resolution:")
print("- Controls gray-level representation.")
print("- Lower gray levels reduce contrast.")
print("- Soft tissues become difficult to distinguish.")

print("\nConclusion:")
print("Pixel resolution improves spatial detail, while intensity resolution improves tissue contrast.")
print("Both are essential for accurate diagnosis in medical imaging.")