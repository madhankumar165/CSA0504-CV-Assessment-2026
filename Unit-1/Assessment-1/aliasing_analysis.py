import cv2
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load Image
# -----------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "image.jpg")

image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    print(image_path)
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -----------------------------
# Create Aliasing
# -----------------------------

# Downsample aggressively
small = cv2.resize(image, (80, 80), interpolation=cv2.INTER_NEAREST)

# Upsample
aliased = cv2.resize(
    small,
    (image.shape[1], image.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

# -----------------------------
# Anti-Aliasing
# -----------------------------

blur = cv2.GaussianBlur(image, (5,5), 0)

smooth = cv2.resize(
    blur,
    (80,80),
    interpolation=cv2.INTER_AREA
)

corrected = cv2.resize(
    smooth,
    (image.shape[1], image.shape[0]),
    interpolation=cv2.INTER_CUBIC
)

# -----------------------------
# Display
# -----------------------------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(aliased)
plt.title("Aliasing Artifact")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(corrected)
plt.title("Corrected Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------
# Explanation
# -----------------------------

print("\n========== ALIASING ANALYSIS ==========\n")

print("Cause:")
print("- Low sampling frequency causes aliasing.")
print("- Fine details cannot be represented correctly.")
print("- Quantization with fewer intensity levels can further reduce quality.")

print("\nObserved Artifacts:")
print("- Jagged edges")
print("- Pixelation")
print("- Loss of fine details")

print("\nCorrective Approach:")
print("- Apply Gaussian Blur before downsampling.")
print("- Increase sampling resolution.")
print("- Use INTER_AREA for shrinking.")
print("- Use INTER_CUBIC or Lanczos for enlargement.")
print("- Use higher bit-depth to reduce quantization errors.")

print("\nConclusion:")
print("Aliasing occurs because the image is sampled below the Nyquist rate.")
print("Proper sampling and anti-aliasing filters significantly improve image quality.")