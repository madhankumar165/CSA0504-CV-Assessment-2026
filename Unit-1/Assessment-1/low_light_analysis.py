import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------
# Load Image
# -------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "low_light.jpg")

image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    print(image_path)
    exit()

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -------------------------------
# Brightness Enhancement
# -------------------------------

enhanced = cv2.convertScaleAbs(image, alpha=1.4, beta=40)

# -------------------------------
# Histogram Equalization
# -------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

equalized = cv2.equalizeHist(gray)

# -------------------------------
# Histogram
# -------------------------------

hist = cv2.calcHist([gray],[0],None,[256],[0,256])

# -------------------------------
# Display
# -------------------------------

plt.figure(figsize=(15,8))

plt.subplot(2,2,1)
plt.imshow(image)
plt.title("Original Low-Light Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(enhanced)
plt.title("Brightness Enhanced")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(equalized,cmap='gray')
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(2,2,4)
plt.plot(hist)
plt.title("Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# -------------------------------
# Analysis
# -------------------------------

print("\n========== IMAGE SENSING ANALYSIS ==========\n")

print("Observation:")
print("- Original image is dark due to insufficient light.")
print("- Sensor captures fewer photons.")
print("- Electronic noise becomes more visible.")
print("- Histogram shows most pixels concentrated in lower intensities.")
print("- Brightness enhancement improves visibility.")
print("- Histogram equalization improves overall contrast.")

print("\nConclusion:")
print("Low-light image acquisition reduces image quality due to limited light reaching the sensor.")
print("Image enhancement techniques improve brightness and contrast,")
print("but cannot completely recover details lost during acquisition.")