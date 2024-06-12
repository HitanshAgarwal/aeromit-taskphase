import cv2
import numpy as np
import matplotlib.pyplot as plt

#Loading the first image
image_path = input('Enter the image path: ')
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# First Gaussian blur
blurred1 = cv2.GaussianBlur(image, (0, 0), 1.0)

# Second Gaussian blur
blurred2 = cv2.GaussianBlur(image, (0, 0), 2.0)

#Calculating the difference in the gaussian blurs
difference = cv2.subtract(blurred1, blurred2)

difference_normalized = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Difference of Gaussians')
plt.imshow(difference_normalized, cmap='gray')
plt.axis('off')

plt.show()