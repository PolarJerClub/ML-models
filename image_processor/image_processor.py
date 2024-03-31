import cv2
import matplotlib.pyplot as plt

# Load an image
image_path = 'path_to_your_image.jpg'  # Replace 'path_to_your_image.jpg' with the path to your image
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect edges in the image using the Canny edge detector
edges = cv2.Canny(gray_image, 100, 200)

# Display the original image and the edges
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

plt.show()