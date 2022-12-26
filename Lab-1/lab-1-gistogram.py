import cv2
from matplotlib import pyplot as plt

# read image
img1 = cv2.imread("./assets/IMG_1274.JPG")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread("./assets/IMG_1275.JPG", cv2.IMREAD_COLOR)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.subplot(2, 2, 1)
plt.imshow(img1)
plt.title('Image 1')

plt.subplot(2, 2, 2)
plt.imshow(img2)
plt.title('Image 2')

plt.subplot(2, 2, 3)
plt.hist(img1.ravel(), 256, [0, 256])
plt.xlabel('Intensity, (0..255)')
plt.ylabel('Count, pcs')
plt.title('Histogram of image')

plt.subplot(2, 2, 4)
plt.hist(img2.ravel(), 256, [0, 256])
plt.xlabel('Intensity, (0..255)')
plt.ylabel('Count, pcs')
plt.title('Histogram of image')
plt.show()
