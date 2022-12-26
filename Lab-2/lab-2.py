import matplotlib.pyplot as plt
import cv2

from skimage.feature import greycomatrix, greycoprops

image = cv2.imread('./pictures/image11282022_500m.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


PATCH_SIZE = 20

light_locations =[(1026, 388), (702, 1094), (1513, 316), (1918, 420)]
light_patches = []

for loc in light_locations:
    light_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                         loc[1]:loc[1] + PATCH_SIZE])

dark_locations = [(2500, 324), (2461, 634), (2666, 700), (3668, 700)]

dark_patches = []
for loc in dark_locations:
    dark_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                        loc[1]:loc[1] + PATCH_SIZE])

# compute GLCM properties each patch
xs = []
ys = []

for patch in (light_patches + dark_patches):
    glcm = greycomatrix(patch, distances=[4], angles=[0], levels=256,
            symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])

fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray,
        vmin=0, vmax=255)

for (y,x) in light_locations: 
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')

for (y,x) in dark_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')

ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(light_patches)], ys[:len(light_patches)], 'go',
        label='Desert')
ax.plot(xs[len(dark_patches):], ys[len(dark_patches):], 'bo',
        label='Water')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')

for i, patch in enumerate(light_patches):
    ax = fig.add_subplot(3, len(light_patches), len(light_patches) * 1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,
            vmin=0, vmax=255)
    ax.set_xlabel('Desert %d' % (i + 1))

for i, patch in enumerate(dark_patches): 
    ax = fig.add_subplot(3, len(dark_patches), len(dark_patches) * 2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,
            vmin=0, vmax=255)
    ax.set_xlabel('Water %d' % (i + 1))

fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()
