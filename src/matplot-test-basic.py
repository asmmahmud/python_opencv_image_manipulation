import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load an color image in grayscale
img = cv.imread('nid.jpg', 1)

plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
