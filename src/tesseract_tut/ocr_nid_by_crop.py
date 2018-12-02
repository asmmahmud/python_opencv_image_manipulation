import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt
import pytesseract

imagePath = "../../resources/img/national-id-card3.jpg"
ori_img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
ori_img = imutils.resize(ori_img, height=1024)
img = ori_img[350:900, 420:1300]
img = cv2.GaussianBlur(img, (5, 5), 0)
otsu_threshold_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

kernel = np.ones((2, 2), np.uint8)
otsu_threshold_img = cv2.dilate(otsu_threshold_img, kernel, iterations=2)
otsu_threshold_img = cv2.erode(otsu_threshold_img, kernel, iterations=2)

plt.imshow(otsu_threshold_img, cmap='gray', interpolation='bicubic')
plt.show()

print(pytesseract.image_to_string(otsu_threshold_img, lang='ben+eng', config="--tessdata-dir ../tessdata --psm 3"))
