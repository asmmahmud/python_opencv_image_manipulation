import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt
import pytesseract

image_path = "../../resources/img/nid4.jpg"
ori_img = cv2.imread(image_path)

if ori_img is None:
	exit(0)

ori_img = imutils.resize(ori_img, width=1024)
ori_img = cv2.GaussianBlur(ori_img, (5, 5), 0)
img_bin = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
img_bin = cv2.bitwise_not(img_bin)
# img_bin = 255 - img_bin
img_bin = cv2.threshold(img_bin, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# img_bin = cv2.adaptiveThreshold(img_bin, 255, cv2.ADAPTIVE_THRESH_MEAN_C,  cv2.THRESH_BINARY, 15, -9)

horiz_line = int(img_bin.shape[1] / 5)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_line, 1))

# img_bin = cv2.dilate(img_bin, kernel, iterations=2)
img_bin = cv2.erode(img_bin, kernel, iterations=1)
img_bin = cv2.dilate(img_bin, kernel, iterations=1)

# img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
# img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=2)

cv2.imshow("ori_img", img_bin)

cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
