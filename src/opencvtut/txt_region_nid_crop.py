import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt
import pytesseract

imagePath = "../../resources/img/nid.jpg"
ori_img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
ori_img = imutils.resize(ori_img, height=1024)
# img = ori_img[300:, 400:]
img = cv2.GaussianBlur(ori_img, (5, 5), 0)
otsu_threshold_img = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

kernel = np.ones((2, 2), np.uint8)
# otsu_threshold_img = cv2.dilate(otsu_threshold_img, kernel, iterations=2)
# otsu_threshold_img = cv2.erode(otsu_threshold_img, kernel, iterations=2)

plt.imshow(otsu_threshold_img, cmap='gray', interpolation='bicubic')
plt.show()
exit(0)
print(pytesseract.image_to_string(otsu_threshold_img, lang='ben+eng', config="--tessdata-dir ../tessdata --psm 3"))

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (90, 8))
black_hat = cv2.morphologyEx(otsu_threshold_img, cv2.MORPH_BLACKHAT, rectKernel)

gradX = cv2.Sobel(black_hat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

gradX_thresh = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
gradX_thresh = cv2.morphologyEx(gradX_thresh, cv2.MORPH_OPEN, rectKernel)
gradX_thresh = cv2.threshold(gradX_thresh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# gradX_thresh = cv2.morphologyEx(gradX_thresh, cv2.MORPH_OPEN, rectKernel)
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
gradX_thresh = cv2.morphologyEx(gradX_thresh, cv2.MORPH_CLOSE, sqKernel)

kernel2 = np.ones((2, 2), np.uint8)
gradX_thresh = cv2.erode(gradX_thresh, kernel2, iterations=2)


cv2.imshow("gradX_thresh", gradX_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
