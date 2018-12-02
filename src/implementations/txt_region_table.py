import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt
import pytesseract

imagePath = "../../resources/img/tabular_data1.png"
ori_img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
ori_img = imutils.resize(ori_img, height=1024)

img = cv2.GaussianBlur(ori_img, (1, 1), 0)
otsu_threshold_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

kernel = np.ones((1, 1), np.uint8)
otsu_threshold_img = cv2.dilate(otsu_threshold_img, kernel, iterations=1)
otsu_threshold_img = cv2.erode(otsu_threshold_img, kernel, iterations=2)

# plt.imshow(otsu_threshold_img, cmap='gray', interpolation='bicubic')
# plt.show()
# cv2.imshow("otsu_threshold_img", otsu_threshold_img)
# print(pytesseract.image_to_string(otsu_threshold_img, lang='ben+eng', config="--tessdata-dir ../tessdata --psm 3"))

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 35))
openRectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
black_hat = cv2.morphologyEx(otsu_threshold_img, cv2.MORPH_BLACKHAT, rectKernel)

gradX = cv2.Sobel(black_hat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))

if (maxVal - minVal) > 0:
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

gradX_thresh = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
gradX_thresh = cv2.morphologyEx(gradX_thresh, cv2.MORPH_OPEN, rectKernel)
gradX_thresh = cv2.threshold(gradX_thresh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
# gradX_thresh = cv2.morphologyEx(gradX_thresh, cv2.MORPH_CLOSE, sqKernel)

kernel2 = np.ones((2, 2), np.uint8)
gradX_thresh = cv2.erode(gradX_thresh, kernel2, iterations=2)

# find contours in the thresholded image and sort them by their
# size
cnts = cv2.findContours(gradX_thresh.copy(), cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)[-2]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# loop over the contours
roiList = []
for c in cnts:
    # compute the bounding box of the contour and use the contour to
    # compute the aspect ratio and coverage ratio of the bounding box
    # width to the width of the image
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    crWidth = w / float(ori_img.shape[1])

    # check to see if the aspect ratio and coverage width are within
    # acceptable criteria

    if ar > 0 and crWidth > 0:
        # pad the bounding box since we applied erosions and now need
        # to re-grow it
        pX = int((x + w) * 0.03)
        pY = int((y + h) * 0.03)
        (x, y) = (x - pX, y - pY)
        (w, h) = (w + (pX * 2), h + (pY * 2))

        # extract the ROI from the image and draw a bounding box
        # surrounding the MRZ
        # roiList.append(image[y:y + h, x:x + w].copy())
        cv2.rectangle(ori_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow("ROI-" + str(thisIndex), roi)


cv2.imshow("gradX_thresh", gradX_thresh)
cv2.imshow("ori_img", ori_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
