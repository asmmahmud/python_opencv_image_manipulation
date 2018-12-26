import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt

bgr_img = cv2.imread('../../resources/omr-imgs/omr-1.png')
bgr_img = imutils.resize(bgr_img, width=1200)
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
# gray_img = cv2.medianBlur(gray_img, 3)
# gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
otsu_threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
otsu_threshold_img = cv2.erode(otsu_threshold_img, el, iterations=2)
otsu_threshold_img = cv2.dilate(otsu_threshold_img, el, iterations=1)

# arr = np.asarray(otsu_threshold_img[350:1150, 160: 660])
# print("console: ", gray_img.shape)
# plt.imshow(arr, cmap='gray')
# plt.xticks(list(range(0, 500, 100)))
# plt.yticks(list(range(0, 800, 100)))
# plt.show()

cropped_grayscale_thresh = otsu_threshold_img[400:1150, 220:410]
cropped_grayscale = gray_img[400:1150, 220:410]
circles = cv2.HoughCircles(cropped_grayscale_thresh, cv2.HOUGH_GRADIENT, dp=2, minDist=30, param1=50, param2=30, minRadius=14,
                           maxRadius=16)

circles = np.uint16(np.around(circles))
print(circles.shape)
if len(circles) is not None:
    cimg = cv2.cvtColor(cropped_grayscale, cv2.COLOR_GRAY2BGR)
    allCircles = circles[0]
    print('allCircles', allCircles.shape, allCircles.dtype)

    # allCircles = sorted(allCircles, key=lambda b: b[1], reverse=False)
    # allCircles = sorted(allCircles, key=lambda b: b[0], reverse=False)
    allCircles = allCircles[np.lexsort((allCircles[:, 0], allCircles[:, 1]))]
    seq = 1
    for i in allCircles:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 1)
        # draw the center of the circle
        # cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        # cv2.putText(cimg, "[" + str(i[0]) + "-" + str(i[1]) + "]", (i[0], i[1]), cv2.FONT_HERSHEY_PLAIN, .6, (0, 0, 0))
        cv2.putText(cimg, "[" + str(seq) + "]", (i[0] - 10, i[1] + 5), cv2.FONT_HERSHEY_PLAIN, .7, (0, 0, 0))
        seq += 1

    cv2.imwrite('../../resources/omr-imgs/img-detected-circle.png', cimg)
    cv2.imshow("otsu_threshold_img", cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
exit(0)
