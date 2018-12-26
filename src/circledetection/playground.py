import numpy as np
import cv2
import imutils
import circledetection.omrutils as omrutils

# img = cv2.imread("../../resources/omr-imgs/mahmud-vai5.png")
img = cv2.imread("../../resources/omr-imgs/omr-mobile.jpg")
im_orig = imutils.resize(img, width=1000)[200:, :]

img = cv2.GaussianBlur(im_orig, (5, 5), 8)
im = omrutils.normalize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
im2, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = list(map(omrutils.get_approx_contour, contours))
# cv2.drawContours(im_orig, contours, -1, (0, 0, 200), 1)
cimg = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

for cnt in contours:
    (x, y, w, h) = cv2.boundingRect(cnt)
    aspect_ratio = float(h) / float(w)

    if 100 < w < 600 and 600 < h < 1200 and 2.5 < aspect_ratio < 2.75:
        pcnt = np.array(cnt)
        pcnt = pcnt[:, 0]
        zpoint4 = pcnt[np.lexsort((pcnt[:, 0], pcnt[:, 1]))]
        # print("after", zpoint4)
        # cnt = omrutils.order_points_clockwise(cnt)
        # orderedPoints = np.vstack([zpoint4[0:2, :], zpoint4[3], zpoint4[2]])
        # print("after order", orderedPoints)
        # transf = omrutils.perspective_transform(img, orderedPoints)
        lenn = 1
        for point in zpoint4:
            cv2.circle(cimg, (point[0], point[1]), 20, (0, 20, 220), 2)
            cv2.putText(cimg, "[" + str(lenn) + "]", (point[0] - 10, point[1] + 5), cv2.FONT_HERSHEY_PLAIN, .7, (0, 200, 0))
            lenn += 1

        # cv2.imshow("im_orig" + str(lenn), transf)

        # cv2.drawContours(im_orig, [cnt], 0, (10, 10, 200), thickness=2)
        # cv2.rectangle(im_orig, (x, y), (x + w, y + h), (10, 10, 220), thickness=2)

cv2.imshow("im_orig" , cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
