import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt


def get_min_rect_points(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)


def get_min_enclosing_circle(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    return center, radius


bgr_img = cv2.imread('../../resources/omr-imgs/omr-1-ans-ori.png')
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
con_img, contours, heir = cv2.findContours(cropped_grayscale_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
boundingBoxes = [cv2.boundingRect(c) for c in contours]
boundingBoxes = np.array(boundingBoxes)
# print(boundingBoxes.shape, boundingBoxes.dtype)
print(cropped_grayscale_thresh.shape)
# combined = zip(contours, boundingBoxes)
# combined = sorted(combined, key=lambda b: b[1][2] * b[1][3], reverse=True)
# (contours, boundingBoxes) = zip(*combined)

cropped_grayscale = cv2.cvtColor(cropped_grayscale, cv2.COLOR_GRAY2BGR)

seq = 1
for cnt in contours:
    (x, y, w, h) = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)
    if 30 < w < 80 and 30 < h < 80 and 0.8 < aspect_ratio < 1.2:
        # cv2.rectangle(cimg, (x, y), (x + w, y + h), (0, 0, 200), 1)
        # cv2.putText(cimg, "[" + str(aspect_ratio) + "]", (x + 10, y + 10), cv2.FONT_HERSHEY_PLAIN, .7, (10, 180, 20))
        # mask = np.zeros(cropped_grayscale_thresh[y: y+h, x: x+w].shape, np.uint8)
        averageColor = np.average(cropped_grayscale_thresh[y: y + h, x: x + w]).round()
        print(averageColor)
        if averageColor < 100:
            # cv2.drawContours(cropped_grayscale, [cnt], 0, (10, 180, 20), thickness=1)
            center, radius = get_min_enclosing_circle(cnt)
            cv2.circle(cropped_grayscale, center, radius, (10, 180, 20), -1)
            seq += 1
        else:
            # cv2.drawContours(cropped_grayscale, [get_min_rect_points(cnt)], 0, (0, 0, 255), thickness=1)
            center, radius = get_min_enclosing_circle(cnt)
            cv2.circle(cropped_grayscale, center, radius, (0, 255, 0), 1)

# cv2.imwrite('../../resources/omr-imgs/img-detected-circle.png', cimg)
cv2.imshow("otsu_threshold_img", cropped_grayscale)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
