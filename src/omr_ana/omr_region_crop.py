import cv2
import numpy as np
from omr_ana import omr as omr_utils
import imutils

im_orig = cv2.imread("../../resources/omr-imgs/m-2.jpg", cv2.IMREAD_COLOR)
im_orig = imutils.resize(im_orig, width=1200)
im = cv2.cvtColor(im_orig, cv2.COLOR_BGR2GRAY)
im = omr_utils.normalize(im)
im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
contours = omr_utils.get_contours(im)

corners = omr_utils.get_corners(contours, im)
cv2.drawContours(im_orig, corners, -1, (20, 200, 80), 1)
hull_points = omr_utils.get_convex_hull_points(corners)
cv2.drawContours(im_orig, [hull_points], -1, (20, 10, 200), 1)

for pt in hull_points:
    cv2.circle(im_orig, (pt[0][0], pt[0][1]), 5, (0, 200, 10), -1)

src_rect = omr_utils.get_outmost_points(hull_points)

# for i, pt in enumerate(npHullTop4):
#     cv2.putText(im_orig, "[" + str(i) + "]", (pt[0], pt[1]), cv2.FONT_HERSHEY_PLAIN, .7, (0, 200, 0))
#
# for i, pt in enumerate(npHullBottom4):
#     cv2.putText(im_orig, "[" + str(i) + "]", (pt[0], pt[1]), cv2.FONT_HERSHEY_PLAIN, .7, (0, 200, 0))

# outmost = omr_utils.order_points(outmost_points)
# (br, bl, tl, tr) = outmost
# (x, y, w, h) = cv2.boundingRect(np.array(outmost))
# cv2.rectangle(im_orig, (x, y), (x + w, y + h), (200, 10, 10), -1)
# print("(br, bl, tl, tr): ", (br, bl, tl, tr))
# cv2.drawContours(im_orig, [hull], -1, (20, 10, 200), 1)
#
# for i, pt in enumerate(outmost):
#     cv2.putText(im_orig, "[" + str(i) + "]", (pt[0], pt[1]), cv2.FONT_HERSHEY_PLAIN, .7, (0, 200, 0))
#     # cv2.putText(im_orig, "[" + str(i) + "]", (pt[0], pt[1]), cv2.FONT_HERSHEY_PLAIN, .7, (0, 200, 0))
#
# src_rect = np.array([tl, tr, br, bl], dtype="float32")

omr_region_img = None
if src_rect is not None:
    omr_region_img = omr_utils.four_point_transform(im, src_rect)
    omr_region_img = imutils.resize(omr_region_img, width=1000)
    omr_region_img = cv2.threshold(omr_region_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # print(im_orig.shape)

cv2.imshow("im_orig", im_orig)
# cv2.imwrite("../../resources/omr-imgs/20190103_152442_processed.jpg", im_orig)

if omr_region_img is not None:
    cv2.imshow("omr_region_img", omr_region_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
