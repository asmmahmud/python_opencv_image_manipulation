import cv2
import numpy as np
from omr_ana import omr as omr_utils
import imutils

# im_orig = cv2.imread("../../resources/omr-imgs/50.png", cv2.IMREAD_COLOR)
im_orig = cv2.imread("../../resources/omr-imgs/20190103_152442.jpg", cv2.IMREAD_COLOR)
# im_orig = cv2.imread("../../resources/omr-imgs/20190103_152500.jpg", cv2.IMREAD_COLOR)

im = cv2.cvtColor(im_orig, cv2.COLOR_BGR2GRAY)
im = omr_utils.normalize(im)
im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
contours = omr_utils.get_contours(im)
corners = omr_utils.get_corners(contours)

cv2.drawContours(im_orig, corners, -1, (200, 10, 10), -1)
outmost = omr_utils.order_points(omr_utils.get_outmost_points(corners))
(br, bl, tl, tr) = outmost
print("outmost: ", outmost)

src_rect = np.array([tl, tr, br, bl], dtype="float32")
omr_region_img = omr_utils.four_point_transform(im, src_rect)
omr_region_img = imutils.resize(omr_region_img, width=1000)
omr_region_img = cv2.threshold(omr_region_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

contours = omr_utils.get_contours(omr_region_img)
separator_contours = omr_utils.get_separator_boxes(contours, omr_region_img)
top_left_corners = [omr_utils.get_top_left_corner(cnt) for cnt in separator_contours]

omr_region_img_c = cv2.cvtColor(omr_region_img, cv2.COLOR_GRAY2BGR)
# cv2.drawContours(omr_region_img_c, contours, -1, (20, 10, 200), 1)

for top_left_corner in top_left_corners:
    print("top_left_corner: ", top_left_corner)
    # cv2.drawContours(transf_c, separator_contours, -1, (20, 10, 200), -1)
    cv2.circle(omr_region_img_c, top_left_corner, 5, (20, 10, 200), -1)

cv2.imshow("omr_region_img_c", omr_region_img_c)
cv2.imshow("im_orig", im_orig)

# cv2.imshow("omr_region_img", omr_region_img)
# cv2.imwrite("../../resources/omr-imgs/omr_region_img_50.png", omr_region_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
