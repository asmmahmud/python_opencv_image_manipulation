import cv2
import numpy as np
from omr_ana import omr as omr_utils

# im_orig = cv2.imread("../../resources/omr-imgs/corner_block.png")
# im_orig = cv2.imread("../../resources/omr-imgs/rectangle.png")

im_orig = cv2.imread("../../resources/omr-imgs/50_only_corners.png", cv2.IMREAD_COLOR)

# blurred = cv2.GaussianBlur(im_orig, (5, 5), 8)
im = cv2.cvtColor(im_orig, cv2.COLOR_BGR2GRAY)
im = omr_utils.normalize(im)
im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

print("image shape: ", im.shape)
print("image height: ", im.shape[0])
print("image width: ", im.shape[1])
contours = omr_utils.get_contours(im)

# contours_features = [omr_utils.get_features_for_box_separation(cnt, im) for cnt in contours]
contours_features = []
for cnt in contours:
    cnt_feature = omr_utils.get_features_for_box_separation(cnt, im)
    print("cnt: ", cnt_feature)
    contours_features.append(cnt_feature)

# print(contours_features)
# print("Before: ", contours)
#
# contours = [cnt for i, cnt in enumerate(contours) if contours_features[i][0] != 1.0 and
# contours_features[i][1] != 1.0 and contours_features[i][2] != 1.0 and contours_features[i][3] != 1.0]

# print(contours_features)
# print("After: ", contours)

c_img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
cv2.drawContours(c_img, contours, -1, (20, 10, 200), thickness=1)
cv2.imshow("c_img", c_img)

# cv2.imshow("im", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
