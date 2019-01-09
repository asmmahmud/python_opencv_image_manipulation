import cv2
import numpy as np
from omr_ana import helper_functions as omr_utils
import imutils

file_name = "omr_m_7"
# file_name = "omr"
ori_img = cv2.imread("../../resources/omr-imgs/final/" + file_name + "/output-region.jpg", cv2.IMREAD_COLOR)
img = cv2.GaussianBlur(ori_img, (9, 9), 10)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = omr_utils.normalize(img)

binary_mg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

cv2.imshow("binary_mg", binary_mg)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
