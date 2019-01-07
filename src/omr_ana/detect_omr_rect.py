import cv2
import numpy as np
from omr_ana import helper_functions as omr_utils
import imutils

# im_orig = cv2.imread("../../resources/omr-imgs/corner_block.png")
# im_orig = cv2.imread("../../resources/omr-imgs/rectangle.png")
file_name = "omr"
im_orig = cv2.imread("../../resources/omr-imgs/final/" + file_name + ".jpg", cv2.IMREAD_COLOR)
im_orig = imutils.resize(im_orig, width=1200)

omr_region_img, c_img, edges_im = omr_utils.get_omr_region(im_orig, True)
if omr_region_img is not None:
    cv2.imwrite("../../resources/omr-imgs/final/" + file_name + "/output-region.jpg", omr_region_img)


cv2.imwrite("../../resources/omr-imgs/final/" + file_name + "/output.jpg", c_img)
cv2.cv2.imwrite("../../resources/omr-imgs/final/" + file_name + "/output-edges.jpg", edges_im)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
