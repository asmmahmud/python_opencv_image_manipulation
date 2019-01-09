import time
import cv2
import numpy as np
from omr_ana import helper_functions as omr_utils
import imutils

start_millis = int(round(time.time() * 1000))
file_name = "omr_m_8"
# file_name = "omr"
# ori_img = cv2.imread("../../resources/omr-imgs/final/" + file_name + "/output-region.jpg", cv2.IMREAD_COLOR)
ori_img = cv2.imread("../../resources/omr-imgs/final/" + file_name + ".jpg", cv2.IMREAD_COLOR)

print("ori_img.shape", ori_img.shape)
result = omr_utils.get_omr_answers(ori_img, 50)

print(result)
