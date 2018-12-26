import cv2
import imutils
from tkinter import *
from PIL import Image
from PIL import ImageTk

def nothing(x):
	pass

image_path = "../../../resources/img/nid5.jpg"

ori_img = cv2.imread(image_path)
if ori_img is None:
	exit(0)

ori_img = imutils.resize(ori_img, width=1024)
ori_img = cv2.GaussianBlur(ori_img, (5, 5), 0)

ori_gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('image_thresholding')
cv2.createTrackbar('Threshold_1', 'image_thresholding', 0, 255, nothing)
cv2.createTrackbar('Threshold_2', 'image_thresholding', 0, 255, nothing)
cv2.createTrackbar('Thresholding_Methods', 'image_thresholding', 0, 2, nothing)

old_thresh1, old_thresh2, old_thresh_method = -1, -1, -1

# cv2.imshow('image_thresholding', ori_gray)

while True:
	
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break
	# get current positions of 3 trackbars
	thresh1 = cv2.getTrackbarPos('Threshold_1', 'image_thresholding')
	thresh2 = cv2.getTrackbarPos('Threshold_2', 'image_thresholding')
	thresh_method = cv2.getTrackbarPos('Thresholding_Methods', 'image_thresholding')
	
	if old_thresh1 != thresh1 or old_thresh2 != thresh2 or old_thresh_method != thresh_method:
		if thresh_method == 0:
			ori_gray = cv2.threshold(ori_gray.copy(), thresh1, thresh2, cv2.THRESH_BINARY)[1]
		elif thresh_method == 1:
			ori_gray = cv2.threshold(ori_gray.copy(), thresh1, thresh2, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		elif thresh_method == 2:
			ori_gray = cv2.adaptiveThreshold(ori_gray.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)[1]
			
		ori_gray = cv2.threshold(ori_gray.copy(), int(thresh1), int(thresh2), cv2.THRESH_BINARY)[1]
		print(thresh1, thresh2, thresh_method)
		old_thresh1, old_thresh2, old_thresh_method = thresh1, thresh2, thresh_method
		ori_gray = Image.fromarray(ori_gray)
		# cv2.imshow('image_thresholding', ori_gray)
		ori_gray = ImageTk.PhotoImage(ori_gray)

cv2.destroyAllWindows()
exit(0)
