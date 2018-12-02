import cv2
import imutils
import tkinter
import PIL


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

cv2.createTrackbar('Binary_Thresholding', 'image_thresholding', 0, 1, nothing)
cv2.createTrackbar('Adaptive_Thresholding', 'image_thresholding', 0, 1, nothing)
cv2.createTrackbar('Otsu_Binary_Thresholding', 'image_thresholding', 0, 1, nothing)

thresh1, thresh2, isBinary, isAdaptive, isOtsu = -1, -1, -1, -1, -1

while True:
	
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break
	# get current positions of four trackbars
	r = cv2.getTrackbarPos('Threshold_1', 'image')
	g = cv2.getTrackbarPos('Threshold_2', 'image')
	b = cv2.getTrackbarPos('Binary_Thresholding', 'image')
	b = cv2.getTrackbarPos('Binary_Thresholding', 'image')
	# s = cv2.getTrackbarPos(switch, 'image')
	#
	# if pb != b or pg != g or pr != r:
	# 	if s == 0:
	# 		img[:] = 0
	# 	else:
	# 		img[:] = [b, g, r]
	#
	# 	print(s, r, g, b)
	# 	pb, pg, pr = b, g, r
	# 	cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)

