import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt
import pytesseract

image_paths = [
	# "../../resources/img/nid1.jpg",
	# "../../resources/img/nid2.jpg",
	# "../../resources/img/nid3.jpg",
	# "../../resources/img/nid4.jpg",
	# "../../resources/img/nid5.jpg",
	"../../resources/img/nid6.jpg",
	"../../resources/img/nid7.jpg",
	# "../../resources/img/nid8.jpg",
	# "../../resources/img/nid9.jpg",
]

for idx, img_path in enumerate(image_paths):
	
	ori_img = cv2.imread(img_path)
	
	if ori_img is None:
		exit(0)
	
	ori_img = imutils.resize(ori_img, width=1024)
	ori_img = cv2.GaussianBlur(ori_img, (3, 3), 0)
	
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))
	ori_img = cv2.erode(ori_img, kernel, iterations=3)
	ori_img = cv2.dilate(ori_img, kernel, iterations=3)
	
	frame_HSV = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV)
	mask1 = cv2.inRange(frame_HSV, (0, 70, 50), (10, 255, 255))
	mask2 = cv2.inRange(frame_HSV, (170, 70, 50), (180, 255, 255))
	
	output_img = mask1 | mask2
	
	output_mask = cv2.bitwise_or(mask1, mask2)
	target = cv2.bitwise_and(ori_img, ori_img, mask=output_mask)
	
	cv2.imshow(img_path , target)

cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
