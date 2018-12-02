import cv2 as cv
import numpy as np
import imutils

image_paths = [
	"../../resources/img/nid1.jpg",
	"../../resources/img/nid2.jpg",
	"../../resources/img/nid3.jpg",
	"../../resources/img/nid4.jpg",
	"../../resources/img/nid5.jpg",
]

square_kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 1))

for (idx, impath) in enumerate(image_paths):
	ori_img = cv.imread(impath)
	ori_img = imutils.resize(ori_img, width=1024)
	gray_frame = cv.cvtColor(ori_img, cv.COLOR_BGR2GRAY)
	
	ori_img_morph = cv.erode(ori_img, square_kernel, iterations=4)
	ori_img_morph = cv.dilate(ori_img_morph, square_kernel, iterations=1)
	
	frame_hsv = cv.cvtColor(ori_img_morph, cv.COLOR_BGR2HSV)
	
	color_mask1 = cv.inRange(frame_hsv, (0, 70, 50), (10, 255, 255))
	color_mask2 = cv.inRange(frame_hsv, (170, 70, 50), (180, 255, 255))
	
	output_mask = cv.bitwise_or(color_mask1, color_mask2)
	# target = cv.bitwise_and(ori_img, ori_img, mask=output_mask)
	
	output_mask = cv.erode(output_mask, square_kernel, iterations=2)
	output_mask = cv.dilate(output_mask, square_kernel, iterations=4)
	output_mask = cv.dilate(output_mask, rect_kernel, iterations=4)
	
	con_img, contours, heir = cv.findContours(output_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		(x, y, w, h) = cv.boundingRect(cnt)
		aspect_ratio = w / float(h)
		if 400 < w < 900 and 40 < h < 150 and 5 < aspect_ratio >= 7:
			print(x, y, w, h, aspect_ratio)
			cv.rectangle(gray_frame, (x, y), (x + w, y + h), (0, 0), 1)

	cv.imshow(impath, gray_frame)

cv.waitKey(0)
cv.destroyAllWindows()
exit(0)
