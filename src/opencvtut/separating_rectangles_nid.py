import numpy as np
import imutils
import cv2
import pytesseract

image_paths = [
	"../../resources/img/nid1.jpg",
	"../../resources/img/nid2.jpg",
	"../../resources/img/nid3.jpg",
	"../../resources/img/nid4.jpg",
	"../../resources/img/nid5.jpg",
	"../../resources/img/nid6.jpg",
	"../../resources/img/nid7.jpg",
	"../../resources/img/nid8.jpg",
]
ori_img = cv2.imread(image_paths[0])

if ori_img is None:
	exit(0)

ori_img = imutils.resize(ori_img, width=1024)
ori_img = cv2.GaussianBlur(ori_img, (5, 5), 0)
ori_gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

img_bin = cv2.threshold(ori_gray, 200, 255, cv2.THRESH_BINARY)[1]
img_bin = 255 - img_bin  # Invert the image

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img_bin = cv2.dilate(img_bin, kernel, iterations=2)
img_bin = cv2.erode(img_bin, kernel, iterations=2)
img_bin = cv2.dilate(img_bin, kernel, iterations=4)
img_bin = cv2.erode(img_bin, kernel, iterations=3)

cnts = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

rois = []
for cnt in cnts:
	(x, y, w, h) = cv2.boundingRect(cnt)
	ar = w / float(h)
	crWidth = w / float(img_bin.shape[1])
	
	if 100 < w  and 100 < h :
		# rois.append(ori_gray[y:y + h, x: x + w])
		print(x, y, w, h)
		cv2.rectangle(img_bin, (x, y), (x + w, y + h), (0, 0), 5)
	elif 800 < w < 950 and 380 < h < 420:
		# cv2.rectangle(img_bin, (x, y), (x + w, y + h), (0, 0), 5)
		rois.append(ori_gray[y:y + h, x: x + w])

if rois is not None and len(rois) > 0:
	final_img = rois[0].copy()
	# img_bin = cv2.equalizeHist(img_bin)
	cv2.imshow("final_img", final_img)
else:
	lines = cv2.HoughLinesP(img_bin.copy(), 1, np.pi / 180, 180, minLineLength=200, maxLineGap=15)
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line[0]
			if x2 - x1 > 400 and y2 - y1 < 50:
				cv2.line(img_bin, (x1, y1), (x2, y2), (255, 255), 5)
			elif y2 - y1 > 200 and x2 - x1 < 50:
				cv2.line(img_bin, (x1, y1), (x2, y2), (255, 255), 5)

cv2.imshow("hist_eq_img", img_bin)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)

# redundant functions

# kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
# square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
# img_bin = cv2.adaptiveThreshold(img_bin, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
# img_bin = cv2.Canny(img_bin, 50, 150)
# img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel_y, iterations=4)
# img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_y, iterations=2)

# lines = cv2.HoughLinesP(img_bin.copy(), 1, np.pi / 180, 180, minLineLength=200, maxLineGap=10)
# if lines is not None:
# 	for line in lines:
# 		x1, y1, x2, y2 = line[0]
# 		if x2 - x1 > 400 and y2 - y1 < 50:
# 			cv2.line(ori_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
# 		elif y2 - y1 > 200 and x2-x1 < 50:
# 			cv2.line(ori_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# cv2.imshow("img_thresh", img_bin)
# # cv2.imshow("ori_img", ori_img)
#
