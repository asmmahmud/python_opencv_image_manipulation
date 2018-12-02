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

kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 8))
kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, (10, 1))

for (idx, impath) in enumerate(image_paths):
	img = cv.imread(impath, cv.IMREAD_GRAYSCALE)
	img = imutils.resize(img, width=1024)
	y_to_omit = int(img.shape[1] / 5)
	x_to = int(img.shape[0] / 2.1)
	
	morph_img = cv.threshold(img[y_to_omit:, 0: x_to], 180, 255, cv.THRESH_BINARY)[1]
	morph_img = 255 - morph_img
	
	morph_img = cv.dilate(morph_img, kernel, iterations=4)
	morph_img = cv.erode(morph_img, kernel, iterations=2)
	
	con_img, contours, heir = cv.findContours(morph_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	
	# boundingBoxes = [cv.boundingRect(c) for c in contours]
	# combined = zip(contours, boundingBoxes)
	# combined = sorted(combined, key=lambda b: b[1][2] / b[1][3], reverse=True)
	# # combined = sorted(combined, key=lambda b: b[1][1], reverse=False)
	# (contours, boundingBoxes) = zip(*combined)
	
	for cnt in contours:
		(x, y, w, h) = cv.boundingRect(cnt)
		aspect_ratio = w / float(h)
		if 100 < w < 600 and 200 < h < 600 and .5 < aspect_ratio <= 1.1:
			print(x, y, w, h, aspect_ratio)
			cv.rectangle(img, (x, y_to_omit + y), (x + w, y_to_omit + y + h), (0, 0), 2)
			cv.rectangle(morph_img, (x, y), (x + w, y + h), (100, 100), 1)
		# pics.append(morph_img[y: y + h, x:x + w])
		if 100 < w < 600 and 100 < h < 600 and 1 <= aspect_ratio < 2:
			# pics.append(morph_img[y: y + h, x:x + w])
			print(x, y, w, h, aspect_ratio)
			cv.rectangle(img, (x, y_to_omit + y), (x + w, y_to_omit + y + h), (0, 0), 2)
			cv.rectangle(morph_img, (x, y), (x + w, y + h), (100, 100), 1)
	
	cv.imshow('img' + str(idx), img)

cv.waitKey(0)
cv.destroyAllWindows()
exit(0)

# final_regions = []
# for impath in image_paths:
# 	img = cv.imread(image_paths[4], cv.IMREAD_GRAYSCALE)
# 	if img is None:
# 		continue
#
