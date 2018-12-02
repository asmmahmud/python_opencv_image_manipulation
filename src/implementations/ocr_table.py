import numpy as np
import imutils
import cv2
import pytesseract

imagePath = "../../resources/img/table_data_with_long_text.png"
ori_img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
img = imutils.resize(ori_img, height=1024)

img = cv2.GaussianBlur(img, (3, 3), 0)
otsu_threshold_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

kernel = np.ones((2, 2), np.uint8)
otsu_threshold_img = cv2.dilate(otsu_threshold_img, kernel, iterations=2)
otsu_threshold_img = cv2.erode(otsu_threshold_img, kernel, iterations=2)

# edges = cv2.Canny(otsu_threshold_img, 80, 150, L2gradient=True)

img, contours, heir = cv2.findContours(otsu_threshold_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# loop over the contours and calculate bounding boxes
boundingBoxes = [cv2.boundingRect(c) for c in contours]
combined = zip(contours, boundingBoxes)
combined = sorted(combined, key=lambda b: b[1][0], reverse=False)
combined = sorted(combined, key=lambda b: b[1][1], reverse=False)

(contours, boundingBoxes) = zip(*combined)

# image container
container_width = 300
container_height = 70000
img_coltxt_container = np.zeros([container_height, container_width])

cur_y = 0
prev_y = 0
row_num = 0
text_height = 50

# iterate through sorted contours
for c in contours:
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	crWidth = w / float(ori_img.shape[1])
	
	if (200 < w < 300) or (200 < h < 300):
		# cv2.rectangle(ori_img, (x, y), (x + w, y + h), color=[20, 20, 255], thickness=3)
		roi = otsu_threshold_img[y:y + h, x:x + w]
		# img_combined[cur_y: cur_y + h, 0: w] = cv2.add(img_combined[cur_y: cur_y + h, 0: w], roi)
		if y - prev_y > 5:
			row_num += 1
		prev_y = y
		print("row#" + str(row_num) + "#", prev_y, roi.shape)
		cv2.rectangle(img_coltxt_container, (0, cur_y), (container_width, cur_y + text_height), (255, 255, 255), -1)
		cv2.putText(img_coltxt_container,
					"(row#" + str(row_num) + ")",
					(0, cur_y + text_height - 30),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.8,
					(0, 0, 0), 2, cv2.LINE_AA)
		img_coltxt_container[cur_y + text_height: cur_y + text_height + roi.shape[0], 0: roi.shape[1]] = roi.copy()
		cur_y += (h + text_height)
		print("-------------")

final_img = img_coltxt_container[:cur_y, :]

cv2.imwrite('../../resources/output_dir/tableoutput.png', final_img)
cv2.imshow("gradX_thresh", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print(pytesseract.image_to_string(final_img, lang='eng', config="--tessdata-dir ../tessdata --psm 3"))
exit(0)
