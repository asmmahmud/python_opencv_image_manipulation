import cv2
import imutils

imagePath = "../../resources/img/tabular_data2_ori.png"
ori_img = cv2.imread(imagePath)
ori_img = imutils.resize(ori_img, height=1024)
ori_gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

horizon_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

ori_img_hist = cv2.equalizeHist(ori_gray)
ori_img_hist = cv2.GaussianBlur(ori_img_hist, (3, 3), 1)
ori_img_hist_thresh = cv2.threshold(ori_img_hist, 80, 255, cv2.THRESH_BINARY)[1]
ori_img_hist_thresh = cv2.erode(ori_img_hist_thresh, horizon_kernel, iterations=2)
ori_img_hist_thresh = cv2.dilate(ori_img_hist_thresh, horizon_kernel, iterations=2)

con_img, contours, heir = cv2.findContours(ori_img_hist_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
boundingBoxes = [cv2.boundingRect(c) for c in contours]
combined = zip(contours, boundingBoxes)
combined = sorted(combined, key=lambda b: b[1][2] * b[1][3], reverse=True)
(contours, boundingBoxes) = zip(*combined)

table_roi = ""
for cnt in contours:
	(x, y, w, h) = cv2.boundingRect(cnt)
	width_ratio = w / float(ori_img_hist_thresh.shape[1])
	height_ratio = h / float(ori_img_hist_thresh.shape[0])
	# print(x, y, w, h, w * h)
	if .6 < width_ratio < .95 and .5 < height_ratio < .95:
		print("--", x, y, w, h, w * h)
		# cv2.rectangle(ori_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		table_roi = ori_gray[y: y + h, x:x + w]

if type(table_roi) is str:
	table_roi = ori_gray

table_roi = cv2.threshold(table_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
table_roi_thresh = cv2.erode(table_roi, sq_kernel, iterations=4)
table_roi_thresh = cv2.dilate(table_roi_thresh, sq_kernel, iterations=2)

con_img, contours, heir = cv2.findContours(table_roi_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
boundingBoxes = [cv2.boundingRect(c) for c in contours]
combined = zip(contours, boundingBoxes)
combined = sorted(combined, key=lambda b: b[1][0], reverse=False)
combined = sorted(combined, key=lambda b: b[1][1], reverse=False)

(contours, boundingBoxes) = zip(*combined)

# iterate through sorted contours
num_of_rows = 0
cur_y = -1
text_arr = []
for cnt in contours:
	(x, y, w, h) = cv2.boundingRect(cnt)
	print(x, y, w, h)
	aspect_ratio = w / float(h)
	width_ratio = w / float(table_roi.shape[1])
	cv2.rectangle(table_roi, (x, y), (x+w, y+h), (0, 0), 1)
	if y - cur_y > 5:
		cur_y = y
		num_of_rows += 1

print(num_of_rows)
cv2.imshow("table_roi_ori", table_roi)
# cv2.imshow("table_roi_thresh", table_roi_thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
