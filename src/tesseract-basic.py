# tabular_data1
import os

import cv2
import numpy as np

try:
    from PIL import Image
except ImportError:
    import Image

import pytesseract

# img_path = 'national-id-card3.jpg'
img_path = "../resources/img/national-id-card3.jpg"

img = cv2.imread(img_path)

# # Extract the file name without the file extension
# file_name = os.path.basename(img_path).split('.')[0]
# file_name = file_name.split()[0]

output_dir = "./output_dir"
# output_path = os.path.join(output_dir, file_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply dilation and erosion to remove some noise
kernel = np.ones((1, 1), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)
img = cv2.erode(img, kernel, iterations=1)

# Apply blur to smooth out the edges
img = cv2.GaussianBlur(img, (5, 5), 0)

otsu_threshold_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
threshold_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]

# cv2.imshow("before", img)
cv2.imshow("threshold", threshold_img)
cv2.imshow("otsu_threshold_img", otsu_threshold_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save_path = os.path.join(output_dir, img_path)
# cv2.imwrite(save_path, img)

# print(pytesseract.image_to_string(Image.fromarray(img), lang='ben+eng', config="--tessdata-dir ./tessdata --psm 3"))
# print(pytesseract.get_tesseract_version())

# print(pytesseract.image_to_string(img, 'ben', '--psm 3', 0))
