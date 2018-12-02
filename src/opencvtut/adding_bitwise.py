import cv2 as cv
import numpy as np

img1 = cv.imread('../../resources/img/national_park.jpg')
# img1 = cv2.imread('../../resources/img/taapsee_pannu.jpg')
img2 = cv.imread('../../resources/img/opencv-logo-small.png')

rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]
# cv2.imshow("added_img3", roi)

# Now create a mask of logo and create its inverse mask also
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)

# Take only region of logo from logo image.
img2_fg = cv.bitwise_and(img2, img2, mask=mask)

# Now black-out the area of logo in ROI
img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)

# cv.imshow("mask", mask)
# cv.imshow("mask_inv", mask_inv)
# cv.imshow("img2_fg", img2_fg)
cv.imshow("img1_bg", img1_bg)


# Put logo in ROI and modify the main image
dst = cv.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst
cv.imshow('res', img1)
cv.waitKey(0)
cv.destroyAllWindows()

# dst = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
# dst2 = cv2.addWeighted(img1, 0.5, img2, 0.5, 5)
# dst3 = cv2.addWeighted(img1, 0.5, img2, 0.5, 150)
# cv2.imshow("added_img1", dst)
# cv2.imshow("added_img2", dst2)
# cv2.imshow("added_img3", dst3)
# cv.waitKey(0)
# cv.destroyAllWindows()
