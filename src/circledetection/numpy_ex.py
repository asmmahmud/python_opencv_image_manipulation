import numpy as np
import cv2
import imutils

img = cv2.imread("../../resources/omr-imgs/mahmud-vai5.png", 0)
bgr_img = imutils.resize(img, width=10)

# bgr_img = bgr_img - 255
circlePoints = bgr_img[np.where(bgr_img <= 255)]
print(circlePoints)
cv2.imshow("bgr_img", bgr_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
