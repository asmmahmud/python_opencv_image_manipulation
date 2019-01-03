import numpy as np
import cv2
import imutils
import circledetection.omrutils as omrutils

img = cv2.imread("../../resources/omr-imgs/omr-mobile-1.jpg")
im_orig = imutils.resize(img, width=1200)

img = cv2.GaussianBlur(im_orig, (5, 5), 8)
img = omrutils.normalize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

# cv2.imshow("img", imutils.resize(img, width=800)[120:, ])

contours = omrutils.get_contours(img)
rect_list = []
rect_points_clockwise_list = []
for cnt in contours:
    (x, y, w, h) = cv2.boundingRect(cnt)
    aspect_ratio = float(h) / float(w)

    if 100 < w < 700 and 500 < h < 1200 and 2.5 < aspect_ratio < 3.0:
        print(x, y, w, h, aspect_ratio)
        npCnt = np.array(cnt, dtype="float32")
        print(npCnt.shape)
        if npCnt.shape[0] == 4:
            point2d = npCnt[:, 0]
            point2dSorted = point2d[point2d[:, 1].argsort()]
            rect_points_clockly = omrutils.sort_points(point2dSorted)
            rect = omrutils.four_point_transform(img, rect_points_clockly)
            rect_points_clockwise_list.append(rect_points_clockly[0][0])
            rect_list.append(rect)
            cv2.drawContours(im_orig, [cnt], -1, (0, 255, 0), 3)

# print("rect_points_clockly_list: ", rect_points_clockly_list)
combined_list = zip(rect_list, rect_points_clockwise_list)
combined_list = sorted(combined_list, key=lambda b: b[1], reverse=False)
(rect_list, rect_points_clockwise_list) = zip(*combined_list)

print(len(rect_list))

if len(rect_list) >= 2:
    lenss = 1
    for rectang in rect_list:
        cv2.imshow("rect-" + str(lenss), rectang)
        lenss += 1
        # cv2.imshow("rect-2", omrutils.find_rect_by_ratio(rect_list[1], 80, 25, 1.2, .8))

    cv2.imshow("main", im_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

exit(0)
