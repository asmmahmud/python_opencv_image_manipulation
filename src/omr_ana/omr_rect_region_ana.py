import time
import cv2
import numpy as np
from omr_ana import helper_functions as omr_utils
import imutils

start_millis = int(round(time.time() * 1000))
file_name = "omr_m_5"
# file_name = "omr"
ori_img = cv2.imread("../../resources/omr-imgs/final/" + file_name + "/output-region.jpg", cv2.IMREAD_COLOR)

img = cv2.GaussianBlur(ori_img, (5, 5), 8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = omr_utils.normalize(img)
binary_mg = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
black_img = 255 - img

# Morphology Operation
# ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# dilete_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
# black_img = cv2.erode(black_img, erode_kernel, iterations=2)
# black_img = cv2.dilate(black_img, dilete_kernel, iterations=5)

region_width = int(img.shape[1] / 3)
left_region_right_x = region_width + 1
middle_region_right_x = (2 * region_width) + 1

contours = omr_utils.get_contours(black_img, cv2.RETR_EXTERNAL)

all_detected_contours = {}
left_points_group = []
middle_points_group = []
right_points_group = []

print_debug = False
for cnt in contours:
    cnt_feature = omr_utils.get_important_contour_featues(cnt, black_img)
    if omr_utils.is_a_circle(cnt_feature):
        cx, cy = omr_utils.get_centroid(cnt)
        all_detected_contours[str(cx) + str(cy)] = cnt
        # detected_centers.append((cx, cy))
        if cx < left_region_right_x:
            left_points_group.append([cx, cy])
            if print_debug:
                omr_utils.draw_point([cx, cy], ori_img)
        elif left_region_right_x < cx < middle_region_right_x:
            middle_points_group.append([cx, cy])
            if print_debug:
                omr_utils.draw_point([cx, cy], ori_img)
        elif middle_region_right_x < cx:
            right_points_group.append([cx, cy])
            if print_debug:
                omr_utils.draw_point([cx, cy], ori_img)

        # if print_debug:
        # print("found", cnt_feature)
    elif print_debug and 30 < cnt_feature['height'] < 50:
        print("not found", cnt_feature)

final_np_array_points = np.array([])
np3dArr1 = omr_utils.arrange_points_according_to_question(left_points_group, False, ori_img)
np3dArr2 = omr_utils.arrange_points_according_to_question(middle_points_group, False, ori_img)
np3dArr3 = omr_utils.arrange_points_according_to_question(right_points_group, False, ori_img)

if np3dArr1.shape[0] != 0 and np3dArr2.shape[0] != 0 and np3dArr3.shape[0] != 0:
    final_np_array_points = np.concatenate((np3dArr1, np3dArr2, np3dArr3))

print(final_np_array_points.shape)

if final_np_array_points.shape[0] != 0:
    answer_list = np.zeros((final_np_array_points.shape[0], final_np_array_points.shape[1]), dtype=bool)

    i_length = final_np_array_points.shape[0]
    j_length = final_np_array_points.shape[1]

    for i in range(i_length):
        for j in range(j_length):
            (cx, cy) = final_np_array_points[i][j]
            block14_contour = all_detected_contours[str(cx) + str(cy)]
            if block14_contour is not None:
                (x, y, w, h) = cv2.boundingRect(block14_contour)
                block = ori_img[y:y + h, x:x + w]
                # block = omr_utils.get_ans_block(final_np_array_points, all_detected_contours, ori_img, i + 1, j + 1)
                processed_img = omr_utils.operate_on_circle_block(block)
                if processed_img is not None:
                    mean_value = processed_img.mean()
                    answer_list[i, j] = 150 < mean_value

            if print_debug:
                omr_utils.draw_text(ori_img, str(i + 1) + "/" + str(j + 1), final_np_array_points[i, j])

    print(answer_list)

if print_debug:

    cv2.imshow("thresh_image", img)

cv2.imshow("ori_img", ori_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

end_millis = int(round(time.time() * 1000))

diff_millis = end_millis - start_millis
print(diff_millis)

exit(0)
