import time
import cv2
import numpy as np
from omr_ana import helper_functions as omr_utils
import imutils

start_millis = int(round(time.time() * 1000))
file_name = "omr_m_8"
# file_name = "omr"
# ori_img = cv2.imread("../../resources/omr-imgs/final/" + file_name + "/output-region.jpg", cv2.IMREAD_COLOR)
ori_img = cv2.imread("../../resources/omr-imgs/final/" + file_name + ".jpg", cv2.IMREAD_COLOR)

img = cv2.GaussianBlur(ori_img, (5, 5), 8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = omr_utils.normalize(img)
# binary_mg = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
c_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

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

print_debug, contour_debug, draw_num_debug = True, True, True
for cnt in contours:
    cnt_feature = omr_utils.get_important_contour_featues(cnt, black_img)
    if omr_utils.is_a_circle(cnt_feature):
        cx, cy = omr_utils.get_centroid(cnt)
        all_detected_contours[str(cx) + str(cy)] = cnt
        # detected_centers.append((cx, cy))
        if cx < left_region_right_x:
            left_points_group.append([cx, cy])
            if contour_debug:
                omr_utils.draw_point([cx, cy], c_img)
        elif left_region_right_x < cx < middle_region_right_x:
            middle_points_group.append([cx, cy])
            if contour_debug:
                omr_utils.draw_point([cx, cy], c_img)
        elif middle_region_right_x < cx:
            right_points_group.append([cx, cy])
            if contour_debug:
                omr_utils.draw_point([cx, cy], c_img)

        # if print_debug:
        # print("found", cnt_feature)
    elif print_debug and 15 < cnt_feature['height'] < 90:
        print("not found", cnt_feature)

final_np_array_points = np.array([])

np3dArr1 = omr_utils.arrange_points_according_to_question(left_points_group)
np3dArr2 = omr_utils.arrange_points_according_to_question(middle_points_group)
np3dArr3 = omr_utils.arrange_points_according_to_question(right_points_group)

if np3dArr1.shape[0] != 0 and np3dArr2.shape[0] != 0 and np3dArr3.shape[0] != 0:
    final_np_array_points = np.concatenate((np3dArr1, np3dArr2, np3dArr3))

print("final_np_array_points: ", final_np_array_points.shape)

if final_np_array_points.shape[0] != 0:
    answer_list = np.zeros((final_np_array_points.shape[0], final_np_array_points.shape[1]), dtype='float16')

    i_length = final_np_array_points.shape[0]
    j_length = final_np_array_points.shape[1]

    t_r, t_c = 46, 1
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
                    unique_vals, counts = np.unique(processed_img, return_counts=True)
                    index_dict = dict(zip(unique_vals, counts))
                    # mean_value = processed_img.mean()

                    total_pix = processed_img.shape[0] * processed_img.shape[1]
                    percet_of_white = (index_dict[255] / float(total_pix) * 100)
                    answer_list[i, j] = percet_of_white
                    if 50 < percet_of_white:
                        cv2.circle(c_img, (cx, cy), 20, (10, 200, 10), 3)

                    if i == t_r and j == t_c:
                        cv2.imshow(str(i) + str(j) + "_processed", processed_img)
                        print(str(i) + "/" + str(j) + ": ", percet_of_white)
                        # cv2.imshow(str(i)+str(j)+"_original", block)

            if draw_num_debug:
                omr_utils.draw_text(c_img, str(i + 1) + "/" + str(j + 1), final_np_array_points[i, j])

    if print_debug:
        print(answer_list)

cv2.namedWindow("c_img", cv2.WINDOW_NORMAL)
cv2.imshow("c_img", c_img)

cv2.namedWindow("ori_img", cv2.WINDOW_NORMAL)
cv2.imshow("ori_img", ori_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

end_millis = int(round(time.time() * 1000))

diff_millis = end_millis - start_millis
print(diff_millis)

exit(0)
