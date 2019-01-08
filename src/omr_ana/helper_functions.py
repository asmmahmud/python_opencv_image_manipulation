import argparse
import cv2
import math
import numpy as np

DETECT_CORNER_POINT = 1
DETECT_INNER_BOX = 2

CORNER_FEATS = (
    0.322965313273202,
    0.19188334690998524,
    1.1514327482234812,
    0.998754685666376,
)

CORNER_FEATS_MINE = (
    0.6369426751592356,
    0.43383947939262474,
    1.1091556449705795,
    0.9675487306284019,
    24.057142857142857,
    21.25,
    1.25,
)

BOX_SEPERATION_FEATURE = (
    0.562893900723996,
    0.5332050522991908,
    1.0201561929985838,
    1.0070849282278802,
    3.2564935064935066,
    29.41176470588235,
    9.058823529411764
)

TRANSF_SIZE = 1024


def get_convex_hull(contour):
    return cv2.convexHull(contour)


def get_contour_area_by_hull_area(contour):
    return (cv2.contourArea(contour) /
            cv2.contourArea(get_convex_hull(contour)))


def get_contour_area_by_bounding_box_area(contour):
    return (cv2.contourArea(contour) /
            cv2.contourArea(get_bounding_rect(contour)))


def get_contour_perim_by_hull_perim(contour):
    return (cv2.arcLength(contour, True) /
            cv2.arcLength(get_convex_hull(contour), True))


def get_contour_perim_by_bounding_box_perim(contour):
    return (cv2.arcLength(contour, True) /
            cv2.arcLength(get_bounding_rect(contour), True))


def get_full_height_by_contour_height(contour, full_height):
    (x, y, w, h) = cv2.boundingRect(contour)
    return full_height / float(h)


def get_full_width_by_contour_width(contour, full_width):
    (x, y, w, h) = cv2.boundingRect(contour)
    return full_width / float(w)


def get_bounding_rect_aspect_ratio(contour):
    (x, y, w, h) = cv2.boundingRect(contour)
    return h / float(w)


def get_important_contour_featues(contour, img):
    (x, y, w, h) = cv2.boundingRect(contour)
    full_width = img.shape[1]
    full_height = img.shape[0]

    try:
        return {
            "hull_area_ratio": get_contour_area_by_hull_area(contour),
            "bounding_area_ratio": get_contour_area_by_bounding_box_area(contour),
            "hull_perim_ratio": get_contour_perim_by_hull_perim(contour),
            "hull_bounding_perim_ratio": get_contour_perim_by_bounding_box_perim(contour),
            "image_height_ratio": h / float(full_height),
            "image_width_ratio": w / float(full_width),
            "aspect_ratio": h / float(w),
            "height": h,
            "width": w
        }
    except ZeroDivisionError:
        return {
            "hull_area_ratio": float("inf"),
            "bounding_area_ratio": float("inf"),
            "hull_perim_ratio": float("inf"),
            "hull_bounding_perim_ratio": float("inf"),
            "image_height_ratio": float("inf"),
            "image_width_ratio": float("inf"),
            "aspect_ratio": float("inf"),
            "height": float("inf"),
            "width": float("inf"),
        }
        # return 7 * [np.inf]


def get_features_for_box_separation(contour, img):
    (x, y, w, h) = cv2.boundingRect(contour)
    full_width = img.shape[1]
    full_height = img.shape[0]

    try:
        return (
            get_contour_area_by_hull_area(contour),
            get_contour_area_by_bounding_box_area(contour),
            get_contour_perim_by_hull_perim(contour),
            get_contour_perim_by_bounding_box_perim(contour),
            full_height / float(h),
            full_width / float(w),
            h / float(w)
        )
    except ZeroDivisionError:
        return 7 * [np.inf]


def get_features(contour):
    try:
        return (
            get_contour_area_by_hull_area(contour),
            get_contour_area_by_bounding_box_area(contour),
            get_contour_perim_by_hull_perim(contour),
            get_contour_perim_by_bounding_box_perim(contour),
        )
    except ZeroDivisionError:
        return 4 * [np.inf]


def get_bounding_rect(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)


def get_top_left_corner(contour):
    (x, y, w, h) = cv2.boundingRect(contour)
    return x, y


def features_distance(f1, f2):
    f1 = np.array(f1)
    f2 = np.array(f2)
    return np.linalg.norm(f1 - f2)


def features_distance_for_box(f1, f2):
    f1 = np.array(f1)
    f2 = np.array(f2)
    return np.linalg.norm(f1 - f2)


# Default mutable arguments should be harmless here
def draw_point(point, img, radius=5, color=(0, 0, 255)):
    cv2.circle(img, tuple(point), radius, color, -1)


def draw_text(img, draw_str, pt, padd=0):
    cv2.putText(img, "[" + draw_str + "]", (pt[0] + padd, pt[1] + padd), cv2.FONT_HERSHEY_PLAIN, .7, (0, 0, 200))


def get_centroid(contour):
    m = cv2.moments(contour)
    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])
    return x, y


def normalize(im):
    return cv2.normalize(im, np.zeros(im.shape), 0, 255, norm_type=cv2.NORM_MINMAX)


def get_approx_contour(contour, tol=.01):
    """Get rid of 'useless' points in the contour"""
    epsilon = tol * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def get_contours(image_gray, mode=cv2.RETR_TREE):
    im2, contours, hierarchy = cv2.findContours(
        image_gray, mode, cv2.CHAIN_APPROX_SIMPLE)

    return [get_approx_contour(cnt) for cnt in contours]
    # return map(get_approx_contour, contours)


def get_corners(contours, img):
    return sorted(
        contours,
        key=lambda c: features_distance_for_box(CORNER_FEATS_MINE, get_features_for_box_separation(c, img)))[:4]


def get_separator_boxes(contours, img):
    return sorted(
        contours,
        key=lambda c: features_distance_for_box(BOX_SEPERATION_FEATURE, get_features_for_box_separation(c, img)))[:2]


def order_points(points):
    """Order points counter-clockwise-ly."""
    origin = np.mean(points, axis=0)

    def positive_angle(p):
        x, y = p - origin
        ang = np.arctan2(y, x)
        return 2 * np.pi + ang if ang < 0 else ang

    return sorted(points, key=positive_angle)


def get_outmost_points_old(contours):
    # print("Before", contours)
    all_points = np.concatenate(contours)
    print("all_points", all_points)
    return get_bounding_rect(all_points)


def get_convex_hull_points(contours):
    all_points = np.concatenate(contours)
    hull_points = cv2.convexHull(all_points, False)
    npHull3d = np.array(hull_points)
    return npHull3d


def get_outmost_points_by_summation(hull_points):
    pts = hull_points[:, 0]
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


# this is buggy
def get_outmost_points(hull_points):
    npHull2d = hull_points[:, 0]
    # print("npHull2d: ", npHull2d.shape)
    if npHull2d.shape[0] <= 3:
        return None

    # sort according to y coord
    npHull2dSorted = npHull2d[npHull2d[:, 1].argsort()]

    # separating top and bottom points
    npHullTop4 = npHull2dSorted[:4, :]
    npHull2dSortedRev = npHull2dSorted[::-1, :]
    npHullBottom4 = npHull2dSortedRev[:4, :]
    # sort top and bottom most points according to x coord
    npHullTop4 = npHullTop4[npHullTop4[:, 0].argsort()]
    npHullBottom4 = npHullBottom4[npHullBottom4[:, 0].argsort()]

    # extracting 4 corner points
    top_left = npHullTop4[0]
    top_right = npHullTop4[3]
    bottom_left = npHullBottom4[0]
    bottom_right = npHullBottom4[3]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def draw_corners(c_img, points):
    (top_left, top_right, bottom_right, bottom_left) = points

    cv2.circle(c_img, tuple(top_left), 5, (10, 200, 20), -1)
    cv2.circle(c_img, tuple(top_right), 5, (10, 200, 20), -1)
    cv2.circle(c_img, tuple(bottom_right), 5, (10, 200, 20), -1)
    cv2.circle(c_img, tuple(bottom_left), 5, (10, 200, 20), -1)
    print("outmost points: ", top_left, top_right, bottom_right, bottom_left)


def perspective_transform(img, points):
    """Transform img so that points are the new corners"""

    source = np.array(
        points,
        dtype="float32")

    dest = np.array([
        [TRANSF_SIZE, TRANSF_SIZE],
        [0, TRANSF_SIZE],
        [0, 0],
        [TRANSF_SIZE, 0]],
        dtype="float32")

    img_dest = img.copy()
    transf = cv2.getPerspectiveTransform(source, dest)
    warped = cv2.warpPerspective(img, transf, (TRANSF_SIZE, TRANSF_SIZE))
    return warped


# image : grayscale image
# pts : 4 points arranged clockwisely
def four_point_transform(image, src):
    # obtain a consistent order of the points and unpack them
    # individually

    (tl, tr, br, bl) = src

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # print(rect, dst)
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


# def get_individual_omr_region(top_left_corners, ):
#     features_distance_for_box


def manual_sort_points(pts):
    tl, tr, br, bl = None, None, None, None

    if pts[0][0] < pts[1][0]:
        tl = pts[0]
        tr = pts[1]
    else:
        tl = pts[1]
        tr = pts[0]

    if pts[2][0] < pts[3][0]:
        bl = pts[2]
        br = pts[3]
    else:
        bl = pts[3]
        br = pts[2]

    # print("1st", pts[0])
    # print("2nd", pts[1])
    # print("3rd", pts[2])
    # print("4th", pts[3])
    # print(tl, tr, br, bl)

    return np.array([tl, tr, br, bl])


def get_omr_region(ori_img, debug=False, binaryThresh1=120):
    img = cv2.GaussianBlur(ori_img, (9, 9), 10)
    # img = cv2.bilateralFilter(ori_img, 9, 75, 75)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = normalize(img)
    # img = cv2.threshold(img, binaryThresh1, 255, cv2.THRESH_BINARY )[1]
    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges_img = cv2.Canny(img, 50, 200, apertureSize=3, L2gradient=True)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # edges_img = cv2.dilate(edges_img, kernel, iterations=1)
    # edges_img = cv2.erode(edges_img, kernel, iterations=1)

    # edges_img = cv2.morphologyEx(edges_img, cv2.MORPH_OPEN, kernel)
    contours = get_contours(edges_img)

    detected_contours = []
    for cnt in contours:
        cnt_feature = get_important_contour_featues(cnt, img)
        if 0.4 < cnt_feature["image_height_ratio"] < 1.0 and 0.6 < cnt_feature["image_width_ratio"] < 1.0:
            detected_contours.append(cnt)

    c_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if len(detected_contours) > 0:
        pixel_to_cut = 10
        hull_points = get_convex_hull_points(detected_contours)
        outmost_points = get_outmost_points_by_summation(hull_points)

        omr_region_img = four_point_transform(ori_img, outmost_points)
        omr_region_img = omr_region_img[pixel_to_cut:omr_region_img.shape[0] - pixel_to_cut, pixel_to_cut:omr_region_img.shape[1] - pixel_to_cut]
        if debug:
            draw_corners(c_img, outmost_points)
            cv2.drawContours(c_img, detected_contours, -1, (20, 10, 200), thickness=1)

        return omr_region_img, c_img, edges_img

    return None, c_img, edges_img


def is_a_circle(features):
    true_for = 0

    if .80 < features['hull_area_ratio'] <= 1:
        true_for += 1
    if .50 < features['bounding_area_ratio'] < 1:
        true_for += 1
    if 0.020 < features['image_height_ratio'] < 0.08:
        true_for += 1
    if 0.020 < features['image_width_ratio'] < 0.08:
        true_for += 1
    if .65 < features['aspect_ratio'] < 1.4:
        true_for += 1
    if 20 < features['height'] < 70:
        true_for += 1
    if 20 < features['width'] < 70:
        true_for += 1

    return true_for == 7


# return numpy 3d array of shape(n, 4, 2) or shape(0,) in case of invalid args
def arrange_points_according_to_question(point_group, is_debug=False, ori_img=None):
    if len(point_group) == 0:
        return np.array([])
    np_point_group = np.array(point_group)
    if np_point_group.shape[0] % 4 != 0:
        return np.array([])

    # sort points according to y coordinates
    np_point_group = np_point_group[np_point_group[:, 1].argsort()]

    # reshape the 2d array to 3d to arrange to blocks/circles according to question number
    shape1d = int(np_point_group.shape[0] / 4)
    np_point_group = np_point_group.reshape(shape1d, 4, 2)

    # now sort ponts in individual row according to the columns x coordinates
    i_length = np_point_group.shape[0]
    j_length = np_point_group.shape[1]
    for i in range(i_length):
        np_point_group[i] = np_point_group[i][np_point_group[i][:, 0].argsort()]
        if is_debug and ori_img is not None:
            for j in range(j_length):
                draw_text(ori_img, str(i + 1) + "/" + str(j + 1), np_point_group[i, j])

    return np_point_group


def get_ans_block(np_array_points, contours_dic, img, row, col, is_debug=False):
    try:
        (cx, cy) = np_array_points[row - 1][col - 1]
        if is_debug:
            print("cx, cy: ", cx, cy)
        block14_contour = contours_dic[str(cx) + str(cy)]
        if block14_contour is not None:
            (x, y, w, h) = cv2.boundingRect(block14_contour)
            return img[y:y + h, x:x + w]

        return None
    except Exception:
        return None


def operate_on_circle_block(block):
    try:
        img = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        black_img = 255 - img
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        black_img = cv2.dilate(black_img, kernel, iterations=5)
        return black_img

    except Exception:
        return None
