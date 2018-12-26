import cv2
import math
import numpy as np

MAX_WIDTH = 350
MAX_HEIGHT = 920


def normalize(im):
    im = cv2.normalize(im, np.zeros(im.shape), 0, 255, norm_type=cv2.NORM_MINMAX)
    return cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def order_points_by_summation(pts):
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


def sort_points(pts):
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


# image : grayscale image
# pts : 4 points arranged clockwisely
def four_point_transform(image, rect):
    # obtain a consistent order of the points and unpack them
    # individually

    (tl, tr, br, bl) = rect

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
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def get_approx_contour(contour, tol=.01):
    """Get rid of 'useless' points in the contour"""
    epsilon = tol * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def get_contours(image_gray):
    im2, contours, hierarchy = cv2.findContours(
        image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return list(map(get_approx_contour, contours))


def get_bounding_rect(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)


def get_convex_hull(contour):
    return cv2.convexHull(contour)


def get_centroid(contour):
    m = cv2.moments(contour)
    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])
    return x, y


def get_min_enclosing_circle(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    return center, radius


def find_rect_by_ratio(img, upper, lower, aspect_upper, aspect_lower):

    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # img = 255 - img

    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # img = cv2.erode(img, el, iterations=2)
    img = cv2.dilate(img, el, iterations=4)


    contours = get_contours(img)
    if contours is not None and len(contours) > 0:
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if lower < w < upper and lower < h < upper and aspect_lower < aspect_ratio < aspect_upper:
                center, radius = get_min_enclosing_circle(cnt)
                cv2.circle(img, center, radius, 200, -2)
                print(x, y, w, h, aspect_ratio)
                # cv2.drawContours(img, [cnt], 0, 200, thickness=-1)

    return img

