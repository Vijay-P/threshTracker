#!/usr/bin/env python3

# Copyright (c) 2017, Vijay Pillai
# vijay@vijaypillai.com

""" This module is for methods that extract features from images. """

import cv2
import numpy as np

# Constants for eye dimensions
EYE_PERCENT_TOP = 25
EYE_PERCENT_SIDE = 13
EYE_PERCENT_HEIGHT = 25
EYE_PERCENT_WIDTH = 30
CAL_RADIUS = 4


def find_eyes(roi_color, width, height):
    """
    Eye sub-frame getter based on facial ratios.

    Keyword arguments:
    roi_color -- frame containing just face
    width -- width of face
    height -- height of face

    Return:
    frame containing just right eye
    frame containing just left eye
    [y, y+h, x, x+w] dimension list of right eye
    [y, y+h, x, x+w] dimension list of left eye
    """

    eye_roi_width = int(width * (EYE_PERCENT_WIDTH / 100))
    eye_roi_height = int(width * (EYE_PERCENT_HEIGHT / 100))
    eye_roi_top = int(height * (EYE_PERCENT_TOP / 100))

    left_x = int(width * (EYE_PERCENT_SIDE / 100))
    left_y = int(eye_roi_top)

    right_x = int(width - eye_roi_width - (width * (EYE_PERCENT_SIDE / 100)))
    right_y = int(eye_roi_top)

    roi_left_eye = roi_color[left_y:left_y + eye_roi_height, left_x:left_x + eye_roi_width]
    roi_right_eye = roi_color[right_y:right_y + eye_roi_height, right_x:right_x + eye_roi_width]
    right_dim = [right_y, right_y + eye_roi_height, right_x, right_x + eye_roi_width]
    left_dim = [left_y, left_y + eye_roi_height, left_x, left_x + eye_roi_width]

    return roi_right_eye, roi_left_eye, right_dim, left_dim


def face_detect(frame, classifier):
    """
    Face detection by haarcascades.

    Keyword arguments:
    frame -- a frame that may or may not contain a face
    classifier -- a trained haar-like feature classifier

    Return:
    [y, y+h, x, x+w] list of dimensions for face, if a face is present in frame
    None, if no face present in frame
    """
    faces = classifier.detectMultiScale(frame, 1.1, 2)
    largest = 0
    biggest_face = 0
    if np.size(faces, 0) > 0:
        if np.size(faces, 0) > 1:
            for face in range(np.size(faces, 0)):
                x_pos, y_pos, width, height = faces[face]
                if width * height > largest:
                    largest = width * height
                    biggest_face = face
        x_pos, y_pos, width, height = faces[biggest_face]
        return [y_pos, y_pos + height, x_pos, x_pos + width]
    return


def get_pupils(right, left):
    """
    Isolates pupils on black background for threshold calibration.

    Keyword arguments:
    right -- right eye sub-frame
    left -- left eye sub-frame

    Return:
    right eye isolated
    left eye isolated
    """
    # convert eyes to gray
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    # identify brightest and darkest pixels
    (_, _, _, max_loc) = cv2.minMaxLoc(right_gray)
    # make black image of same dimensions
    right_mask = np.zeros(shape=(right_gray.shape), dtype=(right_gray.dtype))
    # draw a white circle of radius 6 at brightest pixel
    cv2.circle(right_mask, max_loc, CAL_RADIUS, (255, 0, 0), -1)
    # use this circle image as a mask and do bitwise AND
    r_isolated = cv2.bitwise_and(right, right, mask=right_mask)

    # same process for left eye
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    (_, _, _, max_loc) = cv2.minMaxLoc(left_gray)
    left_mask = np.zeros(shape=(left_gray.shape), dtype=(left_gray.dtype))
    cv2.circle(left_mask, max_loc, 6, (255, 0, 0), -1)
    l_isolated = cv2.bitwise_and(left, left, mask=left_mask)

    return r_isolated, l_isolated


def color_filter_main(top, bottom, roi_face):
    """
    Blurs image to filter out noise, applies color threshold to image, converts image to grayscale.

    Keyword arguments:
    top -- (HSV color value)
    bottom -- (HSV color value)
    roi_face -- face sub-frame

    Return:
    8-bit, grayscale, thresholded image of face
    """
    # blur to filter out noise
    roi_face2 = cv2.medianBlur(roi_face, 5)
    roi_face2 = cv2.bilateralFilter(roi_face2, 5, 75, 75)
    # convert to hsv
    hsv = cv2.cvtColor(roi_face2, cv2.COLOR_BGR2HSV)
    # create a mask for all pixels in range on the hsv image
    mask = cv2.inRange(hsv, bottom, top)
    # apply mask to bgr image
    res = cv2.bitwise_and(roi_face, roi_face, mask=mask)
    # convert image to gray
    isolation = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # blur to remove artifacts
    blur = cv2.medianBlur(isolation, 1)
    blur = cv2.GaussianBlur(isolation, (3, 3), 0)
    # convert to 8-bit
    img8 = blur.astype('uint8')
    return img8


def get_points(img):
    """
    Gets non-black pixels in an image.

    Keyword arguments:
    img -- input image (numpy ndarray)

    Return:
    list of pixels
    """
    points = []

    for row in range(np.size(img, 0)):
        for col in range(np.size(img, 1)):
            if not np.all(img[row, col] == 0):
                points.append((row, col))
    return points
