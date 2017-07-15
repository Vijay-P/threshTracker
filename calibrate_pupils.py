#!/usr/bin/env python3

# Copyright (c) 2017, Vijay Pillai
# vijay@vijaypillai.com

""" This module is for methods that perform feature analysis on image data. """

import cv2
import numpy as np

import isolation_methods

# constants for bottom threshold adjustments
H_ADJUST = 2
S_ADJUST = 30
V_ADJUST = 40  # was 10

# calibration iterations
MAXITER = 3


def set_cap_props(cap):
    """
    Sets VideoCapture object properties.

    Keyword arguments:
    cap -- cv2 VideoCapture object

    Return:
    None
    """
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return


def get_cap_props(cap):
    """
    Prints VideoCapture object properties.

    Keyword arguments:
    cap -- cv2 VideoCapture object

    Return:
    None
    """
    print("FPS:", cap.get(cv2.CAP_PROP_FPS))
    print("WIDTH:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("HEIGHT:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return


def make_histograms(r_isolated, l_isolated):
    """
    Generate histograms for H, S, and V of the isolated pupil images.

    Keyword arguments:
    r_isolated -- right pupil isolated
    l_isolated -- left pupil isolated

    Return:
    histogram for H (numpy ndarray)
    histogram for S (numpy ndarray)
    histogram for V (numpy ndarray)
    """
    histr_v = cv2.calcHist([r_isolated, l_isolated], [2], None, [256], [0, 256])
    histr_s = cv2.calcHist([r_isolated, l_isolated], [1], None, [256], [0, 256])
    histr_h = cv2.calcHist([r_isolated, l_isolated], [0], None, [256], [0, 256])
    return histr_h, histr_s, histr_v


def cohort_isolator(histr, mingap, cohortgap):
    """
    Finds a pair of top and bottom values within certain parameters.
    A "cohort" is a group of values with the same frequency.
    "top" is set to the highest value out of 255 that has the greatest frequency
        (global maximum of greatest value).
    "bottom" is set to the nearest value in a cohort that has a frequency
        difference that is greater than cohortgap.

    Keyword arguments:
    histr -- a histogram (from cv2.calcHist())
    mingap -- a minimum difference in top and bottom values
    cohortgap -- minimum distance in cohort frequency

    Return:
    top HSV value (tuple)
    bottom HSV value (tuple)
    """
    # initialize
    top = 0
    bottom = 0

    # get maxima for value threshold (sorted by frequency)
    maxima = get_local_maxima(histr)
    maxima = np.asarray(maxima)
    maxima_sorted = maxima[maxima[:, 1].argsort()][::-1]

    # bin of things with greatest frequency
    bin_1 = []
    for thing in maxima_sorted:
        if thing[1] == maxima_sorted[0][1]:
            bin_1.append(thing[0])
    top = max(bin_1)

    # bin of things with frequency of bin1-cohortgap, where cohortgap is distance in frequency
    bin_2 = []
    for thing in maxima_sorted:
        if thing[1] == maxima_sorted[0][1] - cohortgap:
            bin_2.append(thing[0])
    min_dist = 255

    # if |bin_2| = 1 is small, then set bottom to the only thing in bin_2
    if len(bin_2) == 1:
        bottom = bin_2[0]
    # if |bin_2| = 0, increase the frequency distance (cohortgap)
    elif len(bin_2) == 0:
        while len(bin_2) == 0:
            # if cohortgap is greater than the greatest frequency, then break. the bin
            # cannot be filled.
            if cohortgap >= maxima_sorted[0][1]:
                break
            cohortgap += 1
            # if you find anything of frequency of bin1-cohortgap, add it to bin2
            for thing in maxima_sorted:
                if thing[1] == maxima_sorted[0][1] - cohortgap:
                    bin_2.append(thing[0])
            # if there is anything in bin_2, set bottom to value with distance from
            # the top that is greater than mingap and closest to top
            for item in bin_2:
                if (abs(top - item) < min_dist) and abs(top - item) > mingap:
                    min_dist = abs(top - item)
                    bottom = item
    # if |bin_2| >= 1, then set bottom to value with distance
    # from top that is greater than mingap and closest to top
    else:
        for item in bin_2:
            if (abs(top - item) < min_dist) and abs(top - item) > mingap:
                min_dist = abs(top - item)
                bottom = item

    # if bottom is less than top, then swap them
    if bottom > top:
        temp = top
        top = bottom
        bottom = temp

    # we only use the top value, but we needed to keep track of bottom in case of swap
    return top


def get_local_maxima(histr):
    """
    Finds local maxima of a given histogram.

    arguents:
    histr -- a histogram

    Return:
    a list of maxima
    """
    maxima = []
    for value in range(1, np.size(histr)):
        if value <= 1:
            last = 0
        else:
            last = histr[value - 1]
        if value < np.size(histr) - 1:
            nex = histr[value + 1]
        else:
            nex = 0
        if (histr[value] > nex) and (histr[value] > last):
            maxima.append([int(value), int(histr[value][0])])
    if len(maxima) == 0:
        full = [histr[x][0] for x in range(1, np.size(histr))]
        top = max(full)
        maxima.append([full.index(top), top])
    return maxima


def get_thresh(top_s, top_v, hist, r_isolated, l_isolated):
    """
    Calculate the color threshold.

    Keyword arguments:
    top_s -- output of cohort isolator for S
    top_v -- output of cohort isolator for V
    hist -- Hue histogram
    r_isolated -- right isolated pupil sub-frame
    l_isolated -- left isolated pupil sub-frame

    Return:
    top (HSV value)
    bottom (HSV value)
    """
    color = []
    for row in range(np.size(r_isolated, 0)):
        for col in range(np.size(r_isolated, 1)):
            if not np.all(r_isolated[row, col] == 0):
                color.append(np.array(r_isolated[row, col]))
    for row in range(np.size(l_isolated, 0)):
        for col in range(np.size(l_isolated, 1)):
            if not np.all(l_isolated[row, col] == 0):
                color.append(np.array(l_isolated[row, col]))

    color = np.array(color)
    _, _, min_loc, max_loc = cv2.minMaxLoc(color)

    top = cv2.cvtColor(np.uint8([[color[max_loc[1]]]]), cv2.COLOR_BGR2HSV)
    bottom = cv2.cvtColor(np.uint8([[color[min_loc[1]]]]), cv2.COLOR_BGR2HSV)

    h_vals = [top[0, 0, 0], bottom[0, 0, 0]]

    h_top = max(h_vals)
    h_bottom = min(h_vals)

    h_top_list = get_local_maxima(hist)

    for value in h_top_list:
        if value[0] > h_top:
            h_top = value[0]

    if int(h_bottom) - int(h_top) <= 1:
        h_bottom = h_top - H_ADJUST

    s_bottom = (top_s - S_ADJUST) % 255
    if s_bottom > top_s:
        s_bottom = 0

    v_top = top_v
    if top_v < 245:
        v_top += 4

    v_bottom = (top_v - V_ADJUST) % 255
    if v_bottom > top_v:
        v_bottom = 0

    top = np.array([h_top, top_s, v_top], dtype=np.uint8)
    bottom = np.array([h_bottom, s_bottom, v_bottom], dtype=np.uint8)

    return top, bottom


def calibration(cap, frame, w_input, w_output):
    """
    Takes a frame and generates color thresholds.

    Keyword arguments:
    cap -- cv2 VideoCapture object
    frame -- a frame to generate thresholds from
    w_input -- face_worker() input queue
    w_output -- face_worker() output queue

    Return:
    top HSV color value (tuple)
    bottom HSV color value (tuple)
    filtered face sub-frame
    right isolated pupil sub-frame
    left isolated pupil sub-frame
    right eye sub-frame
    left eye sub-frame
    face sub-frame
    """
    dim = None
    frame2 = cv2.flip(frame, 1)
    w_input.put(frame2)
    dim = w_output.get()
    while dim is None:
        print("no dim, recap")
        ret, frame = cap.read()
        if ret:
            frame2 = cv2.flip(frame, 1)
            w_input.put(frame2)
            dim = w_output.get()
    roi_face = frame2[dim[0]:dim[1], dim[2]:dim[3]]
    left, right, _, _ = isolation_methods.find_eyes(
        roi_face, dim[3] - dim[2], dim[1] - dim[0])
    r_isolated, l_isolated = isolation_methods.get_pupils(right, left)
    l_isolated_hsv = cv2.cvtColor(l_isolated, cv2.COLOR_BGR2HSV)
    r_isolated_hsv = cv2.cvtColor(r_isolated, cv2.COLOR_BGR2HSV)
    hist, hist2, hist3 = make_histograms(r_isolated_hsv, l_isolated_hsv)
    top, bottom = get_thresh(cohort_isolator(hist2, 10, 2), cohort_isolator(
        hist3, 10, 1), hist, r_isolated, l_isolated)
    cv2.imshow("f", roi_face)
    cv2.imshow("half", isolation_methods.color_filter_main(top, bottom, roi_face))
    cv2.imshow("r", r_isolated)
    cv2.imshow("rw", right)
    cv2.imshow("l", l_isolated)
    cv2.imshow("lw", left)
    return top, bottom


def calibration_windows():
    """
    create and move windows for showing calibration windows
    """
    cv2.namedWindow("f")
    cv2.namedWindow("half")
    cv2.namedWindow("r")
    cv2.namedWindow("rw")
    cv2.namedWindow("l")
    cv2.namedWindow("lw")

    cv2.moveWindow("f", 740, 0)
    cv2.moveWindow("half", 740, 500)
    cv2.moveWindow("r", 1, 1)
    cv2.moveWindow("rw", 0, 300)
    cv2.moveWindow("l", 1480, 0)
    cv2.moveWindow("lw", 1480, 300)


def cal(capture_source, w_input, w_output):
    """
    Runs the calibration function until the user accepts calibration.

    Keyword arguments:
    cap -- cv2 VideoCapture object
    w_input -- the input queue for face_worker()
    w_output -- the output queue for face_worker()

    Return:
    top HSV
    bottom HSV
    """
    cap = cv2.VideoCapture(capture_source)
    set_cap_props(cap)
    get_cap_props(cap)

    avg_top = [0, 0, 0]
    avg_bottom = [0, 0, 0]
    counter = 0
    recap = False

    calibration_windows()

    _, frame = cap.read()
    top, bottom = calibration(cap, frame, w_input, w_output)

    while True:
        print("\nPress g to calibrate on frame. \nPress r to try another. \nFrames remaining to "
              "complete calibration: " + str(3 - counter) + ".\n")
        k = cv2.waitKey()
        if k == ord('g'):
            counter += 1
            avg_top = [sum(x) for x in zip(avg_top, top)]
            avg_bottom = [sum(x) for x in zip(avg_bottom, bottom)]
            recap = True
        if k == ord('r') or recap:
            print("Grabbing new frame.")
            _, frame = cap.read()
            top, bottom = calibration(cap, frame, w_input, w_output)
            recap = False
        if counter == MAXITER:
            break
    cap.release()
    cv2.destroyAllWindows()
    return np.asarray([int(x / 3) for x in avg_top]), np.asarray([int(x / 3) for x in avg_bottom])
