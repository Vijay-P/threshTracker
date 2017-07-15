#!/usr/bin/env python3

# Copyright (c) 2017, Vijay Pillai
# vijay@vijaypillai.com

"""An eye tracking program."""

from multiprocessing import Process, Queue

import cv2
import numpy as np

import isolation_methods
import calibrate_pupils
import smallestenclosingcircle

CAPTURE_SOURCE = 1


def circle_eyes(pos, img8_r, img8_l, roir, roil):
    '''
    Draws circles around eyes and Return coordinates.

    Keyword arguments:
    pos -- list of x,y coordinates of face ROI
    img8_r -- 8-bit right eye (from isolation_methods.color_filter_main())
    img8_l -- 8-bit left eye (from isolation_methods.color_filter_main())
    frame_o -- original frame
    roir -- coordinates of right eye ROI [y, y + h, x, x + w]
    roil -- coordinates of left eye ROI [y, y + h, x, x + w]

    Return:
    frame with circled eyes (numpy ndarray)
    right eye coordinates (tuple)
    left eye coordinates (tuple)
    '''
    right_c = smallestenclosingcircle.make_circle(isolation_methods.get_points(img8_r))
    left_c = smallestenclosingcircle.make_circle(isolation_methods.get_points(img8_l))
    translated_right = None
    translated_left = None
    avg_radius = 0
    if right_c != None:
        translated_right = (int(pos[0] + round(right_c[1] + roir[2])),
                            int(roir[0] + pos[1] + round(right_c[0])))
        avg_radius = right_c[2]
    if left_c != None:
        translated_left = (int(pos[0] + round(left_c[1] + roil[2])),
                           int(roil[0] + pos[1] + round(left_c[0])))
        avg_radius = left_c[2]
    if (left_c != None) and (right_c != None):
        avg_radius = (right_c[2] + left_c[2]) / 2
    return translated_right, translated_left, avg_radius


def face_worker(w_input, w_output):
    """
    Worker process for finding face.

    Keyword arguments:
    w_input -- input queue (frames only, None to terminate)
    w_output -- output queue (ROI dimensions only [y, y + h, x, x + w])

    Return:
    N/A
    """
    try:
        face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        while True:
            frame = w_input.get()
            if isinstance(frame, type(None)):
                break
            dim = isolation_methods.face_detect(frame, face_classifier)
            w_output.put(dim)
    except KeyboardInterrupt:
        pass


def detect_pupils(dim, frame, threshold):
    '''
    Function for getting coordinates of pupils.

    Keyword arguments:
    dim - dimensions of face
    frame - image
    threshold - list of threshold values (top, bottom)

    Return:
    right pupil center
    left pupil center
    average pupil radius
    '''
    frame3 = frame[dim[0]:dim[1], dim[2]:dim[3]]
    left, right, roi_l, roi_r = isolation_methods.find_eyes(
        frame3, dim[3] - dim[2], dim[1] - dim[0])
    halfbit_l = isolation_methods.color_filter_main(threshold[0], threshold[1], left)
    halfbit_r = isolation_methods.color_filter_main(threshold[0], threshold[1], right)
    r_pos, l_pos, avg_radius = circle_eyes([dim[2], dim[0]], halfbit_r, halfbit_l, roi_r, roi_l)
    if avg_radius != 0:
        return [r_pos, l_pos, avg_radius]
    else:
        return [None, None, None]


def track_pupils():
    """
    Function for tracking pupils.

    Keyword arguments:
    input_queue -- input queue (terminate upon input)
    output_queue -- output queue (right pupil (x, y), left pupil (x, y),
        reference point (x, y), average eye radius (float))

    Return:
    N/A
    """
    try:
        face_input = Queue()
        face_output = Queue()

        face = Process(target=face_worker, args=(
            face_input, face_output))
        face.start()

        print("Beginning Calibration.")
        top, bottom = calibrate_pupils.cal(CAPTURE_SOURCE, face_input, face_output)
        print("Calibration complete!")
        print("Tracking eyes.")
        cap = cv2.VideoCapture(CAPTURE_SOURCE)
        calibrate_pupils.set_cap_props(cap)
        calibrate_pupils.get_cap_props(cap)

        counter = 1
        dim = None

        circle_r = None
        circle_l = None
        eye_r = None

        # fourcc = cv2.VideoWriter_fourcc(*'H264')
        # out = cv2.VideoWriter('output.mpeg', fourcc, 30.0, (1280, 720))

        while True:
            ret, frame = cap.read()

            if ret:
                frame = cv2.flip(frame, 1)
                if counter == 7:
                    counter = 1
                else:
                    counter += 1

                if counter == 1:
                    frame_2 = np.copy(frame)
                    face_input.put(frame_2)

                if not face_output.empty():
                    dim = face_output.get()

                if not isinstance(dim, type(None)):
                    ret_vals = detect_pupils(dim, frame, [top, bottom])
                    if ret_vals[0] != None:
                        circle_r = ret_vals[0]
                    if ret_vals[1] != None:
                        circle_l = ret_vals[1]
                    if ret_vals[2] != None:
                        eye_r = ret_vals[2]

                if not isinstance(eye_r, type(None)):
                    cv2.circle(frame, circle_r, int(eye_r), (0, 255, 0), 2)
                    cv2.circle(frame, circle_l, int(eye_r), (0, 255, 0), 2)

                # out.write(frame)
                cv2.imshow("frame", frame)
                cv2.waitKey(1)
            else:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        face_input.put(None)
        face.join()
        cap.release()

if __name__ == '__main__':
    track_pupils()
