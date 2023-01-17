# -*- coding:utf-8 -*-
"""
@Time: 2022/12/4 13:51
@Author: Baochang Zhang
@IDE: PyCharm
@File: converimage.py
@Comment: #Enter some comments at here
"""
import cv2
import os


def save_all_frames(video_path, dir_path, basename, ext='png'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    i = 0
    # a variable to set how many frames you want to skip
    frame_skip = 50

    while True:
        ret, frame = cap.read()
        if ret:
            if i % frame_skip == 0:
                frame = frame[5:5 + 1070, 400:400 + 1070]
                # frame = frame[150:150+800, 550:550 + 800]
                frame = cv2.resize(frame, dsize=(512, 512))
                cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
                n += 1
                i = 0
            i += 1
        else:
            return


save_all_frames('../brainlab/BewegungDraht_C1_Left.mp4', './BewegungDraht_C1_Left/', 'BewegungDraht_C1_Left')