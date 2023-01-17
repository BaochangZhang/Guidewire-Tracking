import glob
import os
from Utilities.file_folder_operation import *
import time
from tqdm import tqdm
import numpy as np
from PIL import Image
import shutil
from Utilities.Data_reader import Data_2DImage_Reader
import cv2


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'Image' + os.sep, os.sep + 'BBox' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]

def img2mask_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'Image' + os.sep, os.sep + 'Label' + os.sep  # /images/, /labels/ substrings
    return ['png'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]


def creat_flow_img(data_dir=None, output_dir=None):

    train_out_path = path_str(join(output_dir, "Training_data"))
    test_out_path = path_str(join(output_dir, "Test_data"))

    train_in_path = path_str(join(data_dir, "Training_data"))
    test_in_path = path_str(join(data_dir, "Test_data"))

    train_studies = subfolders(train_in_path, prefix='sequence')
    test_studies = subfolders(test_in_path, prefix='sequence')

    for test_study in tqdm(test_studies):

        sequence_name = test_study.split('/')[-1]
        sequence_path = path_str(join(test_out_path, sequence_name))
        new_filepath = path_str(join(sequence_path, "Image"))
        new_gtpath = path_str(join(sequence_path, "Label"))
        new_boundingboxpath = path_str(join(sequence_path, "BBox"))

        maybe_makedirs(new_filepath)
        maybe_makedirs(new_gtpath)
        maybe_makedirs(new_boundingboxpath)

        img_file = sorted(glob.glob(join(test_study, "Image/*.png")))
        box_file = img2label_paths(img_file)
        mask_file = img2mask_paths(img_file)
        for fid in range(1, len(img_file)-1):

            lable_path = box_file[fid]
            mask_path = mask_file[fid]

            img1 = Data_2DImage_Reader(img_file[fid-1]).read_rgb_gray_image(gray=True)
            img2 = Data_2DImage_Reader(img_file[fid]).read_rgb_gray_image(gray=True)
            img3 = Data_2DImage_Reader(img_file[fid+1]).read_rgb_gray_image(gray=True)

            flow_img = cv2.merge([img1, img2, img3])

            flow_lable_path = path_str(join(new_boundingboxpath, box_file[fid].split('/')[-1]))
            flow_mask_path = path_str(join(new_gtpath, mask_file[fid].split('/')[-1]))
            flow_img_path = path_str(join(new_filepath, img_file[fid].split('/')[-1]))

            shutil.copyfile(mask_path, flow_mask_path)
            shutil.copyfile(lable_path, flow_lable_path)
            cv2.imwrite(flow_img_path, flow_img)

    for train_study in tqdm(train_studies):

        sequence_name = train_study.split('/')[-1]
        sequence_path = path_str(join(train_out_path, sequence_name))
        new_filepath = path_str(join(sequence_path, "Image"))
        new_gtpath = path_str(join(sequence_path, "Label"))
        new_boundingboxpath = path_str(join(sequence_path, "BBox"))

        maybe_makedirs(new_filepath)
        maybe_makedirs(new_gtpath)
        maybe_makedirs(new_boundingboxpath)

        img_file = sorted(glob.glob(join(train_study, "Image/*.png")))
        box_file = img2label_paths(img_file)
        mask_file = img2mask_paths(img_file)
        for fid in range(1, len(img_file)-1):

            lable_path = box_file[fid]
            mask_path = mask_file[fid]

            img1 = Data_2DImage_Reader(img_file[fid-1]).read_rgb_gray_image(gray=True)
            img2 = Data_2DImage_Reader(img_file[fid]).read_rgb_gray_image(gray=True)
            img3 = Data_2DImage_Reader(img_file[fid+1]).read_rgb_gray_image(gray=True)

            flow_img = cv2.merge([img1, img2, img3])

            flow_lable_path = path_str(join(new_boundingboxpath, box_file[fid].split('/')[-1]))
            flow_mask_path = path_str(join(new_gtpath, mask_file[fid].split('/')[-1]))
            flow_img_path = path_str(join(new_filepath, img_file[fid].split('/')[-1]))

            shutil.copyfile(mask_path, flow_mask_path)
            shutil.copyfile(lable_path, flow_lable_path)
            cv2.imwrite(flow_img_path, flow_img)


if __name__ == '__main__':
    # in_path = '/home/baochang/Datasets/T7_Guidewire_Project/TrTsDataset/raw_detection'
    # out_path = '/home/baochang/Datasets/T7_Guidewire_Project/TrTsDataset/raw_detection_flow'
    # creat_flow_img(in_path, out_path)

    in_path = '/home/baochang/Datasets/T7_Guidewire_Project/TrTsDataset/sythetic_detection'
    out_path = '/home/baochang/Datasets/T7_Guidewire_Project/TrTsDataset/sythetic_detection_flow'
    creat_flow_img(in_path, out_path)
