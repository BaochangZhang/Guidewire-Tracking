import os
import numpy as np
import xml.etree.cElementTree as et
from Anchors_keams import kmeans, avg_iou
from Utilities.file_folder_operation import *
import cv2
import matplotlib.pyplot as plt

FILE_ROOT = "/home/baochang/Projects/Guidewire_Tracking_Segmentation/Detection_phase/data/train_guidewire.txt"
ANCHORS_TXT_PATH = "/home/baochang/Projects/Guidewire_Tracking_Segmentation/Detection_phase/data/anchors.txt"

CLUSTERS = 9
CLASS_NAMES = ['Guidewire']


def load_data(file_list):

    train_label_files = []
    f = open(file_list, "r")
    height, width, _ = cv2.imread(f.readline()[:-1]).shape

    for x in f:
        filename = x.split('/')[-1].split('.')[0]
        file_root = os.path.abspath(os.path.join(x, "../.."))
        label_url = join(file_root, 'BBox/'+filename+'.txt')
        train_label_files.append(label_url)
    boxes = []
    for annotation_file in train_label_files:
        anno = np.reshape(np.loadtxt(annotation_file), [-1, 5])
        for i in range(anno.shape[0]):
            # w = anno[i, 3]
            # h = anno[i, 4]
            box = anno[i, :]
            boxes.append(box)
    return np.array(boxes), height, width


def plot_labels(labels):
    # plot dataset labels
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classees, boxes

    def hist2d(x, y, n=100):
        xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
        hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
        xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
        yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
        return np.log(hist[xidx, yidx])

    fig, ax = plt.subplots(1, 3, figsize=(8*3, 8), tight_layout=True)
    ax = ax.ravel()
    ax[0].hist(c, bins=int(c.max() + 1))
    ax[0].set_xlabel('classes')
    ax[1].scatter(b[0], b[1], c=hist2d(b[0], b[1], 90), cmap='jet')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[2].scatter(b[2], b[3], c=hist2d(b[2], b[3], 90), cmap='jet')
    ax[2].set_xlabel('width')
    ax[2].set_ylabel('height')
    plt.savefig('labels.png', dpi=200)


if __name__ == '__main__':

    anchors_txt = open(ANCHORS_TXT_PATH, "w")
    train_boxes, height, width = load_data(FILE_ROOT)
    # plot_labels(train_boxes)
    train_boxes = train_boxes[:, 3:5]
    count = 1
    best_accuracy = 0
    best_anchors = []
    best_ratios = []

    for i in range(10):      ##### 可以修改，不要太大，否则时间很长
        anchors_tmp = []
        clusters = kmeans(train_boxes, k=CLUSTERS)
        idx = clusters[:, 0].argsort()
        clusters = clusters[idx]
        # print(clusters)

        for j in range(CLUSTERS):
            anchor = [round(clusters[j][0] * width, 2), round(clusters[j][1] * height, 2)]
            anchors_tmp.append(anchor)
            print(f"Anchors:{anchor}")

        temp_accuracy = avg_iou(train_boxes, clusters) * 100
        print("Train_Accuracy:{:.2f}%".format(temp_accuracy))

        ratios = np.around(clusters[:, 0] / clusters[:, 1], decimals=2).tolist()
        ratios.sort()
        print("Ratios:{}".format(ratios))
        print(20 * "*" + " {} ".format(count) + 20 * "*")

        count += 1

        if temp_accuracy > best_accuracy:
            best_accuracy = temp_accuracy
            best_anchors = anchors_tmp
            best_ratios = ratios

    anchors_txt.write("Best Accuracy = " + str(round(best_accuracy, 2)) + '%' + "\r\n")
    anchors_txt.write("Best Anchors = " + str(best_anchors) + "\r\n")
    anchors_txt.write("Best Ratios = " + str(best_ratios))
    anchors_txt.close()