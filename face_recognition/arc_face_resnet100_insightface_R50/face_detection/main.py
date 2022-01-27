from __future__ import print_function
# from scrfd import SCRFD
# import os
# import cv2
# import numpy as np

import argparse
import sys
import os
import time
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import cv2
from rcnn.logger import logger
#from rcnn.config import config, default, generate_config
#from rcnn.tools.test_rcnn import test_rcnn
#from rcnn.tools.test_rpn import test_rpn
from rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes, landmark_pred
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper
from rcnn.processing.bbox_transform import bbox_overlaps
from rcnn.dataset import retinaface
from retinaface import RetinaFace


detector = RetinaFace("/home/d/Downloads/Telegram Desktop/face_and_utvm_models/face_models/retina/mnet.25",
                      0,
                      0,
                      network="net3",
                      nocrop=False,
                      vote=False)



save_folder = "/mnt/hdd/PycharmProjects/Retinaface/prediction/"
dataset_folder = "/home/d/Downloads/Telegram Desktop/labeled/"

if __name__ == '__main__':
    testset_folder = dataset_folder
    testset_list = "/home/d/Downloads/Telegram Desktop/test2021.txt"
    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)
    for i, img_name in enumerate(test_dataset):
        ############################# Add face detection here#######################################
        image_path = testset_folder + img_name
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        ##########################################
        do_flip = True
        #TEST_SCALES = [500, 800, 1200, 1600]
        TEST_SCALES = [640]
        target_size = 800
        max_size = 1200
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        scales = [
            float(scale) / target_size * im_scale for scale in TEST_SCALES
        ]
        boxes, landmarks = detector.detect(img,
                                   threshold=0.02,
                                   scales=scales,
                                   do_flip=do_flip)
        save_name = img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        with open(save_folder+save_name, "w") as fd:
            bboxs = boxes
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence+ " \n"
                fd.write(line)
