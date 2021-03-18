import os
import numpy as np
import cv2


def save_prediction(dets, img_name, save_folder):
    save_name = save_folder + img_name[:-4] + ".txt"
    dirname = os.path.dirname(save_name)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    with open(save_name, "w") as fd:
        bboxs = dets
        file_name = os.path.basename(save_name)[:-4] + "\n"
        bboxs_num = str(len(bboxs)) + "\n"
        fd.write(file_name)
        fd.write(bboxs_num)
        for box in bboxs:
            x = int(box[0])
            y = int(box[1])
            w = int(box[2]) - int(box[0])
            h = int(box[3]) - int(box[1])
            confidence = str(box[4])
            line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
            fd.write(line)
