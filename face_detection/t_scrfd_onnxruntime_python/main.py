from __future__ import print_function
from scrfd import SCRFD
import os
import cv2
import numpy as np

save_folder = "./prediction/"
dataset_folder = "../widerface_val/images/"

if __name__ == '__main__':
    detector = SCRFD(model_file='model/scrfd_10g_bnkps.onnx')
    detector.prepare(-1)
    # face_recognizer = face_common.FaceRecognizer(
    #     True,
    #     "model/retinaface_resnet50_480x480.onnx",
    #     480, 0.02,
    #     False,
    #     ""
    # )
    testset_folder = dataset_folder
    testset_list = dataset_folder[:-7] + "wider_val.txt"
    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)
    for i, img_name in enumerate(test_dataset):
        ############################# Add face detection here#######################################
        image_path = testset_folder + img_name
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        bboxes, kpss = detector.detect(img, 0.02, input_size=(640, 640))
        # results = face_recognizer.Detect(img, False, False)
        dets = []
        for bbox in bboxes:
            x1, y1, x2, y2, score = bbox
            dets.append([x1, y1, x2, y2, score])
        dets = np.array(dets)
        ############################################################################################
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
