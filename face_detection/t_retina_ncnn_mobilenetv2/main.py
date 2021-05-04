from __future__ import print_function
import os
import cv2
import torch
# from centerface import CenterFace

save_folder = "./prediction"
dataset_folder = "../widerface_val/images"


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    testset_folder = dataset_folder
    testset_list = dataset_folder[:-7] + "/wider_val.txt"

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)
    # centerface = CenterFace(landmarks=True)

    for i, img_name in enumerate(test_dataset):
        ############################# Add face detection here#######################################
        image_path = testset_folder + img_name
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        # dets, lms = centerface(img, h, w, threshold=0.35)
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
