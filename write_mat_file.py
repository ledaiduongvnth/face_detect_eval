from __future__ import print_function
import numpy as np
import os
import cv2
import torch
from scipy.io import savemat
from scipy.io import loadmat
from t_centerface.centerface import CenterFace

# Folder for saving face detection results
save_folder = "./z/"
# Folder for containing input images
dataset_folder = "./widerface_val_TEST/images/"


def main():

    # gt_mat = loadmat('1.mat')

    # print(gt_mat['file_list'])

    gtmat = {'face_bbx_list':[], 'event_list':[], 'file_list':[]}
    # gtmat['file_list'].append(['9_Press_Conference_Press_Conference_9_328','9_Press_Conference_Press_Conference_9_328'])
    # gtmat['file_list'].append([np.array([[np.array('9_Press_Conference_Press_Conference_9_328'), np.array('9_Press_Conference_Press_Conference_9_328')]], dtype=object)])
    # gtmat['file_list'].append([np.array([np.array(['10_Press_Conference_Press_Conference_9_328']), np.array(['10_Press_Conference_Press_Conference_9_328'])])])
    # gtmat['file_list'].append([np.array([np.array(['9_Press_Conference_Press_Conference_9_328'])])])
    # gtmat['file_list'].append([np.array([np.array(['10_Press_Conference_Press_Conference_9_328'])])])
    # gtmat['file_list'].append([np.array([['10_Press_Conference_Press_Conference_9_345', '10_Press_Conference_Press_Conference_9_346']])])

    # print(gtmat['file_list'])
    # print("-------------------------")
    # print(np.insert(gtmat['file_list'][0][0][0], 0 , [np.array('9_Press_Conference_Press_Conference_9_328')]))
    # gtmat['file_list'][0][0][0] = gtmat['file_list'][0][0][0][..., [np.array('9_Press_Conference_Press_Conference_9_328')]]
    # gtmat['file_list'][0][0][0] = np.insert(gtmat['file_list'][0][0][0], 0 , [np.array('9_Press_Conference_Press_Conference_9_328')])
    # print("-------------------------")

    # gtmat['file_list'][0][0][0].shape[0] = gtmat['file_list'][0][0][0].shape[0] + 1
    # np.expand_dims(gtmat['file_list'][0][0][0], axis=1)
    # print(gtmat['file_list'][0][0][0].shape)
    # gtmat['file_list'][0][0][0] = np.append(gtmat['file_list'][0][0][0], gtmat['file_list'][0][0][0][0])
    # print(gtmat['file_list'])
    # savemat("1.mat", gtmat)

    # torch.set_grad_enabled(False)
    testset_folder = dataset_folder
    # testset_list = dataset_folder[:-7] + "wider_val.txt"
    testset_list = "./widerface_val_TEST/1.txt"

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)
    centerface = CenterFace(landmarks=True)
    folder_index = -1
    file_index = 0

    for i, img_name in enumerate(test_dataset):
        check_created_folder = False
        # print(img_name.split("/")[2])
        ############################ Add face detection here#######################################
        image_path = testset_folder + img_name
        # print("image_path: ", image_path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        dets, lms = centerface(img, h, w, threshold=0.35)
        ############################################################################################
        save_name = save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
            # gtmat['event_list'].append([np.array([dirname.replace(save_folder + "/", "")])])
            check_created_folder = True
            gtmat['file_list'].append([])
            gtmat['face_bbx_list'].append([])
            folder_index = folder_index + 1
            gtmat['event_list'].append(dirname.split("/")[-1])
            file_index = 0

        with open(save_name, "w") as fd:
            bboxs = dets
            # file_name = os.path.basename(save_name)[:-4] + "\n"
            file_name = os.path.basename(img_name) + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            gtmat['face_bbx_list'][folder_index].append([])
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)
                # print("-------gtmat['face_bbx_list']--------: ", gtmat['face_bbx_list'])
                gtmat['face_bbx_list'][folder_index][file_index].append([x, y, w, h])
            file_index  = file_index + 1
            gtmat['file_list'][folder_index].append(image_path.split("/")[-1])

        # if check_created_folder == True:
        #     folder_index = folder_index + 1
    savemat("1.mat", gtmat)

    # print(gtmat['event_list'])


main()