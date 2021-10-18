import os
import torch
import cv2
from torch import linalg as LA
from torch import nn
import numpy as np
import face_common
from wider_face import WiderFaceDetection


def crop_face(src_img, x1, y1, x2, y2, expanded_face_scale, target_lmk):
    new_width = ((x2 - x1) / 2 * expanded_face_scale)
    new_height = ((y2 - y1) / 2 * expanded_face_scale)
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    new_top = y_center - new_height > 0 and (y_center - new_height) or 0
    new_bottom = y_center + new_height < src_img.shape[0] and (y_center + new_height) or src_img.shape[0]
    new_left = x_center - new_width > 0 and (x_center - new_width) or 0
    new_right = x_center + new_width < src_img.shape[1] and (x_center + new_width) or src_img.shape[1]
    face_image = src_img[int(new_top):int(new_bottom), int(new_left):int(new_right)]
    for landmark in target_lmk:
        landmark[0] = landmark[0] - new_left
        landmark[1] = landmark[1] - new_top
    return face_image, new_left, new_top, target_lmk

dataset = WiderFaceDetection("/mnt/hdd/PycharmProjects/Pytorch_Retinaface/data/widerface/train/label.txt")


def NMELoss(predicted_landmark, target_landmark):
    # landmark is a numpy array which has shape [5, 2]
    num_face_landmark = 5
    leye_nouse_vec = torch.from_numpy(target_landmark[0] - target_landmark[2])
    reye_nouse_vec = torch.from_numpy(target_landmark[1] - target_landmark[2])
    inter_occular_distance = LA.norm(leye_nouse_vec) + LA.norm(reye_nouse_vec)
    loss = nn.MSELoss(reduction="sum")
    preloss = loss(torch.from_numpy(predicted_landmark), torch.from_numpy(target_landmark))
    nme_loss = torch.sqrt(preloss) / (inter_occular_distance * num_face_landmark)
    return nme_loss


face_recognizer = face_common.FaceRecognizer(
    True,
    "model/retinaface_resnet50_480x480.onnx",
    480, 0.01, 0.4,
    False,
    "",
    0
)

result = 0
number_faces = 0

for i in range(len(dataset)):
    img, target = dataset[i]
    for face in target:
        face = face.astype(np.int32)
        area = np.abs(face[2] - face[0]) * np.abs(face[3] - face[1])
        if face[-1] != -1 and area > 12544:
            number_faces = number_faces + 1
            target_lmk = face[4:14].astype(np.float32)
            target_lmk = target_lmk.reshape((5, 2))
            img_face, new_left, new_top, target_lmk = crop_face(img.copy(), face[0], face[1], face[2], face[3], 1.3, target_lmk)

            if i % 500 == 0:
                print('processing', i)
            detection = face_recognizer.Detect(img_face, False, False)
            landmarks = detection[0].landmarks
            predict_lanmark = []
            for v in landmarks:
                predict_lanmark.append([v.x, v.y])
            predict_lanmark = np.array(predict_lanmark)

            for p in target_lmk:
                p = p.astype(np.int)
                cv2.circle(img_face, tuple(p), 1, (0, 0, 255), 2)

            for p in predict_lanmark:
                p = p.astype(np.int)
                cv2.circle(img_face, tuple(p), 1, (0, 255, 0), 2)

            nme_loss = NMELoss(predict_lanmark, target_lmk).item()
            result = result + nme_loss
            print(nme_loss)
            cv2.imshow("le", img_face)
            cv2.waitKey(2000)

result = result/len(number_faces)
print("This is NMELoss:", result)
