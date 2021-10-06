import os
import torch
import cv2
from torch import linalg as LA
from torch import nn
import numpy as np
import face_common



def NMELoss(predicted_landmark, target_landmark):
    # landmark is a numpy array which has shape [5, 2]
    num_face_landmark = 5
    leye_reye_vec = torch.from_numpy(target_landmark[0] - target_landmark[1])
    inter_occular_distance = LA.norm(leye_reye_vec)
    loss = nn.MSELoss(reduction="sum")
    preloss = loss(torch.from_numpy(predicted_landmark), torch.from_numpy(target_landmark))
    nme_loss = torch.sqrt(preloss) / (inter_occular_distance * num_face_landmark)
    return nme_loss


data_path = "/mnt/hdd/IJB"
target = 'IJBC'
img_path = os.path.join(data_path, './%s/loose_crop' % target)
img_list_path = os.path.join(data_path, './%s/meta/%s_name_5pts_score.txt' % (target, target.lower()))

img_list = open(img_list_path)
files = img_list.readlines()
print('files:', len(files))

face_recognizer = face_common.FaceRecognizer(
    True,
    "model/retinaface_resnet50_480x480.onnx",
    480, 0.02, 0.4,
    False,
    "",
    0
)

result = 0
for img_index, each_line in enumerate(files):
    if img_index % 500 == 0:
        print('processing', img_index)
    name_lmk_score = each_line.strip().split(' ')
    img_name = os.path.join(img_path, name_lmk_score[0])
    img = cv2.imread(img_name)
    target_lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
    target_lmk = target_lmk.reshape((5, 2))
    detection = face_recognizer.Detect(img, False, False)
    landmarks = detection[0].landmarks
    predict_lanmark = []
    for v in landmarks:
        predict_lanmark.append([v.x, v.y])
    predict_lanmark = np.array(predict_lanmark)

    # for p in predict_lanmark:
    #     p = p.astype(np.int)
    #     cv2.circle(img, tuple(p), 3, (0, 0, 255), 2)

    nme_loss = NMELoss(predict_lanmark, target_lmk).item()
    result = result + nme_loss
    # print(nme_loss)
    # cv2.imshow("le", img)
    # cv2.waitKey(2000)

result = result/len(files)
print("This is NMELoss:", result)
