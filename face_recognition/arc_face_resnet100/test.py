import numpy as np
import face_common as face_common
import cv2
img = cv2.imread("/mnt/hdd/PycharmProjects/Pytorch_Retinaface/curve/test.jpg")
face_recognizer = face_common.FaceRecognizer(
    True,
    "/mnt/hdd/PycharmProjects/face_eval/face_detection/t_retina_onnxruntime_pytorch_resnet50/model/fd_resnet50_1600.onnx",
    True,
    "/mnt/hdd/CLionProjects/frvt1N/1N/config/model.onnx"
)
# a = face_recognizer.Detect(img, True)
# print(a)
#
# b = face_recognizer.Recognize(img)
# print(b)
from datetime import datetime
while True:
    t0 = datetime.now()
    c = face_recognizer.Process(np.ones((1000, 1000, 3)))
    print((datetime.now() - t0).microseconds / 1000)
    print(c)