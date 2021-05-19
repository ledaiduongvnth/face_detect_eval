import argparse
import os.path

import cv2
import numpy as np
import sys
import mxnet as mx
import datetime
from skimage import transform as trans
import sklearn
from sklearn import preprocessing
import face_common as face_common
import time

class Embedding:
  def __init__(self, prefix, epoch, ctx_id=0):
    # print('loading',prefix, epoch)
    # ctx = mx.gpu(ctx_id)
    # sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    # all_layers = sym.get_internals()
    # sym = all_layers['fc1_output']
    image_size = (112,112)
    self.image_size = image_size
    # model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    # model.bind(for_training=False, data_shapes=[('data', (2, 3, image_size[0], image_size[1]))])
    # model.set_params(arg_params, aux_params)
    # self.model = model
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    src[:,0] += 8.0
    self.src = src
    self.face_recognizer = face_common.FaceRecognizer(
      True,
      "/mnt/hdd/PycharmProjects/face_eval/face_detection/t_retina_onnxruntime_pytorch_resnet50/model/fd_resnet50_1600.onnx",
      True,
      "/mnt/hdd/CLionProjects/frvt1N/1N/config/model.onnx"
    )

  def get(self, rimg, landmark):
    assert landmark.shape[0]==68 or landmark.shape[0]==5
    assert landmark.shape[1]==2
    if landmark.shape[0]==68:
      landmark5 = np.zeros( (5,2), dtype=np.float32 )
      landmark5[0] = (landmark[36]+landmark[39])/2
      landmark5[1] = (landmark[42]+landmark[45])/2
      landmark5[2] = landmark[30]
      landmark5[3] = landmark[48]
      landmark5[4] = landmark[54]
    else:
      landmark5 = landmark

    boxes = self.face_recognizer.Detect(rimg, False, True)
    if len(boxes) == 1:
      landmark5 = np.array([[int(boxes[0].landmarks[0].x), int(boxes[0].landmarks[0].y)],
                               [int(boxes[0].landmarks[1].x), int(boxes[0].landmarks[1].y)],
                               [int(boxes[0].landmarks[2].x), int(boxes[0].landmarks[2].y)],
                               [int(boxes[0].landmarks[3].x), int(boxes[0].landmarks[3].y)],
                               [int(boxes[0].landmarks[4].x), int(boxes[0].landmarks[4].y)]
                               ])
    else:
      print("Cannot detect faces")
      save_path = os.path.join("/mnt/hdd/cannot_detect_face", str(int(time.time())) + ".png")
      cv2.imwrite(save_path, rimg)

    # cv2.circle(rimg, (int(landmark_vts[0][0]), int(landmark_vts[0][1])), 1, (255, 0, 0), 4)
    # cv2.circle(rimg, (int(landmark5[0][0]), int(landmark5[0][1])), 1, (0, 0, 255), 4)

    tform = trans.SimilarityTransform()
    tform.estimate(landmark5, self.src)
    M = tform.params[0:2,:]
    img = cv2.warpAffine(rimg,M,(self.image_size[1],self.image_size[0]), borderValue = 0.0)

    # cv2.imshow("aligned img", img)
    # cv2.waitKey(2000)
    # cv2.destroyAllWindows()

    img_flip = np.fliplr(img)
    feature1 = self.face_recognizer.Recognize(img)
    feature2 = self.face_recognizer.Process(img_flip)
    if len(feature1) == 512 and len(feature2) == 512:
      feature1.extend(feature2)
      result_feature = np.array(feature1)
    else:
      result_feature = np.random.rand(1024)
    assert result_feature.size == 1024
    return result_feature