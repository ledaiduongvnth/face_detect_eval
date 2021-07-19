import argparse
import os.path

import cv2
import numpy as np
import sys
# import mxnet as mx
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
      "/mnt/hdd/CLionProjects/frvt1N/1N/config/retinaface_resnet50_480x480.onnx",
      480, 0.0, 0.4,
      True,
      "/mnt/hdd/CLionProjects/frvt1N/1N/config/glint360k0.1.onnx"
    )

  def get(self, rimg, landmark):
    result_feature = self.face_recognizer.Process(rimg, True, True)
    result_feature = np.array(result_feature)
    assert result_feature.size == 1024
    return result_feature