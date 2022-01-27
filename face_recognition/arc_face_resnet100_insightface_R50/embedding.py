import argparse
import cv2
import numpy as np
import sys
import mxnet as mx
import datetime
from skimage import transform as trans
import sklearn
from sklearn import preprocessing
from face_detection.retinaface import RetinaFace
import face_common as face_common

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
    self.detector = RetinaFace("/mnt/hdd/PycharmProjects/export-model-for-serving/R50",
                          0,
                          0,
                          network="net3",
                          nocrop=False,
                          vote=False)

    self.face_recognizer = face_common.FaceRecognizer(
      True,
      "/mnt/hdd/PycharmProjects/insightface/detection/scrfd/scrfd_34g_n1/scrfd_34g_shape320x320.onnx",
      320, 0.01, 0.4,
      True,
      "models/iresnet100.onnx",
      0
    )


  def get(self, rimg, _):
    do_flip = True
    # TEST_SCALES = [500, 800, 1200, 1600]
    TEST_SCALES = [640]
    target_size = 800
    max_size = 1200
    im_shape = rimg.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
      im_scale = float(max_size) / float(im_size_max)
    scales = [
      float(scale) / target_size * im_scale for scale in TEST_SCALES
    ]
    boxes, landmarks = self.detector.detect(rimg,
                                       threshold=0.02,
                                       scales=scales,
                                       do_flip=do_flip)
    # TODO find maxbox corresponding to this scenarios
    assert len(boxes) > 0
    # max_box = boxes[0]
    landmarks = landmarks[0]
    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, self.src)
    M = tform.params[0:2,:]
    alignedFace = cv2.warpAffine(rimg,M,(self.image_size[1],self.image_size[0]), borderValue = 0.0)
    cv2.imshow("img", alignedFace)
    cv2.waitKey(2000)
    featureVector = self.face_recognizer.RecognizePytorch(alignedFace)
    alignedFaceFlip = cv2.flip(alignedFace, 1)
    featureVectorFlip = self.face_recognizer.RecognizePytorch(alignedFaceFlip)
    featureVector.extend(featureVectorFlip)
    return featureVector