import yaml
import cv2
import numpy as np
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from scrfd import SCRFD
detector = SCRFD(model_file='/mnt/hdd/PycharmProjects/insightface/detection/scrfd/scrfd_34g_n1/scrfd_34g_shape480x480.onnx')
detector.prepare(-1)


with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f)

if __name__ == '__main__':
    model_path = 'models'
    scene = 'non-mask'
    model_category = 'face_alignment'
    model_name = model_conf[scene][model_category]

    faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
    model, cfg = faceAlignModelLoader.load_model()
    faceAlignModelHandler = FaceAlignModelHandler(model, 'cuda:0', cfg)

    image_path = '/home/d/Documents/selfie.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    bboxes, kpss = detector.detect(image, 0.01, input_size=(480, 480))
    bboxes = np.array(sorted(bboxes, key=lambda a_entry: a_entry[4],  reverse=True))
    bbox = bboxes[0].astype(np.int)
    landmarks = faceAlignModelHandler.inference_on_image(image, bbox)
    image_show = image.copy()
    for (x, y) in landmarks.astype(np.int32):
        cv2.circle(image_show, (x, y), 2, (255, 0, 0), -1)
    cv2.imshow("aaa", image_show)
    cv2.waitKey(0)
