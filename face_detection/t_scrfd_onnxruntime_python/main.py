from scrfd import SCRFD
import cv2

if __name__ == '__main__':
    detector = SCRFD(model_file='/mnt/hdd/PycharmProjects/insightface/detection/scrfd/scrfd_34g_n1/scrfd_34g_shape480x480.onnx')
    detector.prepare(-1)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    bboxes, kpss = detector.detect(img, 0.02, input_size=(480, 480))
