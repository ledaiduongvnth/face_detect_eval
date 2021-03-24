#include "face_recognition.h"

int main() {
    std::string ImagesPath = "/home/d/Downloads";
    std::string detectorParamPath = "/mnt/hdd/CLionProjects/frvt1N/1N/config/retina.param";
    std::string detectorBinPath = "/mnt/hdd/CLionProjects/frvt1N/1N/config/retina.bin";
    std::unique_ptr<FaceRecognizer> faceRecognizer( new FaceRecognizer(detectorParamPath, detectorBinPath));
    std::string image_path = "/mnt/hdd/CLionProjects/frvt1N/common/images/eyesClosed.ppm";
    cv::Mat originImage = cv::imread(image_path);
    if (!originImage.data) {
        printf("load error");
        throw std::exception();
    }
    std::vector<cv::Rect> boundingBoxes = faceRecognizer->Detect(originImage);
    printf("\n");
}

