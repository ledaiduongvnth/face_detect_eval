//#include "src/face_recognition_class.h"
#include "src/face_recognition.h"



int main() {
    while (1){
        std::string ImagesPath = "/home/d/Downloads";
        std::string detectorParamPath = "/mnt/hdd/CLionProjects/frvt1N/1N/config/retina.param";
        std::string detectorBinPath = "/mnt/hdd/CLionProjects/frvt1N/1N/config/retina.bin";
        std::string recognizerParamPath = "/mnt/hdd/CLionProjects/frvt1N/1N/config/ncnn_face_reg.param";
        std::string recognizerBinPath = "/mnt/hdd/CLionProjects/frvt1N/1N/config/ncnn_face_reg.bin";
        std::unique_ptr<FaceRecognizer> faceRecognizer( new FaceRecognizer(detectorParamPath, detectorBinPath,
                                                                           recognizerParamPath, recognizerBinPath));
        cv::Mat alignedFace;
        std::string image_path = "/mnt/hdd/CLionProjects/frvt1N/common/images/eyesClosed.ppm";
        cv::Mat originImage = cv::imread(image_path);
        if (!originImage.data) {
            printf("load error");
            throw std::exception();
        }
        std::vector<float> featureVector;
        std::vector<float> eyes;
        std::tie(featureVector, eyes) = faceRecognizer->Process(originImage);
        for (int idx = 0; idx < 512; idx++){
            printf("%f ", featureVector[idx]);
        }
        printf("\n");
    }
}

