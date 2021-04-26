#include "utils.h"
#include "boost/filesystem.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <string>
#include <vector>

int main() {

    std::string saveFolder = "../prediction";
    std::string datasetFolder = "/mnt/hdd/PycharmProjects/face_eval/widerface_val/images";
    std::string testSetList = "/mnt/hdd/PycharmProjects/face_eval/widerface_val/wider_val.txt";
    std::vector<std::string> testDataset = readLines(testSetList);

#ifdef ORIGINAL_SIZE
    std::string modelFilepath{"../model/fd_resnet50_dynamic.onnx"};
#else
    std::string modelFilepath{"../model/fd_resnet50_1600.onnx"};
#endif

    FaceRecognizer faceRecognizer(true, modelFilepath, false, "");

    for (std::string imageName : testDataset) {
        std::string image_path = datasetFolder + imageName;
        cv::Mat img = cv::imread(image_path);
        if (!img.data) {
            printf("Load image error\n");
            throw std::exception();
        }
        std::string save_name = saveFolder + imageName.substr(0, imageName.size() - 4) + ".txt";
        boost::filesystem::path p(save_name);
        boost::filesystem::path dir = p.parent_path();
        std::string dirname = dir.string();
        if(!boost::filesystem::exists(dirname)){
            bool isCreated = boost::filesystem::create_directories(dirname);
            printf("%s is created: %s\n", dirname.c_str(), std::to_string(isCreated).c_str());
        }
        std::vector<Box> outputBoxes;
        cv::Mat detectImg = img.clone();
        outputBoxes = faceRecognizer.Detect(detectImg, false);
        WriteResultToFile(save_name, outputBoxes);
    }
}