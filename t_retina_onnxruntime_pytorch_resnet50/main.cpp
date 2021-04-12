#include "FaceDetection.h"
#include "utils.h"
#include "boost/filesystem.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <chrono>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main() {

    std::string save_folder = "../prediction";
    std::string dataset_folder = "/mnt/hdd/PycharmProjects/face_eval/widerface_val/images";
    std::string tesetset_list = "/mnt/hdd/PycharmProjects/face_eval/widerface_val/wider_val.txt";
    std::vector<std::string> test_dataset = readLines(tesetset_list);

#ifdef ORIGINAL_SIZE
    std::string modelFilepath{"../model/fd_resnet50_dynamic.onnx"};
#else
    std::string modelFilepath{"../model/fd_resnet50_1600.onnx"};
#endif
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING);
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(8);
#ifdef USE_OPENVINO
    OrtOpenVINOProviderOptions options;
    options.device_type = "CPU_FP32";
    options.num_of_threads = 8;
    sessionOptions.AppendExecutionProvider_OpenVINO(options);
#endif
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);
    std::vector<const char*> inputNames{"input"};
    std::vector<const char*> outputNames{"output", "outputt", "outputtt"};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    for (std::string image_name : test_dataset) {
        std::string image_path = dataset_folder + image_name;
        cv::Mat img = cv::imread(image_path);
        if (!img.data) {
            printf("Load image error\n");
            throw std::exception();
        }
        std::string save_name = save_folder + image_name.substr(0, image_name.size() - 4) + ".txt";
        boost::filesystem::path p(save_name);
        boost::filesystem::path dir = p.parent_path();
        std::string dirname = dir.string();
        if(!boost::filesystem::exists(dirname)){
            bool isCreated = boost::filesystem::create_directories(dirname);
            printf("%s is created: %s\n", dirname.c_str(), std::to_string(isCreated).c_str());
        }

        /////////////////////////////////////////////////////////////////////////

        cv::Mat originImage= img.clone();
        cv::Mat resizedImage, preprocessedImage;
        float scale;
#ifdef ORIGINAL_SIZE
        scale = 1;
#else
        PreProcess(img, img, scale);
#endif
        std::vector<box> anchor;
        create_anchor_retinaface(anchor, img.cols, img.rows);
        int anchorsSize = anchor.size();
        std::vector<int> outputDims{anchorsSize*4, anchorsSize*2, anchorsSize*10};
        std::vector<int64_t> inputDims {1, 3, img.rows, img.cols};
        img.convertTo(resizedImage, CV_32F);
        cv::Mat channels[3];
        cv::split(resizedImage, channels);
        channels[0] = channels[0] - 104;
        channels[1] = channels[1] - 117;
        channels[2] = channels[2] - 123;
        cv::merge(channels, 3, resizedImage);
        cv::dnn::blobFromImage(resizedImage, preprocessedImage);
        Ort::Value inputTensors = Ort::Value::CreateTensor<float>(memoryInfo, (float*)preprocessedImage.data, img.rows*img.cols*3, inputDims.data(), 4);
        std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensors, 1, outputNames.data(), 3);
        std::vector<bbox> predictedBoxes;
        std::vector<std::vector<float>> results;
        for (int l = 0; l < 3; l++) {
            std::vector<float> outputi = std::vector<float>(ort_outputs[l].GetTensorMutableData<float>(), ort_outputs[l].GetTensorMutableData<float>() + outputDims[l]);
            results.emplace_back(outputi);
        }
        Detect(predictedBoxes, results, img.rows, img.cols, anchor);
        for (auto &box:predictedBoxes) {
            box.x1 = box.x1 / scale;
            box.y1 = box.y1 / scale;
            box.x2 = box.x2 / scale;
            box.y2 = box.y2 / scale;
            for (auto &point:box.point) {
                point._x = point._x / scale;
                point._y = point._y / scale;
            }
        }
#ifdef SHOW_IMG
        ShowBoxLandmarkFaces(originImage, predictedBoxes, 1000);
#endif
        WriteResultToFile(save_name, predictedBoxes);
    }
}

