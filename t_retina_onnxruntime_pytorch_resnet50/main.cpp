#include "FaceDetector.h"
#include <iostream>
#include "boost/filesystem.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "FaceDetector.h"

std::vector<std::string> readLines(const std::string &filename) {
    std::ifstream ifs(filename);
    std::vector<std::string> lines;
    if (!ifs) {
        std::cerr << "Cannot open file: " << filename << std::endl;
    } else {
        for (std::string line; std::getline(ifs, line); /**/) {
            lines.push_back(line);
        }
        std::cout << std::to_string(lines.size()) << " lines read from [" << filename << "]" << std::endl;
    }
    return lines;
}

void PreProcess(const cv::Mat &img, cv::Mat& outputImg, float& scale) {
    float long_side = std::max(img.cols, img.rows);
    scale = 640 / long_side;
    if (img.cols > img.rows) {
        cv::copyMakeBorder(img, outputImg, 0, img.cols - img.rows, 0, 0,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    } else if (img.cols < img.rows) {
        cv::copyMakeBorder(img, outputImg, 0, 0, 0, img.rows - img.cols,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    } else{
        outputImg = img;
    }
    cv::resize(outputImg, outputImg, cv::Size(640, 640), 0, 0,cv::InterpolationFlags::INTER_CUBIC);
}


int main() {
    Detector detector;
    std::string modelFilepath{"/mnt/hdd/PycharmProjects/Pytorch_Retinaface/FaceDetector.onnx"};
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING);
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    OrtOpenVINOProviderOptions options;
    options.device_type = "CPU_FP32";
    options.num_of_threads = 8;
    sessionOptions.AppendExecutionProvider_OpenVINO(options);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);
    std::vector<int64_t> inputDims {1, 3, 640, 640};
    std::vector<const char*> inputNames{"input0"};
    std::vector<const char*> outputNames{"output0", "833", "832"};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    std::vector<int> outputDims{16800*4, 16800*2, 16800*10};

    std::string save_folder = "/mnt/hdd/PycharmProjects/face_eval/t_retina_onnxruntime_pytorch_resnet50/prediction";
    std::string dataset_folder = "/mnt/hdd/PycharmProjects/face_eval/widerface_val/images";
    std::string testset_folder = dataset_folder;
    std::string tesetset_list = "/mnt/hdd/PycharmProjects/face_eval/widerface_val/wider_val.txt";
    std::vector<std::string> test_dataset = readLines(tesetset_list);

    for (std::string image_name : test_dataset) {
        std::string image_path = testset_folder + image_name;
        cv::Mat img = cv::imread(image_path);
        if (!img.data) {
            printf("load error");
            throw std::exception();
        }
        std::string save_name = save_folder + image_name.substr(0, image_name.size() - 4) + ".txt";
        std::ofstream myfile;
        boost::filesystem::path p(save_name);
        boost::filesystem::path dir = p.parent_path();
        std::string dirname = dir.string();
        if(!boost::filesystem::exists(dirname)){
            bool isCreated = boost::filesystem::create_directories(dirname);
            printf("%s\n", std::to_string(isCreated).c_str());
            printf("%s\n", dirname.c_str());
            usleep(1000);
        }

        while (!boost::filesystem::exists(dirname)){
            printf("waiting ....");
            usleep(1000);
        }

        /////////////////////////////////////////////////////////////////////////
        cv::Mat originImage= img.clone();
        cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
        float scale;
        PreProcess(img, resizedImageBGR, scale);
        resizedImageBGR.convertTo(resizedImage, CV_32F);
        cv::Mat channels[3];
        cv::split(resizedImage, channels);
        channels[0] = channels[0] - 104;
        channels[1] = channels[1] - 117;
        channels[2] = channels[2] - 123;
        cv::merge(channels, 3, resizedImage);
        cv::dnn::blobFromImage(resizedImage, preprocessedImage);
        Ort::Value inputTensors = Ort::Value::CreateTensor<float>(memoryInfo, (float*)preprocessedImage.data, 640*640*3, inputDims.data(), 4);
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensors, 1, outputNames.data(), 3);
        std::vector<bbox> result;
        std::vector<std::vector<float>> results;
        for (int l = 0; l < 3; l++) {
            std::vector<float> outputi = std::vector<float>(ort_outputs[l].GetTensorMutableData<float>(), ort_outputs[l].GetTensorMutableData<float>() + outputDims[l]);
            results.emplace_back(outputi);
        }
        detector.Detect(result, results);
        /////////////////////////////////////////////////////////////////////////
        std::string abs_file_name = p.filename().string();
        std::string file_name = abs_file_name.substr(0, abs_file_name.size() - 4) + "\n";
        myfile.open(save_name);
        myfile <<file_name;
        printf("%s\n", save_name.c_str());
        myfile << std::to_string(result.size()) + "\n";

        for (auto box : result) {
            cv::Rect cvbox = cv::Rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
            int x = cvbox.x / scale, y = cvbox.y / scale, w = cvbox.width / scale, h = cvbox.height/scale;

            cv::rectangle(
                    img,
                    cv::Point(x, y),
                    cv::Point(x + w, y + h),
                    cv::Scalar(0, 0, 255),
                    2
            );

            float confidence = box.s;
            std::string line =
                    std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(w) + " " + std::to_string(h) +
                    " " + std::to_string(confidence) + " \n";
            myfile <<line;
        }
        myfile.close();

//        cv::imshow("aaa", img);
//        cv::waitKey(0);
    }
}

