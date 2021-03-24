#include "face_recognition.h"
#include <iostream>
#include "boost/filesystem.hpp"


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


int main() {
    std::string save_folder = "/mnt/hdd/PycharmProjects/face_eval/t_retina_ncnn_mobilenetv2/prediction";
    std::string dataset_folder = "/mnt/hdd/PycharmProjects/face_eval/widerface_val/images";
    std::string testset_folder = dataset_folder;
    std::string tesetset_list = "/mnt/hdd/PycharmProjects/face_eval/widerface_val/wider_val.txt";
    std::vector<std::string> test_dataset = readLines(tesetset_list);
    std::string detectorParamPath = "/mnt/hdd/CLionProjects/frvt1N/1N/config/retina.param";
    std::string detectorBinPath = "/mnt/hdd/CLionProjects/frvt1N/1N/config/retina.bin";
    std::unique_ptr<FaceRecognizer> faceRecognizer(new FaceRecognizer(detectorParamPath, detectorBinPath));
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

        std::vector<Anchor> result = faceRecognizer->Detect(img);
        float scale = faceRecognizer->scale;
        std::string abs_file_name = p.filename().string();
        std::string file_name = abs_file_name.substr(0, abs_file_name.size() - 4) + "\n";
        myfile.open(save_name);
        myfile <<file_name;
        printf("%s\n", save_name.c_str());
        myfile << std::to_string(result.size()) + "\n";

        for (auto box : result) {
            cv::Rect cvbox = cv::Rect(box.finalbox.x, box.finalbox.y,
                                      box.finalbox.width - box.finalbox.x,
                                      box.finalbox.height - box.finalbox.y);
            int x = cvbox.x / scale, y = cvbox.y / scale, w = cvbox.width / scale, h = cvbox.height/scale;

//            cv::rectangle(
//                    img,
//                    cv::Point(x, y),
//                    cv::Point(x + w, y + h),
//                    cv::Scalar(0, 0, 255),
//                    2
//            );

            float confidence = box.score;
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

