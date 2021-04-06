#include "FaceDetector.h"
#include <iostream>
#include "boost/filesystem.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

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

bool cmp(bbox a, bbox b) {
    if (a.s > b.s)
        return true;
    return false;
}

void nms(std::vector<bbox> &input_boxes, float NMS_THRESH)
{
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float   h = std::max(float(0), yy2 - yy1 + 1);
            float   inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}


void Detect(std::vector<bbox>& boxes,  std::vector<std::vector<float>> results, int inh, int inw, std::vector<box>& anchor)
{
    std::vector<bbox > total_box;
    std::vector<float> loc = results[0];
    std::vector<float> landms = results[2];
    std::vector<float> score = results[1];
    int indexLoc=0;
    int indexLm=0;
    int indexsco=0;

    // #pragma omp parallel for num_threads(2)
    for (int i = 0; i < anchor.size(); ++i)
    {
        if (score[indexsco+1] > 0.99)
        {
            box tmp = anchor[i];
            box tmp1;
            bbox result;
            // loc and conf
            tmp1.cx = tmp.cx + loc[indexLoc] * 0.1 * tmp.sx;
            tmp1.cy = tmp.cy + loc[indexLoc+1] * 0.1 * tmp.sy;
            tmp1.sx = tmp.sx * exp(loc[indexLoc+2] * 0.2);
            tmp1.sy = tmp.sy * exp(loc[indexLoc+3] * 0.2);
            result.x1 = (tmp1.cx - tmp1.sx/2) * inw;
            if (result.x1<0)
                result.x1 = 0;
            result.y1 = (tmp1.cy - tmp1.sy/2) * inh;
            if (result.y1<0)
                result.y1 = 0;
            result.x2 = (tmp1.cx + tmp1.sx/2) * inw;
            if (result.x2>inw)
                result.x2 = inw;
            result.y2 = (tmp1.cy + tmp1.sy/2)* inh;
            if (result.y2>inh)
                result.y2 = inh;
            result.s = score[indexsco+1];
            // landmark
            for (int j = 0; j < 5; ++j)
            {
                result.point[j]._x =( tmp.cx + landms[indexLm + (j<<1)] * 0.1 * tmp.sx ) * inw;
                result.point[j]._y =( tmp.cy + landms[indexLm + (j<<1) + 1] * 0.1 * tmp.sy ) * inh;
            }

            total_box.push_back(result);
        }
        indexLoc += 4;
        indexsco += 2;
        indexLm += 10;
    }

    std::sort(total_box.begin(), total_box.end(), cmp);
    nms(total_box, 0.4);
//    printf("Total box %d\n", indexsco);

    for (int j = 0; j < total_box.size(); ++j)
    {
        boxes.push_back(total_box[j]);
    }
}


void create_anchor_retinaface(std::vector<box> &anchor, int w, int h)
{
//    anchor.reserve(num_boxes);
    anchor.clear();
    std::vector<std::vector<int> > feature_map(3), min_sizes(3);
    float steps[] = {8, 16, 32};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));
    }
    std::vector<int> minsize1 = {16, 32};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {64, 128};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {256, 512};
    min_sizes[2] = minsize3;

    for (int k = 0; k < feature_map.size(); ++k)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l = 0; l < min_size.size(); ++l)
                {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    float cx = (j + 0.5) * steps[k]/w;
                    float cy = (i + 0.5) * steps[k]/h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }

    }

}


int main() {
    std::string modelFilepath{"/mnt/hdd/PycharmProjects/Pytorch_Retinaface/FaceDetector.onnx"};
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING);
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    OrtOpenVINOProviderOptions options;
    options.device_type = "CPU_FP32";
    options.num_of_threads = 1;
    sessionOptions.AppendExecutionProvider_OpenVINO(options);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);
    std::vector<const char*> inputNames{"input"};
    std::vector<const char*> outputNames{"output", "outputt", "outputtt"};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);



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
        std::vector<box> anchor;
        create_anchor_retinaface(anchor, img.cols, img.rows);
        int anchorsSize = anchor.size();
        std::vector<int> outputDims{anchorsSize*4, anchorsSize*2, anchorsSize*10};
        cv::Mat originImage= img.clone();
        cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
        float scale = 1;
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
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensors, 1, outputNames.data(), 3);
        std::vector<bbox> result;
        std::vector<std::vector<float>> results;
        for (int l = 0; l < 3; l++) {
            std::vector<float> outputi = std::vector<float>(ort_outputs[l].GetTensorMutableData<float>(), ort_outputs[l].GetTensorMutableData<float>() + outputDims[l]);
            results.emplace_back(outputi);
        }
        Detect(result, results, img.rows, img.cols, anchor);

//        for (int j = 0; j < result.size(); ++j) {
//            cv::Rect rect(result[j].x1/scale, result[j].y1/scale, result[j].x2/scale - result[j].x1/scale, result[j].y2/scale - result[j].y1/scale);
//            cv::rectangle(originImage, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
//            char test[80];
//            sprintf(test, "%f", result[j].s);
//            cv::putText(originImage, test, cv::Size((result[j].x1/scale), result[j].y1/scale), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
//            cv::circle(originImage, cv::Point(result[j].point[0]._x / scale, result[j].point[0]._y / scale), 1, cv::Scalar(0, 0, 225), 4);
//            cv::circle(originImage, cv::Point(result[j].point[1]._x / scale, result[j].point[1]._y / scale), 1, cv::Scalar(0, 255, 225), 4);
//            cv::circle(originImage, cv::Point(result[j].point[2]._x / scale, result[j].point[2]._y / scale), 1, cv::Scalar(255, 0, 225), 4);
//            cv::circle(originImage, cv::Point(result[j].point[3]._x / scale, result[j].point[3]._y / scale), 1, cv::Scalar(0, 255, 0), 4);
//            cv::circle(originImage, cv::Point(result[j].point[4]._x / scale, result[j].point[4]._y / scale), 1, cv::Scalar(255, 0, 0), 4);
//        }
//        cv::imshow("img", originImage);
//        cv::waitKey(0);

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

