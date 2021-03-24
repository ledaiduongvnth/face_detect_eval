//
// Created by d on 22/10/2020.
//

#ifndef FACE_DETECTION_FACE_RECOGNITION_H
#define FACE_DETECTION_FACE_RECOGNITION_H

#include "anchor_generator.h"
#include "opencv2/opencv.hpp"
#include "config.h"
#include "tools.h"
#include "aligner.h"
#include <string>
#include <experimental/filesystem>
#include <faiss/IndexFlat.h>
#include <faiss/MetaIndexes.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/MetaIndexes.h>
#include <random>
#include "faiss/IndexIVF.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/index_factory.h"

// Vinh: manual debug
//#include <iostream>

class FaceRecognizer {
public:
    int modelSize = 640;
    float pixel_mean[3] = {0, 0, 0};
    float pixel_std[3] = {1, 1, 1};
    float scale;
    std::vector<cv::Point2f> landmarks;
    ncnn::Net R50RetinaFace;
    ncnn::Net R100ArcFace;

    FaceRecognizer(std::string detectorParamPath, std::string detectorBinPath,
                   std::string recognizerParamPath, std::string recognizerBinPath) {
        R50RetinaFace.load_param(detectorParamPath.data());
        R50RetinaFace.load_model(detectorBinPath.data());
        R100ArcFace.load_param(recognizerParamPath.data());
        R100ArcFace.load_model(recognizerBinPath.data());
    }

    ~FaceRecognizer(){
        R50RetinaFace.clear();
        R100ArcFace.clear();
    }

    cv::Mat PreProcess(const cv::Mat &img) {
        cv::Mat scaledImage;
        float long_side = std::max(img.cols, img.rows);
        scale = modelSize / long_side;
        cv::resize(img, scaledImage, cv::Size(img.cols * scale, img.rows * scale));
        return scaledImage;
    }

    bool Detect(cv::Mat &scaledImage, cv::Mat originImage) {
        landmarks.clear();
        ncnn::Mat input = ncnn::Mat::from_pixels_resize(
                scaledImage.data,
                ncnn::Mat::PIXEL_BGR2RGB,
                scaledImage.cols, scaledImage.rows,
                scaledImage.cols, scaledImage.rows
        );
        input.substract_mean_normalize(pixel_mean, pixel_std);
        ncnn::Extractor _extractor = R50RetinaFace.create_extractor();                
        _extractor.input("data", input);
        std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
        for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
            int stride = _feat_stride_fpn[i];
            ac[i].Init(stride, anchor_cfg[stride], false);
        }
        std::vector<Anchor> proposals;
        proposals.clear();
        for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
            ncnn::Mat cls;
            ncnn::Mat reg;
            ncnn::Mat pts;
            char clsname[100];
            sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
            char regname[100];
            sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
            char ptsname[100];
            sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);            
            _extractor.extract(clsname, cls);
            _extractor.extract(regname, reg);
            _extractor.extract(ptsname, pts);
            ac[i].FilterAnchor(cls, reg, pts, proposals);
        }
        std::vector<Anchor> result;
        nms_cpu(proposals, nms_threshold, result);
        int max_area = 0, max_index = 0;
        for (int i = 0; i < result.size(); i++) {
            cv::Rect box = cv::Rect(result[i].finalbox.x, result[i].finalbox.y,
                                    result[i].finalbox.width - result[i].finalbox.x,
                                    result[i].finalbox.height - result[i].finalbox.y);
            int temp_area = box.area();
            if (temp_area > max_area){
                max_area = temp_area;
                max_index = i;
            }
        }

        if (max_area > 0){
            for (int j = 0; j < result[max_index].pts.size(); ++j) {
                cv::Point point = cv::Point(
                        result[max_index].pts[j].x / scale,
                        result[max_index].pts[j].y / scale
                );
                landmarks.emplace_back(point);
            }
        }
        return !result.empty();
    }

    std::vector<float> Recognize(cv::Mat alignedFace) {
        ncnn::Mat input = ncnn::Mat::from_pixels(
                alignedFace.data,
                ncnn::Mat::PIXEL_BGR2RGB,
                alignedFace.cols, alignedFace.rows
        );
        ncnn::Extractor ex = R100ArcFace.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", input);
        ncnn::Mat out;
        ex.extract("fc1", out);
        std::vector<float> result;
        int dataSize = out.channel(0).w;
        for (int j = 0; j < dataSize; j++) {
            result.push_back(out.channel(0)[j]);
        }
        return result;
    }

    std::pair<std::vector<float>, std::vector<float>> Process(cv::Mat originImage) {
        std::vector<float> featureVector;
        std::vector<float> eyes;
        
        cv::Mat scaledImg = PreProcess(originImage);
        
        bool isDetected = Detect(scaledImg, originImage);
        
        if (!isDetected) {
            cv::rotate(originImage, originImage, cv::ROTATE_180);
            cv::Mat scaledImg = PreProcess(originImage);
            isDetected = Detect(scaledImg, originImage);
        }
        if (isDetected) {
            cv::Mat alignedFace;
            Aligner aligner;
            aligner.AlignFace(originImage, landmarks, &alignedFace);
            eyes.push_back(landmarks[0].x); // right eye
            eyes.push_back(landmarks[0].y);
            eyes.push_back(landmarks[1].x); // left eye
            eyes.push_back(landmarks[1].y);
            featureVector = Recognize(alignedFace);
        }
        return std::make_pair(featureVector, eyes);
    }
};
#endif //FACE_DETECTION_FACE_RECOGNITION_H
