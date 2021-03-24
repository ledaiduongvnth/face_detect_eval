//
// Created by d on 22/10/2020.
//

#ifndef FACE_DETECTION_FACE_RECOGNITION_H
#define FACE_DETECTION_FACE_RECOGNITION_H

#include "anchor_generator.h"
#include "opencv2/opencv.hpp"
#include "config.h"
#include "tools.h"
#include <string>
#include <experimental/filesystem>

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

    FaceRecognizer(std::string detectorParamPath, std::string detectorBinPath) {
        R50RetinaFace.load_param(detectorParamPath.data());
        R50RetinaFace.load_model(detectorBinPath.data());
    }

    ~FaceRecognizer(){
        R50RetinaFace.clear();
    }

    cv::Mat PreProcess(const cv::Mat &img) {
        cv::Mat scaledImage;
        float long_side = std::max(img.cols, img.rows);
        scale = modelSize / long_side;
        if (img.cols > img.rows) {
            cv::copyMakeBorder(img, scaledImage, 0, img.cols - img.rows, 0, 0,
                               cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        } else if (img.cols < img.rows) {
            cv::copyMakeBorder(img, scaledImage, 0, 0, 0, img.rows - img.cols,
                               cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        } else{
            scaledImage = img;
        }
        cv::resize(scaledImage, scaledImage, cv::Size(modelSize, modelSize));
        return scaledImage;
    }

    std::vector<cv::Rect> Detect(cv::Mat& originImage) {
        cv::Mat scaledImage = PreProcess(originImage);
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
        std::vector<cv::Rect> boundingBoxes;
        for (int i = 0; i < result.size(); i++) {
            cv::Rect box = cv::Rect(result[i].finalbox.x, result[i].finalbox.y,
                                    result[i].finalbox.width - result[i].finalbox.x,
                                    result[i].finalbox.height - result[i].finalbox.y);
            boundingBoxes.push_back(box);
        }
        return boundingBoxes;
    }

};
#endif //FACE_DETECTION_FACE_RECOGNITION_H
