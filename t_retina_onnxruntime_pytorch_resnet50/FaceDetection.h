#include <vector>
#include <opencv2/opencv.hpp>

struct Point{
    float _x;
    float _y;
};
struct bbox{
    float x1;
    float y1;
    float x2;
    float y2;
    float s;
    Point point[5];
};

struct box{
    float cx;
    float cy;
    float sx;
    float sy;
};



void PreProcess(const cv::Mat &img, cv::Mat& outputImg, float& scale) {
    float long_side = std::max(img.cols, img.rows);
    scale = 1600 / long_side;
    cv::Mat resizedImg;
    cv::resize(img, resizedImg, cv::Size(), scale, scale,cv::INTER_LINEAR);
    if (resizedImg.cols > resizedImg.rows) {
        cv::copyMakeBorder(resizedImg, outputImg, 0, resizedImg.cols - resizedImg.rows, 0, 0,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    } else if (resizedImg.cols < resizedImg.rows) {
        cv::copyMakeBorder(resizedImg, outputImg, 0, 0, 0, resizedImg.rows - resizedImg.cols,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    } else{
        outputImg = resizedImg;
    }
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
        if (score[indexsco+1] > 0.5)
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