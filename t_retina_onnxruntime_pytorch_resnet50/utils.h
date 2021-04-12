#include <chrono>
#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <boost/filesystem/path.hpp>


class Timer {
public:
    Timer() : beg_(clock_::now()) {}

    void reset() { beg_ = clock_::now(); }

    double elapsed() const {
        return std::chrono::duration_cast<second_>
                (clock_::now() - beg_).count();
    }

    double out(std::string message = "") {
        double t = elapsed();
        std::cout << message << ":" << t * 1000 << "ms\n" << std::endl;
        reset();
        return t;
    }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

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

void ShowBoxLandmarkFaces(cv::Mat img, std::vector<bbox> boxes, int delay){
    for (int j = 0; j < boxes.size(); ++j) {
        cv::Rect rect(boxes[j].x1 , boxes[j].y1 , boxes[j].x2  - boxes[j].x1 , boxes[j].y2  - boxes[j].y1 );
        cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
        char test[80];
        sprintf(test, "%f", boxes[j].s);
        cv::putText(img, test, cv::Size((boxes[j].x1 ), boxes[j].y1 ), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255));
        cv::circle(img, cv::Point(boxes[j].point[0]._x , boxes[j].point[0]._y ), 1, cv::Scalar(0, 0, 225), 4);
        cv::circle(img, cv::Point(boxes[j].point[1]._x , boxes[j].point[1]._y ), 1, cv::Scalar(0, 255, 225), 4);
        cv::circle(img, cv::Point(boxes[j].point[2]._x , boxes[j].point[2]._y ), 1, cv::Scalar(255, 0, 225), 4);
        cv::circle(img, cv::Point(boxes[j].point[3]._x , boxes[j].point[3]._y ), 1, cv::Scalar(0, 255, 0), 4);
        cv::circle(img, cv::Point(boxes[j].point[4]._x , boxes[j].point[4]._y ), 1, cv::Scalar(255, 0, 0), 4);
    }
    cv::imshow("img", img);
    cv::waitKey(delay);
}

void WriteResultToFile(std::string& save_name, std::vector<bbox> predictedBoxes){
    std::ofstream myfile;
    boost::filesystem::path p(save_name);
    std::string abs_file_name = p.filename().string();
    std::string file_name = abs_file_name.substr(0, abs_file_name.size() - 4) + "\n";
    myfile.open(save_name);
    myfile <<file_name;
    printf("%s\n", save_name.c_str());
    myfile << std::to_string(predictedBoxes.size()) + "\n";

    for (auto box : predictedBoxes) {
        cv::Rect cvbox = cv::Rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
        int x = cvbox.x , y = cvbox.y , w = cvbox.width , h = cvbox.height;
        float confidence = box.s;
        std::string line =
                std::to_string(x) + " " + std::to_string(y) + " " + std::to_string(w) + " " + std::to_string(h) +
                " " + std::to_string(confidence) + " \n";
        myfile <<line;
    }
    myfile.close();
}