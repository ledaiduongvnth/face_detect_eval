//
// Created by dl on 19-7-19.
//

#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H



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
#endif //
