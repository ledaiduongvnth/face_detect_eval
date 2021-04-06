cmake_minimum_required(VERSION 3.2)
project(retinafacencnnmnetv2)

add_executable(
        retinafacencnnmnetv2
        FaceDetector.h
        FaceDetector.cpp
        main.cpp
)
find_package(OpenCV REQUIRED)
set(CMAKE_PREFIX_PATH /opt/intel/openvino_2021.2.200/deployment_tools/inference_engine/share)
find_package(InferenceEngine REQUIRED)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/include
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib
        ${OpenCV_INCLUDE_DIRS}
        ${InferenceEngine_INCLUDE_DIRS}
        /mnt/hdd/CLionProjects/onnxruntime/include/onnxruntime/core/session
)

target_link_libraries(
        retinafacencnnmnetv2
        -lboost_filesystem -lboost_system
        ${OpenCV_LIBS}
        -lstdc++fs
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}

        /mnt/hdd/CLionProjects/onnxruntime/build/Linux/Release/libonnxruntime.so
        /mnt/hdd/CLionProjects/onnxruntime/build/Linux/Release/libonnxruntime_providers_shared.so
        /mnt/hdd/CLionProjects/onnxruntime/build/Linux/Release/libonnxruntime_providers_openvino.so
)
