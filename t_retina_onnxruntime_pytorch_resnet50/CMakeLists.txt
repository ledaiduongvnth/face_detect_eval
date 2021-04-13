cmake_minimum_required(VERSION 3.2)

project(retinaface-pytorch-onnxruntime-resnet50)

set(ORIGINAL_SIZE OFF)
set(USE_OPENVINO ON)
set(LOG_TIME ON)
set(SHOW_IMG OFF)


if (ORIGINAL_SIZE)
    add_definitions(-DORIGINAL_SIZE)
endif()

if (LOG_TIME)
    add_definitions(-DLOG_TIME)
endif()

if (SHOW_IMG)
    add_definitions(-DSHOW_IMG)
endif ()

if (USE_OPENVINO)
    add_definitions(-DUSE_OPENVINO)
endif ()

add_executable(
        retinaface-pytorch-onnxruntime-resnet50
        main.cpp
        /mnt/hdd/CLionProjects/frvt1N/1N/src/vtsimp/src/FaceRecognition.h
        /mnt/hdd/CLionProjects/frvt1N/1N/src/vtsimp/src/FaceDetectionUtils.h
        utils.h
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
        retinaface-pytorch-onnxruntime-resnet50
        -lboost_filesystem -lboost_system
        ${OpenCV_LIBS}
        -lstdc++fs
        ${InferenceEngine_LIBRARIES}
        /mnt/hdd/CLionProjects/frvt1N/1N/lib/onnxruntime-linux-x64-1.7.0-openvino/lib/libonnxruntime.so
        /mnt/hdd/CLionProjects/frvt1N/1N/lib/onnxruntime-linux-x64-1.7.0-openvino/lib/libonnxruntime_providers_shared.so
        /mnt/hdd/CLionProjects/frvt1N/1N/lib/onnxruntime-linux-x64-1.7.0-openvino/lib/libonnxruntime_providers_openvino.so
)
