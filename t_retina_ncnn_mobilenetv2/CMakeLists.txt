cmake_minimum_required(VERSION 3.2)
project(retinafacencnnmnetv2)

add_executable(
        retinafacencnnmnetv2
        main.cpp
        config.h
        config.cpp
        tools.h
        tools.cpp
        anchor_generator.h
        anchor_generator.cpp
        face_recognition.h
)
find_package(OpenCV REQUIRED)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/include
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
        retinafacencnnmnetv2
        -lboost_filesystem -lboost_system
        ${OpenCV_LIBS}
        ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/lib/libncnn.a
)
