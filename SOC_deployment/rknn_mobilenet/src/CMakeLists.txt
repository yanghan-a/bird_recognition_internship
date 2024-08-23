cmake_minimum_required(VERSION 3.4.1)

project(rknn_mobilenet)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../runtime/librknn_api)
set(RKNN_RT_LIB ${RKNN_API_PATH}/armhf-uclibc/librknnmrt.so)

# rga
set(RGA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/rga)
set(LIBRGA ${RGA_PATH}/libs/librga.so)
set(LIBRGA_INCLUDES ${RGA_PATH}/include)

# dma
# include(${CMAKE_CURRENT_SOURCE_DIR}/../utils/CMakeLists.txt)

# include rknn
include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/../../3rdparty)
# include rga
include_directories(${LIBRGA_INCLUDES})


set(CMAKE_INSTALL_RPATH "lib")

add_executable(rknn_mobilenet
    main.cpp

    utils.cpp
    inference.cpp
    read_yuv.cpp
    yuv_to_rgb.cpp

    save_ppm.cpp
    
)

target_link_libraries(rknn_mobilenet
  ${RKNN_RT_LIB}
  ${LIBRGA}
)

set(model ${CMAKE_CURRENT_SOURCE_DIR}/../model)
set(pictures ${CMAKE_CURRENT_SOURCE_DIR}/../pictures)
install(DIRECTORY ${model} DESTINATION ./)
# install(DIRECTORY ${pictures} DESTINATION ./)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/../install/rknn_mobilenet)

install(TARGETS rknn_mobilenet DESTINATION ./)
# install(DIRECTORY model DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${LIBRGA} DESTINATION lib)
