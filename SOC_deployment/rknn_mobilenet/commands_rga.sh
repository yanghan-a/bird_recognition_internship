#!/bin/bash
./build-linux_RV1106_RGA.sh


adb push ./install/rknn_mobilenet_rga/rknn_mobilenet_rga ./test/rknn_mobilenet_rga
# adb push ./install/rknn_mobilenet_rga ./test


