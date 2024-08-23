#!/bin/bash
./build-linux_RV1106.sh


adb push ./install/rknn_mobilenet/rknn_mobilenet ./test/rknn_mobilenet
# adb push ./install/rknn_mobilenet ./test/



