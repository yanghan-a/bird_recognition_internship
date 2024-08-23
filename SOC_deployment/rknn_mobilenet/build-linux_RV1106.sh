#!/bin/bash
set -e

if [ -z $RK_RV1106_TOOLCHAIN ]; then
  echo "Please set the RK_RV1106_TOOLCHAIN environment variable!"
  echo "example:"
  echo "  export RK_RV1106_TOOLCHAIN=<path-to-your-dir/arm-rockchip830-linux-uclibcgnueabihf>"
  exit
fi

# for arm
GCC_COMPILER=$RK_RV1106_TOOLCHAIN

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# build
BUILD_DIR=${ROOT_PWD}/build/

if [[ -d "${BUILD_DIR}" ]]; then
  rm -rf ${BUILD_DIR}
fi


if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ../src \
    -DCMAKE_C_COMPILER=${GCC_COMPILER}-gcc \
    -DCMAKE_CXX_COMPILER=${GCC_COMPILER}-g++
make -j4
make install
cd -
