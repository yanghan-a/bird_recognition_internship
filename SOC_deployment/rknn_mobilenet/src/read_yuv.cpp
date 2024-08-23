#include "read_yuv.h"
#include <iostream>
#include <fstream>

int readYUV420NV12(const std::string& filepath, int width, int height, char* data) {
    size_t frameSize = width * height;
    size_t uvSize = frameSize / 2;
    size_t totalSize = frameSize + uvSize;

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filepath << std::endl;
        return -1;
    }

    // 读取整个YUV数据到缓冲区
    file.read(data, totalSize);

    if (file.fail()) {
        std::cerr << "Error reading file: " << filepath << std::endl;
        file.close();
        return -1;
    }

    file.close();

    // 返回成功标志
    return 0;
}

int readYUV420P(const std::string& filepath, int width, int height, char* data) {
    size_t frameSize = width * height;
    size_t uvSize = frameSize / 2;
    size_t totalSize = frameSize + uvSize;

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filepath << std::endl;
        return -1;
    }

    // 读取整个YUV数据到缓冲区
    file.read(data, totalSize);

    if (file.fail()) {
        std::cerr << "Error reading file: " << filepath << std::endl;
        file.close();
        return -1;
    }

    file.close();

    // 返回成功标志
    return 0;
}


int resizeYUV420P(char* src, char* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
    // 计算缩放比例
    float xScale = static_cast<float>(srcWidth) / dstWidth;
    float yScale = static_cast<float>(srcHeight) / dstHeight;

    // 缩小Y分量
    for (int j = 0; j < dstHeight; ++j) {
        for (int i = 0; i < dstWidth; ++i) {
            int srcX = static_cast<int>(i * xScale);
            int srcY = static_cast<int>(j * yScale);
            dst[j * dstWidth + i] = src[srcY * srcWidth + srcX];
        }
    }

    // 缩小U分量和V分量
    char* srcU = src + srcWidth * srcHeight;
    char* srcV = srcU + (srcWidth / 2) * (srcHeight / 2);
    char* dstU = dst + dstWidth * dstHeight;
    char* dstV = dstU + (dstWidth / 2) * (dstHeight / 2);

    // 计算UV分量缩放比例，后面debug添加的
    float xScaleUV = static_cast<float>(srcWidth / 2) / (dstWidth / 2);
    float yScaleUV = static_cast<float>(srcHeight / 2) / (dstHeight / 2);

    for (int j = 0; j < dstHeight / 2; ++j) {
        for (int i = 0; i < dstWidth / 2; ++i) {
            int srcX = static_cast<int>(i * xScaleUV);
            int srcY = static_cast<int>(j * yScaleUV);
            dstU[j * (dstWidth / 2) + i] = srcU[srcY * (srcWidth / 2) + srcX];
            dstV[j * (dstWidth / 2) + i] = srcV[srcY * (srcWidth / 2) + srcX];
        }
    }
    return 0;
}


