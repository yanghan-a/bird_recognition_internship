#include "yuv_to_rgb.h"
#include <iostream>
#include <cstdint>
#include <algorithm>

int yuv420nv12_to_rgb(char* yuv420nv12_img, char* rgb_img, int width, int height) {
    int frameSize = width * height;
    char* yPlane = yuv420nv12_img;
    char* uvPlane = yPlane + frameSize;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int yIndex = y * width + x;
            int uvIndex = (y / 2) * (width / 2) * 2 + (x & ~1);

            uint8_t Y = yPlane[yIndex];
            int8_t U = uvPlane[uvIndex] - 128;
            int8_t V = uvPlane[uvIndex + 1] - 128;

            int R = Y + 1.402 * V;
            int G = Y - 0.344136 * U - 0.714136 * V;
            int B = Y + 1.772 * U;

            R = std::min(std::max(R, 0), 255);
            G = std::min(std::max(G, 0), 255);
            B = std::min(std::max(B, 0), 255);

            int rgbIndex = (y * width + x) * 3;
            rgb_img[rgbIndex] = R;
            rgb_img[rgbIndex + 1] = G;
            rgb_img[rgbIndex + 2] = B;
        }
    }
    return 0;
}

int YUV420PToRGB(char* yuv, char* rgb, int width, int height) {
    int frameSize = width * height;
    char* yPlane = yuv;
    char* uPlane = yuv + frameSize;
    char* vPlane = uPlane + frameSize / 4;

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            int yIndex = j * width + i;
            int uvIndex = (j / 2) * (width / 2) + (i / 2);

            int Y = yPlane[yIndex];
            int U = uPlane[uvIndex] - 128;
            int V = vPlane[uvIndex] - 128;

            int R = Y + 1.402 * V;
            int G = Y - 0.344136 * U - 0.714136 * V;
            int B = Y + 1.772 * U;

            // 将结果夹在 0 到 255 之间
            R = std::min(std::max(R, 0), 255);
            G = std::min(std::max(G, 0), 255);
            B = std::min(std::max(B, 0), 255);

            int rgbIndex = yIndex * 3;
            rgb[rgbIndex] = static_cast<uint8_t>(R);
            rgb[rgbIndex + 1] = static_cast<uint8_t>(G);
            rgb[rgbIndex + 2] = static_cast<uint8_t>(B);
        }
    }
    return 0;
}

int cropRGBImageLeftMiddleRight(char* rgb, int width, int height, int cropSize, char* leftImage, char* middleImage, char* rightImage) {
    // 确保裁剪大小不超过图像宽度或高度
    // if (cropSize > std::min(width, height)) {
    //     std::cerr << "cropSize must be less than or equal to the minimum of width and height" << std::endl;
    //     return 0;
    // }

    if (width >= height) {
        // 宽大于高，在宽的方向上裁剪
        // 裁剪左侧图像
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < cropSize; ++i) {
                int rgbIndex = (j * width + std::min(i, width - 1)) * 3;
                int leftIndex = (j * cropSize + i) * 3;
                leftImage[leftIndex] = rgb[rgbIndex];
                leftImage[leftIndex + 1] = rgb[rgbIndex + 1];
                leftImage[leftIndex + 2] = rgb[rgbIndex + 2];
            }
        }

        // 裁剪中间图像
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < cropSize; ++i) {
                int rgbIndex = (j * width + std::min(i + (width - cropSize) / 2, width - 1)) * 3;
                int middleIndex = (j * cropSize + i) * 3;
                middleImage[middleIndex] = rgb[rgbIndex];
                middleImage[middleIndex + 1] = rgb[rgbIndex + 1];
                middleImage[middleIndex + 2] = rgb[rgbIndex + 2];
            }
        }

        // 裁剪右侧图像
        for (int j = 0; j < height; ++j) {
            for (int i = 0; i < cropSize; ++i) {
                int rgbIndex = (j * width + std::min(i + (width - cropSize), width - 1)) * 3;
                int rightIndex = (j * cropSize + i) * 3;
                rightImage[rightIndex] = rgb[rgbIndex];
                rightImage[rightIndex + 1] = rgb[rgbIndex + 1];
                rightImage[rightIndex + 2] = rgb[rgbIndex + 2];
            }
        }
    } else {
        // 高大于宽，在高的方向上裁剪
        // 裁剪顶部图像
        for (int j = 0; j < cropSize; ++j) {
            for (int i = 0; i < width; ++i) {
                int rgbIndex = (j * width + i) * 3;
                int topIndex = (j * width + i) * 3;
                leftImage[topIndex] = rgb[rgbIndex];
                leftImage[topIndex + 1] = rgb[rgbIndex + 1];
                leftImage[topIndex + 2] = rgb[rgbIndex + 2];
            }
        }

        // 裁剪中间图像
        for (int j = 0; j < cropSize; ++j) {
            for (int i = 0; i < width; ++i) {
                int rgbIndex = ((j + (height - cropSize) / 2) * width + i) * 3;
                int middleIndex = (j * width + i) * 3;
                middleImage[middleIndex] = rgb[rgbIndex];
                middleImage[middleIndex + 1] = rgb[rgbIndex + 1];
                middleImage[middleIndex + 2] = rgb[rgbIndex + 2];
            }
        }

        // 裁剪底部图像
        for (int j = 0; j < cropSize; ++j) {
            for (int i = 0; i < width; ++i) {
                int rgbIndex = ((j + (height - cropSize)) * width + i) * 3;
                int bottomIndex = (j * width + i) * 3;
                rightImage[bottomIndex] = rgb[rgbIndex];
                rightImage[bottomIndex + 1] = rgb[rgbIndex + 1];
                rightImage[bottomIndex + 2] = rgb[rgbIndex + 2];
            }
        }
    }

    return 0;
}