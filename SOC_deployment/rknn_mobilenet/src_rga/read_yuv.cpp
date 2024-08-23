#include "yuv.h"
#include <iostream>
#include <fstream>

// int readYUV420NV12(const std::string& filepath, int width, int height, char* data) {
//     size_t frameSize = width * height;
//     size_t uvSize = frameSize / 2;
//     size_t totalSize = frameSize + uvSize;

//     std::ifstream file(filepath, std::ios::binary);
//     if (!file.is_open()) {
//         std::cerr << "Error opening file: " << filepath << std::endl;
//         return -1;
//     }

//     // 读取整个YUV数据到缓冲区
//     file.read(data, totalSize);

//     if (file.fail()) {
//         std::cerr << "Error reading file: " << filepath << std::endl;
//         file.close();
//         return -1;
//     }

//     file.close();

//     // 返回成功标志
//     return 0;
// }

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



