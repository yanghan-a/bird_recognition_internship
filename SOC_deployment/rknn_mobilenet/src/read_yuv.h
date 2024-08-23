#ifndef READ_YUV_H
#define READ_YUV_H

#include <string>
#include <cstdint>

int readYUV420NV12(const std::string& filepath, int width, int height, char* data);

int resizeYUV420P(char* src, char* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight);

int readYUV420P(const std::string& filepath, int width, int height, char* data) ;


#endif // READ_YUV_H
