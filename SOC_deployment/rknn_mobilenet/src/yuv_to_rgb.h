#ifndef YUV_TO_RGB_H
#define YUV_TO_RGB_H

#include <cstdint>

int yuv420nv12_to_rgb(char* yuv420nv12_img, char* rgb_img, int width, int height);
int YUV420PToRGB(char* yuv420P, char* rgb, int width, int height);
int cropRGBImageLeftMiddleRight(char* rgb, int width, int height, int cropSize, char* leftImage, char* middleImage, char* rightImage);

#endif // YUV_TO_RGB_H
