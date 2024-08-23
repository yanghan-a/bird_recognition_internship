#ifndef RGB_H
#define RGB_H
#include <cstdint>

int rga_crop_rgb(char * src_buf, char * dst_buf, int src_dma_fd, int dst_dma_fd, int src_width,int src_height,int dst_width,int dst_height, int crop_x, int crop_y);
#endif // YUV_TO_RGB_H
