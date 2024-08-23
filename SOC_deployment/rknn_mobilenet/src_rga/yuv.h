#ifndef YUV_H
#define YUV_H

#include <string>
#include <cstdint>

int readYUV420P(const std::string& filepath, int width, int height, char* data) ;

int rga_resize_yuv(char * src_buf, char * dst_buf, int src_dma_fd, int dst_dma_fd, int src_width,int src_height,int dst_width,int dst_height);

int rga_yuv_to_rgb(char * src_buf, char * dst_buf, int src_dma_fd, int dst_dma_fd, int src_width,int src_height,int dst_width,int dst_height);
#endif // READ_YUV_H
