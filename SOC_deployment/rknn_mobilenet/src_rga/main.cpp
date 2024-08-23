#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <regex>
#include <dirent.h>
#include <sys/types.h>
#include <sstream>

#include "yuv.h"
#include "rgb.h"
#include "save_ppm.h"
#include "inference.h"
#include "dma_alloc.h"


#include <list>
#include <chrono>	// record time
#include <sys/resource.h>// record consumed resource


#define RESULTS "./results/"
#define BIRD_PATH "./pictures/bird_yuv/"
#define OTHER_PATH "./pictures/other_yuv/"


void printMemoryUsage() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        std::cout << "Memory usage: " << usage.ru_maxrss << " KB" << std::endl;
    } else {
        std::cerr << "Error getting memory usage" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::list<const char *> lst = {BIRD_PATH, OTHER_PATH};

    std::regex pattern(R"((\d+)x(\d+)_\d+\.yuv)");
    std::smatch matches;
    DIR* dir;
    struct dirent* ent;
    int all = 0;
    int part = 0;
    int bird_all = 0;
    int brid_part = 0;
    int other_all = 0;
    int other_part = 0;
    for (const auto& dir_name  : lst) {
        if ((dir = opendir(dir_name)) != NULL) {
            while ((ent = readdir(dir)) != NULL) {
                std::string filename = ent->d_name;
                std::cout << "当前文件名称：" << filename  << std::endl; // 输出字符串并换行;

                
                if (std::regex_search(filename, matches, pattern)) {
                    int ret;
                    all++;

                    printf("第%d张图片:\n",all+bird_all);
                    int width = std::stoi(matches[1]);
                    int height = std::stoi(matches[2]);
                    // 分配内存以存储YUV数据
                    char* yuv;
                    int yuv_size = width * height*3/2;
                    yuv = (char *)malloc(yuv_size);
                    if (yuv == NULL) {
                        perror("malloc failed");
                        return 1;
                    }
                    // 读取YUV文件
                    std::string filepath = dir_name + filename;
                    readYUV420P(filepath, width, height, yuv);

                    // save_rgb_to_ppm(yuv, width, height, "read_yuv.ppm");

                   char* yuv_dma;
                    int yuv_dma_fd;
                    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, yuv_size, &yuv_dma_fd, (void **)&yuv_dma);
                    if (ret < 0) {
                        printf("alloc src CMA buffer failed!\n");
                        return 1;
                    }
                    memcpy(yuv_dma, yuv, yuv_size);
                    dma_sync_cpu_to_device(yuv_dma_fd);
                    free(yuv); 

                    // save_rgb_to_ppm(yuv_dma, width, height, "copy_yuv.ppm");


                    int width_resized;
                    int height_resized;
                    // 这块是让缩放后的图像边长4对齐
                    if (width >= height){
                        width_resized = width*224/height;
                        if(width_resized%4!=0){
                            width_resized = (width_resized+3)&~3;
                        }
                        height_resized = 224;
                    }else{
                        width_resized = 224;
                        height_resized = height*224/width;
                        if(height_resized%4!=0){
                            height_resized = (height_resized+3)&~3;
                        }
                    }

                    char * yuv_resized_dma;
                    int yuv_resized_dma_fd;
                    int yuv_resized_size = width_resized * height_resized*3;
                    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, yuv_resized_size, &yuv_resized_dma_fd, (void **)&yuv_resized_dma);
                    if (ret < 0) {
                        printf("alloc src CMA buffer failed!\n");
                        return 1;
                    }
                    // 使用rga调整yuv大小
                    if(rga_resize_yuv(yuv_dma, yuv_resized_dma, yuv_dma_fd, yuv_resized_dma_fd, width, height,width_resized, height_resized) !=0){

                        std::cerr << "Failed to resize YUV file" << std::endl;
                        return 1;
                    }
                    dma_buf_free(yuv_size, &yuv_dma_fd, yuv_dma);

                    // save_rgb_to_ppm(yuv_resized_dma, width_resized, height_resized, "resized_yuv.ppm");



                    char * rgb_dma;
                    int rgb_dma_fd;
                    int rgb_size = width_resized * height_resized*3;
                    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, rgb_size, &rgb_dma_fd, (void **)&rgb_dma);
                    if (ret < 0) {
                        printf("alloc src CMA buffer failed!\n");
                        return 1;
                    }
                    // 使用rga将yuv转为rgb
                    if(rga_yuv_to_rgb(yuv_resized_dma, rgb_dma, yuv_resized_dma_fd, rgb_dma_fd, width_resized, height_resized,width_resized, height_resized) !=0){

                        std::cerr << "Failed to resize YUV file" << std::endl;
                        return 1;
                    }
                    dma_buf_free(yuv_resized_size, &yuv_resized_dma_fd, yuv_resized_dma);

                    // save_rgb_to_ppm(rgb_dma, width_resized, height_resized, "rgb.ppm");



                    

                    char* rgb_croped_dma1;
                    char* rgb_croped_dma2;
                    char* rgb_croped_dma3;
                    int rgb_croped_dma_fd1;
                    int rgb_croped_dma_fd2;
                    int rgb_croped_dma_fd3;
                    int rgb_croped_size = 224 * 224*3;
                    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, rgb_croped_size, &rgb_croped_dma_fd1, (void **)&rgb_croped_dma1);
                    if (ret < 0) {
                        printf("alloc src CMA buffer failed!\n");
                        return 1;
                    }
                    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, rgb_croped_size, &rgb_croped_dma_fd2, (void **)&rgb_croped_dma2);
                    if (ret < 0) {
                        printf("alloc src CMA buffer failed!\n");
                        return 1;
                    }
                    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, rgb_croped_size, &rgb_croped_dma_fd3, (void **)&rgb_croped_dma3);
                    if (ret < 0) {
                        printf("alloc src CMA buffer failed!\n");
                        return 1;
                    }

                    int x1,y1,x2,y2,x3,y3;
                    x1 = 0;
                    y1 = 0;
                    if(width_resized>=height_resized){
                        x2 = width_resized/2-height_resized/2;
                        y2 =0;
                        x3 = width_resized-height_resized;
                        y3 =0;
                    }else{
                        x2 = 0;
                        y2 =height_resized/2-width_resized/2;
                        x3 = 0;
                        y3 =height_resized-width_resized;
                    }
                    // 使用rga对rgb图像裁剪，裁剪为三个部分
                    if(rga_crop_rgb(rgb_dma, rgb_croped_dma1, rgb_dma_fd, rgb_croped_dma_fd1, width_resized, height_resized,224,224, x1,y1) !=0){

                        std::cerr << "Failed to resize YUV file" << std::endl;
                        return 1;
                    }
                    if(rga_crop_rgb(rgb_dma, rgb_croped_dma2, rgb_dma_fd, rgb_croped_dma_fd2, width_resized, height_resized,224,224, x2,y2) !=0){

                        std::cerr << "Failed to resize YUV file" << std::endl;
                        return 1;
                    }
                    if(rga_crop_rgb(rgb_dma, rgb_croped_dma3, rgb_dma_fd, rgb_croped_dma_fd3, width_resized, height_resized,224,224, x3,y3) !=0){

                        std::cerr << "Failed to resize YUV file" << std::endl;
                        return 1;
                    }



                    // inference(224,224,rgb_resized_data2,&a);
                    inference3(rgb_croped_dma2,rgb_croped_dma1,rgb_croped_dma3,&part);


                    // printf("a:%a\n",a);
                    // 用于保存为ppm图片，方便查看处理效果
                    // const std::string file_name = RESULTS+std::to_string(all+bird_all) + ".ppm";
                    // save_rgb_to_ppm(rgb_croped_dma2, 224, 224, file_name);
                    // save_rgb_to_ppm(rgb_dma, width_resized, height_resized, file_name);


                    dma_buf_free(rgb_size, &rgb_dma_fd, rgb_dma);


                    dma_buf_free(rgb_croped_size, &rgb_croped_dma_fd1, rgb_croped_dma1);
                    dma_buf_free(rgb_croped_size, &rgb_croped_dma_fd2, rgb_croped_dma2);
                    dma_buf_free(rgb_croped_size, &rgb_croped_dma_fd3, rgb_croped_dma3);

                    printf("current number:%d\n",all);
                    printf("predicted number:%d\n",part);
                } else {
                    std::cerr << "Filename format is incorrect: " << filename << std::endl;
                }
            }

            closedir(dir);

            if(dir_name==BIRD_PATH){
                bird_all = all;
                brid_part = part;
            }else{
                other_all = all;
                other_part = part; 
            }
            // 完成一个文件夹后置0，进行下一个文件夹推理
            all = 0;
            part = 0;
        } else {
            // 无法打开目录
            perror("opendir");
        }
    }

    for (auto& dir_name  : lst) {
        if(dir_name==BIRD_PATH){
            float bird = brid_part/(float)bird_all;
            printf("accuracy with bird:%.2f\n",bird);
        }else{
            float non_bird = (other_all-other_part)/(float)other_all;
            printf("accuracy without bird:%.2f\n",non_bird);          
        }
    }

    return 0;
}
