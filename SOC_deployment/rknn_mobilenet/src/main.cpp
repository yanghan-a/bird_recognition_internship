#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <regex>
#include <dirent.h>
#include <sys/types.h>
#include <sstream>

#include "read_yuv.h"
#include "yuv_to_rgb.h"
#include "save_ppm.h"
#include "inference.h"

#include <list>
#include <chrono>	// record time
#include <sys/resource.h>// record consumed resource


#define RESULTS "./results/"
// #define BIRD_PATH "./pictures/bird8_9yuv/"
// #define OTHER_PATH "./pictures/other8_9yuv/"
// #define BIRD_PATH "./pictures/bird_yuv/"
// #define OTHER_PATH "./pictures/other_yuv/"
// #define BIRD_PATH "./pictures/test8_15/convert_resize/bird_yuv/"
// #define OTHER_PATH "./pictures/test8_15/convert_resize/other_yuv/"
#define BIRD_PATH "./pictures/test8_15/convert/bird_yuv/"
#define OTHER_PATH "./pictures/test8_15/convert/other_yuv/"

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
                    all++;
                    int width = std::stoi(matches[1]);
                    int height = std::stoi(matches[2]);
                    // 分配内存以存储YUV数据
                    char* yuv_data;
                    yuv_data = (char *)malloc(width * height*3/2);
                    if (yuv_data == NULL) {
                        perror("malloc failed");
                        return 1;
                    }
                    // 读取YUV文件
                    std::string filepath = dir_name + filename;
                    readYUV420P(filepath, width, height, yuv_data);

                    int width_resized;
                    int height_resized;
                    // 这块是让缩放后的yuv边长2对齐
                    if (width >= height){
                        width_resized = width*224/height;
                        if(width_resized%2!=0){
                            width_resized -=1;
                        }
                        height_resized = 224;
                    }else{
                        width_resized = 224;
                        height_resized = height*224/width;
                        if(height_resized%2!=0){
                            height_resized -=1;
                        }
                    }

                    char* yuv_resized_data;
                    yuv_resized_data = (char *)malloc(width_resized * height_resized*3/2);
                    
                    // resize YUV文件
                    if (resizeYUV420P(yuv_data, yuv_resized_data, width,height,width_resized,height_resized) != 0) {
                        std::cerr << "Failed to resize YUV file" << std::endl;
                        return 1;
                    }
                    free(yuv_data);

                    char * rgb_data;
                    rgb_data = (char *)malloc(width_resized * height_resized*3);

                    // 转换为RGB
                    YUV420PToRGB(yuv_resized_data, rgb_data, width_resized, height_resized);

                    free(yuv_resized_data);
                    int final_pixel = 224;
                    char * rgb_resized_data1;
                    rgb_resized_data1 = (char *)malloc(final_pixel * final_pixel*3);

                    char * rgb_resized_data2;
                    rgb_resized_data2 = (char *)malloc(final_pixel * final_pixel*3);

                    char * rgb_resized_data3;
                    rgb_resized_data3 = (char *)malloc(final_pixel * final_pixel*3);

                    // 裁剪为224*224
                    cropRGBImageLeftMiddleRight(rgb_data,width_resized,height_resized,  final_pixel  ,rgb_resized_data1,rgb_resized_data2,rgb_resized_data3);

                    free(rgb_data);


                    // inference(224,224,rgb_resized_data2,&a);
                    inference3(rgb_resized_data2, rgb_resized_data1, rgb_resized_data3, &part);


                    // printf("a:%a\n",a);
                    // 用于保存为ppm图片，方便查看处理效果
                    //  const std::string file_name = RESULTS+std::to_string(all+bird_all) + ".ppm";
                    // save_rgb_to_ppm(rgb_resized_data2, final_pixel, final_pixel, file_name);

                    free(rgb_resized_data1);
                    free(rgb_resized_data2);
                    free(rgb_resized_data3);
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
