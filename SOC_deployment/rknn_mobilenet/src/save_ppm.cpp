#include "save_ppm.h"
#include <iostream>
#include <fstream>

int save_rgb_to_ppm( char* rgb_img, int width, int height, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return 1;
    }

    // 写入PPM头
    ofs << "P6\n" << width << " " << height << "\n255\n";

    // 写入RGB数据
    ofs.write(rgb_img, width * height * 3);

    ofs.close();
    return 0;
}
