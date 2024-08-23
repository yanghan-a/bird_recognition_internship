#ifndef SAVE_PPM_H
#define SAVE_PPM_H

#include <string>
#include <cstdint>

int save_rgb_to_ppm(char* rgb_img, int width, int height, const std::string& filename);

#endif // SAVE_PPM_H
