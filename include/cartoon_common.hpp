#ifndef CARTOON_COMMON_HPP
#define CARTOON_COMMON_HPP

#include <cmath>
#include <vector>

const int CHANNELS = 3;
const int THRESHOLD = 64;

const int SOBEL3_X[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
const int SOBEL3_Y[3][3] = {{ 1, 2, 1}, { 0, 0, 0}, {-1,-2,-1}};

const int SOBEL5_X[5][5] = {
    {-1, -2, 0,  2, 1},
    {-4, -8, 0,  8, 4},
    {-6,-12, 0, 12, 6},
    {-4, -8, 0,  8, 4},
    {-1, -2, 0,  2, 1}
};
const int SOBEL5_Y[5][5] = {
    { 1,  4,  6,  4,  1},
    { 2,  8, 12,  8,  2},
    { 0,  0,  0,  0,  0},
    {-2, -8,-12, -8, -2},
    {-1, -4, -6, -4, -1}
};

inline unsigned char clamp_byte(int v) {
    return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

inline int get_pixel(const unsigned char* img, int w, int h, int x, int y, int c) {
    if (x < 0 || x >= w || y < 0 || y >= h) return 0;
    return img[(y * w + x) * CHANNELS + c];
}

// 1. Noise removal (mean blur filter)
inline unsigned char blur_pixel(const unsigned char* img, int w, int h, int x, int y, int c, int ksize) {
    int r = ksize / 2, sum = 0;
    for (int ky = -r; ky <= r; ky++)
        for (int kx = -r; kx <= r; kx++)
            sum += get_pixel(img, w, h, x + kx, y + ky, c);
    return (unsigned char)(sum / (ksize * ksize));
}

// 2 & 3. Gradient highlight (Sobel) + thresholding
inline unsigned char edge_pixel(const unsigned char* blurred, int w, int h, int x, int y, int c, int ksize) {
    int r = ksize / 2, gx = 0, gy = 0;
    for (int ky = -r; ky <= r; ky++) {
        for (int kx = -r; kx <= r; kx++) {
            int p = get_pixel(blurred, w, h, x + kx, y + ky, c);
            gx += p * (ksize == 3 ? SOBEL3_X[ky + r][kx + r] : SOBEL5_X[ky + r][kx + r]);
            gy += p * (ksize == 3 ? SOBEL3_Y[ky + r][kx + r] : SOBEL5_Y[ky + r][kx + r]);
        }
    }
    if (ksize == 5) { gx /= 12; gy /= 12; }
    int mag = (int)std::sqrt(gx * gx + gy * gy);
    return (mag >= THRESHOLD) ? 255 : 0;
}

// 4. Color quantization (posterization)
inline unsigned char posterize_pixel(unsigned char v, int levels) {
    int bin = (v * levels) / 256;
    return (unsigned char)((bin * 255) / (levels - 1));
}

#endif
