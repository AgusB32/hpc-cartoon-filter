#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "cartoon_common.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

int main(int argc, char* argv[]) {
    if (argc < 6) {
        printf("Usage: %s <input> <output> <filter: 3|5> <poster: 3|9> <threads>\n", argv[0]);
        return 1;
    }

    int ksize = std::atoi(argv[3]);
    int levels = std::atoi(argv[4]);
    int threads = std::atoi(argv[5]);
    omp_set_num_threads(threads);

    int w, h, ch;
    unsigned char* raw = stbi_load(argv[1], &w, &h, &ch, CHANNELS);
    if (!raw) {
        printf("Error loading image: %s\n", argv[1]);
        return 1;
    }

    int total_bytes = w * h * CHANNELS;
    std::vector<unsigned char> input(raw, raw + total_bytes);
    stbi_image_free(raw);

    std::vector<unsigned char> blurred(total_bytes);
    std::vector<unsigned char> cartoon(total_bytes);

    double start = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (int c = 0; c < CHANNELS; c++)
                blurred[(y * w + x) * CHANNELS + c] = blur_pixel(input.data(), w, h, x, y, c, ksize);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < CHANNELS; c++) {
                int idx = (y * w + x) * CHANNELS + c;
                unsigned char edge = edge_pixel(blurred.data(), w, h, x, y, c, ksize);
                unsigned char post = posterize_pixel(input[idx], levels);
                cartoon[idx] = clamp_byte(edge + post);
            }
        }
    }

    double elapsed = omp_get_wtime() - start;

    if (std::strcmp(argv[2], "-") != 0)
        stbi_write_jpg(argv[2], w, h, CHANNELS, cartoon.data(), 95);

    printf("[OMP %d threads] %dx%d | Filter %dx%d | Poster %d | Time: %.6f s\n",
           threads, w, h, ksize, ksize, levels, elapsed);
    return 0;
}
