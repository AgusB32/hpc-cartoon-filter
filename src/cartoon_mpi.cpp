#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "cartoon_common.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 5) {
        if (rank == 0) printf("Usage: %s <input> <output> <filter: 3|5> <poster: 3|9>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int ksize = std::atoi(argv[3]);
    int levels = std::atoi(argv[4]);
    int w = 0, h = 0, ch = 0;
    std::vector<unsigned char> full_input;

    if (rank == 0) {
        unsigned char* raw = stbi_load(argv[1], &w, &h, &ch, CHANNELS);
        if (!raw) {
            printf("Error loading image: %s\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        full_input.assign(raw, raw + w * h * CHANNELS);
        stbi_image_free(raw);
    }

    MPI_Bcast(&w, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&h, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    int halo = ksize - 1;
    int row_bytes = w * CHANNELS;

    int start_y = (rank * h) / size;
    int end_y = ((rank + 1) * h) / size;
    int local_h = end_y - start_y;

    int ext_start = std::max(0, start_y - halo);
    int ext_end = std::min(h, end_y + halo);
    int ext_h = ext_end - ext_start;
    int top_halo = start_y - ext_start;

    std::vector<unsigned char> local_in(ext_h * row_bytes);

    if (rank == 0) {
        std::copy(full_input.begin() + ext_start * row_bytes,
                  full_input.begin() + ext_end * row_bytes,
                  local_in.begin());
        for (int p = 1; p < size; p++) {
            int p_ext_s = std::max(0, (p * h) / size - halo);
            int p_ext_e = std::min(h, ((p + 1) * h) / size + halo);
            MPI_Send(full_input.data() + p_ext_s * row_bytes,
                     (p_ext_e - p_ext_s) * row_bytes, MPI_UNSIGNED_CHAR, p, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(local_in.data(), ext_h * row_bytes, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    std::vector<unsigned char> local_blurred(ext_h * row_bytes);
    std::vector<unsigned char> local_cartoon(local_h * row_bytes);

    for (int y = 0; y < ext_h; y++)
        for (int x = 0; x < w; x++)
            for (int c = 0; c < CHANNELS; c++)
                local_blurred[(y * w + x) * CHANNELS + c] = blur_pixel(local_in.data(), w, ext_h, x, y, c, ksize);

    for (int y = 0; y < local_h; y++) {
        int ly = y + top_halo;
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < CHANNELS; c++) {
                int out_idx = (y * w + x) * CHANNELS + c;
                int in_idx = (ly * w + x) * CHANNELS + c;
                unsigned char edge = edge_pixel(local_blurred.data(), w, ext_h, x, ly, c, ksize);
                unsigned char post = posterize_pixel(local_in[in_idx], levels);
                local_cartoon[out_idx] = clamp_byte(edge + post);
            }
        }
    }

    std::vector<unsigned char> final_cartoon;
    std::vector<int> counts(size), displs(size);
    if (rank == 0) {
        final_cartoon.resize(h * row_bytes);
        for (int p = 0; p < size; p++) {
            int p_s = (p * h) / size;
            int p_e = ((p + 1) * h) / size;
            counts[p] = (p_e - p_s) * row_bytes;
            displs[p] = p_s * row_bytes;
        }
    }

    MPI_Gatherv(local_cartoon.data(), local_h * row_bytes, MPI_UNSIGNED_CHAR,
                final_cartoon.data(), counts.data(), displs.data(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - start;

    if (rank == 0) {
        if (std::strcmp(argv[2], "-") != 0)
            stbi_write_jpg(argv[2], w, h, CHANNELS, final_cartoon.data(), 95);
        printf("[MPI %d procs] %dx%d | Filter %dx%d | Poster %d | Time: %.6f s\n",
               size, w, h, ksize, ksize, levels, elapsed);
    }

    MPI_Finalize();
    return 0;
}
