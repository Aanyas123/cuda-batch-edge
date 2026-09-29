// Minimal binary PGM (P5) reader/writer. No external dependencies.
#ifndef CUDA_BATCH_EDGE_PGM_IO_H_
#define CUDA_BATCH_EDGE_PGM_IO_H_

#include <cstdint>
#include <string>
#include <vector>

namespace pgm {

struct Image {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> pixels;  // Row-major, 8-bit grayscale.
};

// Reads an 8-bit binary PGM. Returns false and fills |error| on failure.
bool Read(const std::string& path, Image* image, std::string* error);

// Writes an 8-bit binary PGM. Returns false on failure.
bool Write(const std::string& path, int width, int height,
           const uint8_t* pixels);

}  // namespace pgm

#endif  // CUDA_BATCH_EDGE_PGM_IO_H_
