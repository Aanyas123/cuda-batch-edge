// CUDA kernels and launch helpers for the edge-detection pipeline.
#ifndef CUDA_BATCH_EDGE_KERNELS_CUH_
#define CUDA_BATCH_EDGE_KERNELS_CUH_

#include <cstdint>

#include <cuda_runtime.h>

namespace kernels {

constexpr int kTileSize = 16;
constexpr int kNumBins = 256;

// Uploads the 5x5 Gaussian weights to constant memory. Call once at startup.
cudaError_t InitGaussianWeights();

// Stage 1: 5x5 Gaussian blur (shared-memory tiled, clamp-to-edge).
void LaunchGaussianBlur(const uint8_t* in, uint8_t* out, int width, int height,
                        cudaStream_t stream);

// Stage 2: Sobel gradient magnitude, clamped to [0, 255].
void LaunchSobel(const uint8_t* in, uint8_t* out, int width, int height,
                 cudaStream_t stream);

// Stage 3: histogram equalization to stretch edge contrast.
// |histogram| and |lut| are device buffers of kNumBins uints / bytes.
void LaunchEqualize(const uint8_t* in, uint8_t* out, unsigned int* histogram,
                    uint8_t* lut, int width, int height, cudaStream_t stream);

}  // namespace kernels

#endif  // CUDA_BATCH_EDGE_KERNELS_CUH_
