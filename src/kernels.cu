#include "kernels.cuh"

namespace kernels {
namespace {

constexpr int kRadius = 2;  // 5x5 Gaussian.
constexpr int kHalo = kTileSize + 2 * kRadius;

__constant__ float kGaussian[5][5];

__device__ __forceinline__ int Clamp(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void GaussianBlurKernel(const uint8_t* in, uint8_t* out, int width,
                                   int height) {
  __shared__ uint8_t tile[kHalo][kHalo];
  const int x0 = blockIdx.x * kTileSize;
  const int y0 = blockIdx.y * kTileSize;

  // Cooperative load of the tile plus halo.
  for (int i = threadIdx.y * kTileSize + threadIdx.x; i < kHalo * kHalo;
       i += kTileSize * kTileSize) {
    const int ty = i / kHalo;
    const int tx = i % kHalo;
    const int gx = Clamp(x0 + tx - kRadius, 0, width - 1);
    const int gy = Clamp(y0 + ty - kRadius, 0, height - 1);
    tile[ty][tx] = in[gy * width + gx];
  }
  __syncthreads();

  const int x = x0 + threadIdx.x;
  const int y = y0 + threadIdx.y;
  if (x >= width || y >= height) return;

  float sum = 0.0f;
#pragma unroll
  for (int dy = 0; dy < 5; ++dy) {
#pragma unroll
    for (int dx = 0; dx < 5; ++dx) {
      sum += kGaussian[dy][dx] * tile[threadIdx.y + dy][threadIdx.x + dx];
    }
  }
  out[y * width + x] = static_cast<uint8_t>(fminf(sum + 0.5f, 255.0f));
}

__device__ __forceinline__ int Pixel(const uint8_t* in, int x, int y,
                                     int width, int height) {
  return in[Clamp(y, 0, height - 1) * width + Clamp(x, 0, width - 1)];
}

__global__ void SobelKernel(const uint8_t* in, uint8_t* out, int width,
                            int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int tl = Pixel(in, x - 1, y - 1, width, height);
  const int tc = Pixel(in, x, y - 1, width, height);
  const int tr = Pixel(in, x + 1, y - 1, width, height);
  const int ml = Pixel(in, x - 1, y, width, height);
  const int mr = Pixel(in, x + 1, y, width, height);
  const int bl = Pixel(in, x - 1, y + 1, width, height);
  const int bc = Pixel(in, x, y + 1, width, height);
  const int br = Pixel(in, x + 1, y + 1, width, height);

  const int gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
  const int gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
  const float mag = sqrtf(static_cast<float>(gx * gx + gy * gy));
  out[y * width + x] = static_cast<uint8_t>(fminf(mag, 255.0f));
}

// Per-block shared-memory histogram, merged into global with atomics.
__global__ void HistogramKernel(const uint8_t* in, unsigned int* histogram,
                                int num_pixels) {
  __shared__ unsigned int local[kNumBins];
  for (int i = threadIdx.x; i < kNumBins; i += blockDim.x) local[i] = 0;
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pixels;
       i += gridDim.x * blockDim.x) {
    atomicAdd(&local[in[i]], 1u);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < kNumBins; i += blockDim.x) {
    if (local[i] > 0) atomicAdd(&histogram[i], local[i]);
  }
}

// Builds the equalization lookup table from the CDF. One block, 256 threads.
__global__ void BuildLutKernel(const unsigned int* histogram, uint8_t* lut,
                               int num_pixels) {
  __shared__ unsigned int cdf[kNumBins];
  __shared__ unsigned int cdf_min;
  const int t = threadIdx.x;
  cdf[t] = histogram[t];
  __syncthreads();

  // Hillis-Steele inclusive scan.
  for (int offset = 1; offset < kNumBins; offset <<= 1) {
    const unsigned int add = (t >= offset) ? cdf[t - offset] : 0u;
    __syncthreads();
    cdf[t] += add;
    __syncthreads();
  }

  // Smallest non-zero CDF value.
  if (t == 0) {
    cdf_min = 0;
    for (int i = 0; i < kNumBins; ++i) {
      if (cdf[i] > 0) {
        cdf_min = cdf[i];
        break;
      }
    }
  }
  __syncthreads();

  const unsigned int denom = static_cast<unsigned int>(num_pixels) - cdf_min;
  if (denom == 0) {
    lut[t] = static_cast<uint8_t>(t);  // Constant image: identity mapping.
  } else {
    const float v = (static_cast<float>(cdf[t]) - static_cast<float>(cdf_min)) /
                    static_cast<float>(denom) * 255.0f;
    lut[t] = static_cast<uint8_t>(fminf(fmaxf(v + 0.5f, 0.0f), 255.0f));
  }
}

__global__ void ApplyLutKernel(const uint8_t* in, uint8_t* out,
                               const uint8_t* lut, int num_pixels) {
  __shared__ uint8_t local_lut[kNumBins];
  if (threadIdx.x < kNumBins) local_lut[threadIdx.x] = lut[threadIdx.x];
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pixels;
       i += gridDim.x * blockDim.x) {
    out[i] = local_lut[in[i]];
  }
}

}  // namespace

cudaError_t InitGaussianWeights() {
  // Binomial approximation of a Gaussian: [1 4 6 4 1] outer product / 256.
  const float k1d[5] = {1.f, 4.f, 6.f, 4.f, 1.f};
  float w[5][5];
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) w[i][j] = k1d[i] * k1d[j] / 256.0f;
  }
  return cudaMemcpyToSymbol(kGaussian, w, sizeof(w));
}

void LaunchGaussianBlur(const uint8_t* in, uint8_t* out, int width, int height,
                        cudaStream_t stream) {
  const dim3 block(kTileSize, kTileSize);
  const dim3 grid((width + kTileSize - 1) / kTileSize,
                  (height + kTileSize - 1) / kTileSize);
  GaussianBlurKernel<<<grid, block, 0, stream>>>(in, out, width, height);
}

void LaunchSobel(const uint8_t* in, uint8_t* out, int width, int height,
                 cudaStream_t stream) {
  const dim3 block(kTileSize, kTileSize);
  const dim3 grid((width + kTileSize - 1) / kTileSize,
                  (height + kTileSize - 1) / kTileSize);
  SobelKernel<<<grid, block, 0, stream>>>(in, out, width, height);
}

void LaunchEqualize(const uint8_t* in, uint8_t* out, unsigned int* histogram,
                    uint8_t* lut, int width, int height, cudaStream_t stream) {
  const int n = width * height;
  const int threads = 256;
  int blocks = (n + threads - 1) / threads;
  if (blocks > 128) blocks = 128;
  cudaMemsetAsync(histogram, 0, kNumBins * sizeof(unsigned int), stream);
  HistogramKernel<<<blocks, threads, 0, stream>>>(in, histogram, n);
  BuildLutKernel<<<1, kNumBins, 0, stream>>>(histogram, lut, n);
  ApplyLutKernel<<<blocks, threads, 0, stream>>>(in, out, lut, n);
}

}  // namespace kernels
