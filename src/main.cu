// Batch GPU edge detection: Gaussian blur -> Sobel -> histogram equalization.
//
// Processes every .pgm file in --input_dir using several CUDA streams so that
// host<->device copies of one image overlap with kernels of another.

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "kernels.cuh"
#include "pgm_io.h"

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err_ = (call);                                             \
    if (err_ != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                   cudaGetErrorString(err_), __FILE__, __LINE__);          \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

namespace {

struct Options {
  std::string input_dir;
  std::string output_dir;
  std::string log_path;
  int num_streams = 4;
  bool equalize = true;
  bool help = false;
};

void PrintUsage(const char* prog) {
  std::printf(
      "Usage: %s --input_dir DIR --output_dir DIR [options]\n"
      "  --input_dir DIR    Directory of 8-bit binary .pgm images (required)\n"
      "  --output_dir DIR   Where edge maps are written (required)\n"
      "  --streams N        Number of CUDA streams (default 4)\n"
      "  --no_equalize      Skip the histogram-equalization stage\n"
      "  --log FILE         Write per-image CSV timings to FILE\n"
      "  --help             Show this message\n",
      prog);
}

bool ParseArgs(int argc, char** argv, Options* opts) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](std::string* dst) {
      if (i + 1 >= argc) return false;
      *dst = argv[++i];
      return true;
    };
    std::string value;
    if (arg == "--input_dir") {
      if (!next(&opts->input_dir)) return false;
    } else if (arg == "--output_dir") {
      if (!next(&opts->output_dir)) return false;
    } else if (arg == "--log") {
      if (!next(&opts->log_path)) return false;
    } else if (arg == "--streams") {
      if (!next(&value)) return false;
      opts->num_streams = std::atoi(value.c_str());
    } else if (arg == "--no_equalize") {
      opts->equalize = false;
    } else if (arg == "--help" || arg == "-h") {
      opts->help = true;
      return true;
    } else {
      return false;
    }
  }
  return !opts->input_dir.empty() && !opts->output_dir.empty() &&
         opts->num_streams >= 1 && opts->num_streams <= 32;
}

std::vector<std::string> ListPgmFiles(const std::string& dir) {
  std::vector<std::string> files;
  DIR* d = opendir(dir.c_str());
  if (d == nullptr) return files;
  while (dirent* entry = readdir(d)) {
    const std::string name = entry->d_name;
    if (name.size() > 4 && name.compare(name.size() - 4, 4, ".pgm") == 0) {
      files.push_back(name);
    }
  }
  closedir(d);
  std::sort(files.begin(), files.end());
  return files;
}

// Per-stream working buffers, grown on demand.
struct Slot {
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr, stop = nullptr;
  size_t capacity = 0;
  uint8_t *h_in = nullptr, *h_out = nullptr;         // Pinned host memory.
  uint8_t *d_a = nullptr, *d_b = nullptr;            // Device ping-pong.
  unsigned int* d_hist = nullptr;
  uint8_t* d_lut = nullptr;
  int index = -1;  // File index currently in flight (-1 = idle).
  int width = 0, height = 0;

  void Init() {
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaMalloc(&d_hist, kernels::kNumBins * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_lut, kernels::kNumBins));
  }
  void Reserve(size_t bytes) {
    if (bytes <= capacity) return;
    Release();
    CUDA_CHECK(cudaMallocHost(&h_in, bytes));
    CUDA_CHECK(cudaMallocHost(&h_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    capacity = bytes;
  }
  void Release() {
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    cudaFree(d_a);
    cudaFree(d_b);
    h_in = h_out = d_a = d_b = nullptr;
    capacity = 0;
  }
  void Destroy() {
    Release();
    cudaFree(d_hist);
    cudaFree(d_lut);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
  }
};

struct Stats {
  int processed = 0;
  int failed = 0;
  double total_gpu_ms = 0.0;
  double total_megapixels = 0.0;
};

// Waits for a slot's work, writes its result, and logs timings.
void Finish(Slot* slot, const std::vector<std::string>& files,
            const Options& opts, std::ofstream* log, Stats* stats) {
  if (slot->index < 0) return;
  CUDA_CHECK(cudaStreamSynchronize(slot->stream));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, slot->start, slot->stop));
  const std::string& name = files[slot->index];
  const std::string out_path = opts.output_dir + "/edges_" + name;
  if (!pgm::Write(out_path, slot->width, slot->height, slot->h_out)) {
    std::fprintf(stderr, "Failed to write %s\n", out_path.c_str());
    ++stats->failed;
  } else {
    ++stats->processed;
    stats->total_gpu_ms += ms;
    stats->total_megapixels += slot->width * slot->height / 1.0e6;
    if (log != nullptr) {
      *log << name << "," << slot->width << "," << slot->height << "," << ms
           << "\n";
    }
  }
  slot->index = -1;
}

}  // namespace

int main(int argc, char** argv) {
  Options opts;
  const bool args_ok = ParseArgs(argc, argv, &opts);
  if (opts.help) {
    PrintUsage(argv[0]);
    return 0;
  }
  if (!args_ok) {
    PrintUsage(argv[0]);
    return 1;
  }

  const std::vector<std::string> files = ListPgmFiles(opts.input_dir);
  if (files.empty()) {
    std::fprintf(stderr, "No .pgm files found in %s\n",
                 opts.input_dir.c_str());
    return 1;
  }
  if (mkdir(opts.output_dir.c_str(), 0755) != 0 && errno != EEXIST) {
    std::fprintf(stderr, "Cannot create %s\n", opts.output_dir.c_str());
    return 1;
  }

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s (compute %d.%d, %d SMs)\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount);
  std::printf("Found %zu images in %s; using %d streams; equalize=%s\n",
              files.size(), opts.input_dir.c_str(), opts.num_streams,
              opts.equalize ? "on" : "off");

  CUDA_CHECK(kernels::InitGaussianWeights());

  std::ofstream log;
  if (!opts.log_path.empty()) {
    log.open(opts.log_path);
    log << "file,width,height,gpu_ms\n";
  }

  std::vector<Slot> slots(opts.num_streams);
  for (Slot& s : slots) s.Init();

  Stats stats;
  const auto wall_start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < files.size(); ++i) {
    Slot& slot = slots[i % slots.size()];
    Finish(&slot, files, opts, log.is_open() ? &log : nullptr, &stats);

    pgm::Image image;
    std::string error;
    if (!pgm::Read(opts.input_dir + "/" + files[i], &image, &error)) {
      std::fprintf(stderr, "Skipping: %s\n", error.c_str());
      ++stats.failed;
      continue;
    }
    const size_t bytes = image.pixels.size();
    slot.Reserve(bytes);
    slot.width = image.width;
    slot.height = image.height;
    slot.index = static_cast<int>(i);
    std::memcpy(slot.h_in, image.pixels.data(), bytes);

    CUDA_CHECK(cudaEventRecord(slot.start, slot.stream));
    CUDA_CHECK(cudaMemcpyAsync(slot.d_a, slot.h_in, bytes,
                               cudaMemcpyHostToDevice, slot.stream));
    kernels::LaunchGaussianBlur(slot.d_a, slot.d_b, image.width, image.height,
                                slot.stream);
    kernels::LaunchSobel(slot.d_b, slot.d_a, image.width, image.height,
                         slot.stream);
    uint8_t* result = slot.d_a;
    if (opts.equalize) {
      kernels::LaunchEqualize(slot.d_a, slot.d_b, slot.d_hist, slot.d_lut,
                              image.width, image.height, slot.stream);
      result = slot.d_b;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(slot.h_out, result, bytes,
                               cudaMemcpyDeviceToHost, slot.stream));
    CUDA_CHECK(cudaEventRecord(slot.stop, slot.stream));
  }
  for (Slot& s : slots) {
    Finish(&s, files, opts, log.is_open() ? &log : nullptr, &stats);
  }

  const double wall_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - wall_start)
                            .count();
  std::printf("Processed %d images (%d failed) in %.3f s wall-clock\n",
              stats.processed, stats.failed, wall_s);
  std::printf("Total: %.1f MPixels, throughput %.1f MPixels/s, "
              "mean GPU time/image %.3f ms\n",
              stats.total_megapixels, stats.total_megapixels / wall_s,
              stats.processed ? stats.total_gpu_ms / stats.processed : 0.0);

  for (Slot& s : slots) s.Destroy();
  return stats.failed == 0 ? 0 : 2;
}
