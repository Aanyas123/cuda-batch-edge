# CUDA Batch Edge Detection

GPU pipeline that processes a **large batch of images in one execution**
(default: 200 images of 512x512 = 52 MPixels): 5x5 Gaussian blur -> Sobel edge
magnitude -> histogram equalization. Custom CUDA kernels, no external GPU
library required.

## Requirements
- NVIDIA GPU, CUDA Toolkit (`nvcc`), GNU make
- Python 3 with `numpy` and `Pillow` (only for data generation / montage)

## Quick start
```bash
./run.sh sm_75 200      # args: GPU arch (default sm_75), number of images (default 200)
```
`run.sh` builds the binary, generates the dataset if missing, runs the
pipeline twice (4 streams vs. 1 stream), and writes proof of execution to
`artifacts/` (`execution_log.txt`, `timings.csv`, `montage.png`).
Find your arch with `nvidia-smi --query-gpu=compute_cap --format=csv`
(e.g. 8.6 -> `sm_86`).

## Manual usage
```bash
make ARCH=sm_75
python3 scripts/generate_data.py --out data/input --count 200 --size 512
./bin/cuda_batch_edge --input_dir data/input --output_dir data/output \
    --streams 4 --log artifacts/timings.csv
```

### CLI arguments
| Flag | Description |
|---|---|
| `--input_dir DIR` | Directory of 8-bit binary `.pgm` images (required) |
| `--output_dir DIR` | Output directory for `edges_*.pgm` (required) |
| `--streams N` | CUDA streams for copy/compute overlap (1-32, default 4) |
| `--no_equalize` | Skip histogram equalization |
| `--log FILE` | Per-image CSV timings |
| `--help` | Usage |

### Using your own images (e.g. USC-SIPI)
```bash
python3 scripts/convert_to_pgm.py path/to/pngs data/input
```

## Repository layout
```
src/main.cu        CLI, batching, multi-stream pipeline, timing
src/kernels.cu/.cuh  CUDA kernels + launch wrappers
src/pgm_io.cc/.h   Dependency-free PGM reader/writer
scripts/           Data generation, conversion, montage
run.sh, Makefile   Build + run + proof-of-execution capture
artifacts/         Logs, CSV timings, montage (committed after running)
data/output/       Result images (commit these before leaving the lab)
```

## Algorithms / kernels
1. **GaussianBlurKernel** - 16x16 blocks load a 20x20 tile (with halo) into shared
   memory; 5x5 binomial weights live in constant memory. Clamp-to-edge borders.
2. **SobelKernel** - 3x3 gradient magnitude `sqrt(gx^2+gy^2)`, clamped to 255.
3. **HistogramKernel** - per-block shared-memory histogram merged into global
   memory with atomics (far fewer global atomics than one atomic per pixel).
4. **BuildLutKernel** - one block does a Hillis-Steele parallel scan of the
   histogram to get the CDF and builds the equalization lookup table on the GPU
   (no host round trip).
5. **ApplyLutKernel** - grid-stride kernel applying the LUT from shared memory.

Each image runs in its own CUDA stream with pinned host buffers, so uploads,
kernels, and downloads of different images overlap.

## Proof of execution
All produced by a single `./run.sh` in the Coursera lab and committed:
- `artifacts/execution_log.txt` - `nvidia-smi` output, the program's console
  output (`Found 200 images ... Processed 200 images`), and the 1-stream vs.
  4-stream comparison.
- `artifacts/timings.csv` - one row per image (200 rows) with its GPU time.
- `artifacts/montage.png` - 5 inputs (top) vs. their GPU edge maps (bottom).
- `data/output/edges_*.pgm` - all 200 result images.

## Results
| Config | Wall time | Throughput | Mean GPU ms/image |
|---|---|---|---|
| 4 streams | _fill from log_ | _fill_ | _fill_ |
| 1 stream  | _fill from log_ | _fill_ | _fill_ |

## Lessons learned
- **Halo handling:** the tiled blur must load a 2-pixel border around each
  16x16 tile; clamping coordinates during the cooperative load avoids
  out-of-bounds reads and gives clamp-to-edge borders for free.
- **Atomic contention:** one global atomic per pixel serializes badly on
  low-entropy edge images (most pixels are 0). Per-block shared histograms cut
  global atomics to at most 256 per block.
- **Stay on the device:** building the equalization LUT with an on-GPU scan
  removes a device->host->device round trip per image, which is what allows
  images to be fully pipelined across streams.
- **Streams are not free speed:** overlap only helps when copies and kernels
  are comparable in cost; for small images the CPU-side file read/write
  becomes the bottleneck, which the 1- vs. 4-stream comparison shows.

## Limitations
Only 8-bit binary PGM (P5) input; grayscale only. File I/O is single-threaded
on the host.
