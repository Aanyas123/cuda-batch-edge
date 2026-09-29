# Build with: make            (set ARCH for your GPU, e.g. make ARCH=sm_75)
NVCC  ?= nvcc
ARCH  ?= sm_75
FLAGS := -O2 -std=c++14 -arch=$(ARCH) -Xcompiler -Wall

TARGET := bin/cuda_batch_edge
SRCS   := src/main.cu src/kernels.cu src/pgm_io.cc

all: $(TARGET)

$(TARGET): $(SRCS) src/kernels.cuh src/pgm_io.h
	@mkdir -p bin
	$(NVCC) $(FLAGS) -o $@ $(SRCS)

run: $(TARGET)
	./run.sh $(ARCH)

clean:
	rm -rf bin data/output/*.pgm

.PHONY: all run clean
