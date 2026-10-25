# Targets
#   make cpu   -> bin/fpgrowth_cpu       sequential FP-growth (any C++17 compiler)
#   make gpu   -> bin/fpgrowth_gpu       CUDA FP-growth (needs nvcc)
#   make emu   -> bin/fpgrowth_gpu_emu   the CUDA source compiled as plain C++
#                                        (serial kernel emulation, for testing
#                                        on machines without an NVIDIA GPU)
#   make       -> cpu + emu, plus gpu when nvcc is on the PATH
#
# Override e.g. CUDA_ARCH=-arch=sm_86 for a specific GPU (default: native).

CXX       ?= g++
NVCC      ?= nvcc
CXXFLAGS  ?= -O3 -std=c++17 -Wall -Wextra
CUDA_ARCH ?= -arch=native
NVCCFLAGS ?= -O3 -std=c++17 $(CUDA_ARCH)

INC    := -Iinclude
BIN    := bin
COMMON := src/common.cpp include/common.h

ifeq ($(OS),Windows_NT)
  EXE := .exe
  # Another MinGW runtime earlier on PATH (e.g. Git for Windows' mingw64/bin)
  # makes ld and the produced binaries load mismatched DLLs. Put the
  # compiler's own directory first and link the C++ runtime statically.
  CXX_DIR := $(dir $(shell command -v $(CXX) 2>/dev/null))
  ifneq ($(CXX_DIR),)
    export PATH := $(CXX_DIR):$(PATH)
  endif
  LDFLAGS ?= -static
endif

HAVE_NVCC := $(shell command -v $(NVCC) 2>/dev/null)

.PHONY: all cpu gpu emu clean
all: cpu emu $(if $(HAVE_NVCC),gpu)

cpu: $(BIN)/fpgrowth_cpu$(EXE)
emu: $(BIN)/fpgrowth_gpu_emu$(EXE)
gpu: $(BIN)/fpgrowth_gpu$(EXE)

$(BIN):
	mkdir -p $(BIN)

$(BIN)/fpgrowth_cpu$(EXE): src/fpgrowth_cpu.cpp $(COMMON) | $(BIN)
	$(CXX) $(CXXFLAGS) $(INC) src/fpgrowth_cpu.cpp src/common.cpp $(LDFLAGS) -o $@

$(BIN)/fpgrowth_gpu_emu$(EXE): src/fpgrowth_gpu_emu.cpp src/fpgrowth_gpu.cu include/cuda_compat.h $(COMMON) | $(BIN)
	$(CXX) $(CXXFLAGS) $(INC) src/fpgrowth_gpu_emu.cpp src/common.cpp $(LDFLAGS) -o $@

$(BIN)/fpgrowth_gpu$(EXE): src/fpgrowth_gpu.cu include/cuda_compat.h $(COMMON) | $(BIN)
	$(NVCC) $(NVCCFLAGS) $(INC) src/fpgrowth_gpu.cu src/common.cpp -o $@

clean:
	rm -rf $(BIN)
