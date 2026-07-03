# fused-llm-kernels
# Reproduce every number with: make test && make bench

NVCC      ?= nvcc
ARCH      ?= sm_89
BUILD     := build
# -lineinfo lets ncu map metrics back to source lines; the profiling evidence
# chain depends on it.
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -Wall
LDLIBS    := -lcublas

KERNEL_SRCS := $(wildcard csrc/kernels/*.cu)
SRCS        := csrc/harness/runner.cu $(KERNEL_SRCS)
OBJS        := $(patsubst %.cu,$(BUILD)/%.o,$(SRCS))
BIN         := $(BUILD)/runner

NCU        ?= ncu
NCU_SECS   := --section SpeedOfLight --section MemoryWorkloadAnalysis \
              --section Occupancy --section SchedulerStats \
              --section WarpStateStats --section LaunchStats
K          ?= k1
SHAPE      ?= 4096x4096x4096

.PHONY: all build test bench sweep profile format clean

all: build

build: $(BIN)

$(BIN): $(OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

# Correctness over three shapes: main / non-divisible / tiny
test: build
	$(BIN) --preset correctness --no-bench

# Headline number
bench: build
	@mkdir -p results
	$(BIN) --shape $(SHAPE) --csv results/gemm_4096.csv

# 512..8192 sweep: shows launch overhead at the small end and L2 effects
sweep: build
	@mkdir -p results
	$(BIN) --preset sweep --csv results/gemm_sweep.csv --reps 50

# Collect an ncu report for one rung: make profile K=k1
profile: build
	@mkdir -p profiling/reports
	$(NCU) $(NCU_SECS) -k regex:$(K) --target-processes all \
	  -o profiling/reports/$(K)_$(shell date +%Y%m%d_%H%M%S) --force-overwrite \
	  $(BIN) --kernel $(K) --shape $(SHAPE) --reps 1 --warmup 0 --no-check

format:
	clang-format -i $(shell find csrc -name '*.cu' -o -name '*.h')

clean:
	rm -rf $(BUILD)
