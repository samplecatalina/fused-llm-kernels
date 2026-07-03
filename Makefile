# fused-llm-kernels
# Reproduce every number with: make test && make bench
#
# Target selection:
#   ARCH    compute capability to build for     (default sm_89)
#   DEVICE  device tag; results and reports are  (default rtx4060-laptop)
#           kept in one directory per device
# e.g. make bench ARCH=sm_90 DEVICE=<tag>

SHELL     := /bin/bash
NVCC      ?= nvcc
ARCH      ?= sm_89
DEVICE    ?= rtx4060-laptop
# One build directory per ARCH, so switching targets never links stale objects.
BUILD     := build/$(ARCH)
# -lineinfo lets ncu map metrics back to source lines; the profiling evidence
# chain depends on it.
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -Wall -Xcompiler -pthread
LDLIBS    := -lcublas -lpthread

KERNEL_SRCS := $(wildcard csrc/kernels/*.cu)
SRCS        := csrc/harness/runner.cu $(KERNEL_SRCS)
OBJS        := $(patsubst %.cu,$(BUILD)/%.o,$(SRCS))
HEADERS     := $(wildcard csrc/*/*.h)
BIN         := $(BUILD)/runner

RESULTS    := results/$(DEVICE)
REPORTS    := profiling/reports/$(DEVICE)

NCU        ?= ncu
NCU_SECS   := --section SpeedOfLight --section MemoryWorkloadAnalysis \
              --section Occupancy --section SchedulerStats \
              --section WarpStateStats --section LaunchStats
K          ?= k1
KERNELS    ?= all
SHAPE      ?= 4096x4096x4096
REPS       ?= 100
# Minimum warmup; the runner keeps going until the SM clock settles.
WARMUP_S   ?= 30
BENCH_FLAGS = --device-tag $(DEVICE) --log-clocks --warmup-seconds $(WARMUP_S) \
              --reps $(REPS)

.PHONY: all build test bench sweep profile format clean

all: build

build: $(BIN)

$(BIN): $(OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/%.o: %.cu $(HEADERS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

# Correctness over three shapes: main / non-divisible / tiny
test: build
	$(BIN) --preset correctness --no-bench

# Headline number. One rung only: make bench KERNELS=k0
bench: build
	@mkdir -p $(RESULTS)
	$(BIN) --kernel $(KERNELS) --shape $(SHAPE) $(BENCH_FLAGS) \
	  --csv $(RESULTS)/gemm_4096.csv

# 512..8192 sweep: shows launch overhead at the small end and L2 effects
sweep: build
	@mkdir -p $(RESULTS)
	$(BIN) --kernel $(KERNELS) --preset sweep $(BENCH_FLAGS) \
	  --csv $(RESULTS)/gemm_sweep.csv

# Collect an ncu report for one rung: make profile K=k1
# Some managed GPUs deny counter access to users (ERR_NVGPUCTRPERM); that is a
# host policy, not a build problem, and is reported as such.
profile: build
	@mkdir -p $(REPORTS)
	@set -o pipefail; log=$$(mktemp); \
	out=$(REPORTS)/$(K)_$$(date +%Y%m%d_%H%M%S); \
	$(NCU) $(NCU_SECS) -k regex:$(K) --target-processes all \
	  -o $$out --force-overwrite \
	  $(BIN) --kernel $(K) --shape $(SHAPE) --reps 1 --warmup-seconds 0 \
	  --warmup 0 --no-check 2>&1 | tee $$log; rc=$$?; \
	if grep -q ERR_NVGPUCTRPERM $$log; then \
	  echo; \
	  echo "profile: GPU performance counters are not accessible to this user on"; \
	  echo "this device (ERR_NVGPUCTRPERM). Collect counters on a device where they"; \
	  echo "are enabled; timing numbers from 'make bench' are unaffected."; \
	  rc=1; \
	fi; \
	rm -f $$log; exit $$rc

format:
	clang-format -i $(shell find csrc -name '*.cu' -o -name '*.h')

clean:
	rm -rf build
