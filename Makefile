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
ROOFLINE    := $(BUILD)/roofline

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
# Minimum enforced GPU power limit (W) a benchmark may run under; 0 disables
# the check. The mobile 4060 loses most of its power budget when the host
# leaves its high-performance power plan or runs on a weaker adapter.
MIN_POWER_LIMIT_W ?= $(if $(filter rtx4060-laptop,$(DEVICE)),75,0)
# cuBLAS baseline at 4096^3 for this device (median of independent runs) and
# its run-to-run band, in percent. A run whose k0 falls outside the band is
# flagged in every row it writes.
K0_BASELINE ?= $(if $(filter rtx4060-laptop,$(DEVICE)),9028.0,0)
K0_BAND_PCT ?= $(if $(filter rtx4060-laptop,$(DEVICE)),2.2,0)
BENCH_FLAGS = --device-tag $(DEVICE) --log-clocks --warmup-seconds $(WARMUP_S) \
              --reps $(REPS) --min-power-limit $(MIN_POWER_LIMIT_W) \
              --k0-baseline $(K0_BASELINE) --k0-band $(K0_BAND_PCT)

.PHONY: all build test test-verify bench sweep tune roofline profile format clean

all: build

build: $(BIN) $(ROOFLINE)

$(BIN): $(OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(ROOFLINE): $(BUILD)/csrc/roofline/roofline.o
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/%.o: %.cu $(HEADERS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

# Correctness over three shapes: main / non-divisible / tiny
test: build
	$(BIN) --preset correctness --no-bench

# Margin test for the correctness tolerance (host only, no GPU)
test-verify:
	@mkdir -p $(BUILD)
	$(CXX) -O2 -std=c++17 csrc/tests/verify_margin.cpp -o $(BUILD)/verify_margin
	$(BUILD)/verify_margin

# Headline number. One rung only: make bench KERNELS=k0
bench: build
	@mkdir -p $(RESULTS)
	$(BIN) --kernel $(KERNELS) --shape $(SHAPE) $(BENCH_FLAGS) \
	  --csv $(RESULTS)/gemm_4096.csv

# Parameter search over the tiling configurations of one rung
TUNE_KERNELS ?= k7,k7c1,k7c2,k7c3,k7c4,k7c5,k7c6,k7c7,k7c8
tune: build
	@mkdir -p $(RESULTS)
	$(BIN) --kernel k0,$(TUNE_KERNELS) --shape $(SHAPE) $(BENCH_FLAGS) \
	  --csv $(RESULTS)/tuning_k7.csv

# Measured roofline ceilings: memory bandwidth and sustained FLOP rate
roofline: build
	@mkdir -p $(RESULTS)
	$(ROOFLINE) --device-tag $(DEVICE) --warmup-seconds $(WARMUP_S) \
	  --reps $(REPS) --min-power-limit $(MIN_POWER_LIMIT_W) \
	  --csv $(RESULTS)/roofline.csv

# 512..8192 sweep: shows launch overhead at the small end and L2 effects.
# Defaults to cuBLAS, the untiled coalesced rung (whose working set is all of
# A and B) and the best rung; the slow early rungs would take hours at 8192.
SWEEP_KERNELS ?= k0,k2,k7
sweep: build
	@mkdir -p $(RESULTS)
	$(BIN) --kernel $(SWEEP_KERNELS) --preset sweep $(BENCH_FLAGS) \
	  --csv $(RESULTS)/gemm_sweep.csv

# Collect an ncu report for one rung: make profile K=k1
# The binary .ncu-rep stays local (tens of MB); its details page is exported
# as CSV next to it and that text export is what gets committed.
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
	if [[ -f $$out.ncu-rep ]]; then \
	  $(NCU) -i $$out.ncu-rep --csv --page details > $$out.details.csv && \
	  echo "profile: details exported to $$out.details.csv"; \
	fi; \
	rm -f $$log; exit $$rc

format:
	clang-format -i $(shell find csrc -name '*.cu' -o -name '*.h')

clean:
	rm -rf build
