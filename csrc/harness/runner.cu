// runner - correctness check, timing and CSV output.
// Every kernel is measured through this one path, under identical conditions.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../common/cuda_utils.h"
#include "../common/matrix.h"
#include "../common/verify.h"
#include "../kernels/kernels.h"

namespace {

struct Shape {
  int M, N, K;
};

struct Options {
  std::vector<std::string> kernels;
  std::vector<Shape> shapes;
  int reps = 100;
  int warmup = 25;
  float alpha = 1.0f;
  float beta = 0.0f;
  bool check = true;
  bool bench = true;
  std::string csv;
  std::string tag;
};

struct Timing {
  double median_ms = 0, p10_ms = 0, p90_ms = 0, min_ms = 0;
};

struct GpuState {
  int sm_clock_mhz = -1;
  int temp_c = -1;
};

// Sample SM clock and temperature via nvidia-smi. A laptop GPU's power budget
// floats, so this is sampled before and after each benchmark and written to the
// CSV: without it there is no way to tell "this version is faster" apart from
// "this version happened to run in a high-clock window".
GpuState sample_gpu_state() {
  GpuState s;
  FILE* fp = popen(
      "nvidia-smi --query-gpu=clocks.current.sm,temperature.gpu "
      "--format=csv,noheader,nounits 2>/dev/null",
      "r");
  if (!fp) return s;
  int a = -1, b = -1;
  if (std::fscanf(fp, "%d, %d", &a, &b) == 2) {
    s.sm_clock_mhz = a;
    s.temp_c = b;
  }
  pclose(fp);
  return s;
}

double percentile(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) return 0.0;
  const double pos = q * (static_cast<double>(sorted.size()) - 1.0);
  const size_t lo = static_cast<size_t>(pos);
  const size_t hi = std::min(lo + 1, sorted.size() - 1);
  const double frac = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

// Time each iteration separately rather than dividing a total by a count: the
// per-iteration distribution exposes clock drift and occasional stalls.
Timing time_kernel(SgemmFn fn, const Shape& s, float alpha, const float* dA,
                   const float* dB, float beta, float* dC, int warmup,
                   int reps) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) fn(s.M, s.N, s.K, alpha, dA, dB, beta, dC);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> times;
  times.reserve(reps);
  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    fn(s.M, s.N, s.K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    times.push_back(static_cast<double>(ms));
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::sort(times.begin(), times.end());
  Timing t;
  t.median_ms = percentile(times, 0.50);
  t.p10_ms = percentile(times, 0.10);
  t.p90_ms = percentile(times, 0.90);
  t.min_ms = times.front();
  return t;
}

double gflops_of(const Shape& s, double ms) {
  const double flop = 2.0 * s.M * s.N * s.K;
  return (ms > 0.0) ? flop / (ms * 1e-3) / 1e9 : 0.0;
}

bool parse_shape(const char* str, Shape* out) {
  int m, n, k;
  if (std::sscanf(str, "%dx%dx%d", &m, &n, &k) == 3 && m > 0 && n > 0 && k > 0) {
    *out = {m, n, k};
    return true;
  }
  return false;
}

void usage() {
  std::printf(
      "usage: runner [options]\n"
      "  --kernel <name|all>   comma separated, default all\n"
      "  --shape MxNxK         repeatable, default 4096x4096x4096\n"
      "  --preset correctness  three shapes: main / non-divisible / tiny\n"
      "  --preset sweep        square sweep, 512..8192\n"
      "  --reps N              default 100\n"
      "  --warmup N            default 25\n"
      "  --alpha F --beta F    default 1.0 / 0.0\n"
      "  --no-check            skip the correctness check\n"
      "  --no-bench            correctness only\n"
      "  --csv <path>          append to CSV (header written if new)\n"
      "  --tag <str>           free-form label, stored in the CSV\n");
}

void split_csv(const std::string& in, std::vector<std::string>* out) {
  size_t start = 0;
  while (start <= in.size()) {
    size_t comma = in.find(',', start);
    if (comma == std::string::npos) comma = in.size();
    if (comma > start) out->push_back(in.substr(start, comma - start));
    start = comma + 1;
  }
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s requires an argument\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--kernel") {
      split_csv(next("--kernel"), &o.kernels);
    } else if (a == "--shape") {
      Shape s;
      if (!parse_shape(next("--shape"), &s)) {
        std::fprintf(stderr, "--shape expects MxNxK\n");
        return 2;
      }
      o.shapes.push_back(s);
    } else if (a == "--preset") {
      std::string p = next("--preset");
      if (p == "correctness") {
        o.shapes.push_back({4096, 4096, 4096});
        o.shapes.push_back({4097, 513, 129});
        o.shapes.push_back({64, 64, 64});
      } else if (p == "sweep") {
        for (int n = 512; n <= 8192; n *= 2) o.shapes.push_back({n, n, n});
      } else {
        std::fprintf(stderr, "unknown preset: %s\n", p.c_str());
        return 2;
      }
    } else if (a == "--reps") {
      o.reps = std::atoi(next("--reps"));
    } else if (a == "--warmup") {
      o.warmup = std::atoi(next("--warmup"));
    } else if (a == "--alpha") {
      o.alpha = static_cast<float>(std::atof(next("--alpha")));
    } else if (a == "--beta") {
      o.beta = static_cast<float>(std::atof(next("--beta")));
    } else if (a == "--no-check") {
      o.check = false;
    } else if (a == "--no-bench") {
      o.bench = false;
    } else if (a == "--csv") {
      o.csv = next("--csv");
    } else if (a == "--tag") {
      o.tag = next("--tag");
    } else if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else {
      std::fprintf(stderr, "unknown option: %s\n", a.c_str());
      usage();
      return 2;
    }
  }
  if (o.shapes.empty()) o.shapes.push_back({4096, 4096, 4096});
  if (o.kernels.empty())
    for (int i = 0; i < kNumKernels; ++i) o.kernels.push_back(kKernels[i].name);
  if (o.kernels.size() == 1 && o.kernels[0] == "all") {
    o.kernels.clear();
    for (int i = 0; i < kNumKernels; ++i) o.kernels.push_back(kKernels[i].name);
  }

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::printf("GPU: %s  sm_%d%d  SMs=%d  L2=%dKB  smemPerBlock=%zuKB\n",
              prop.name, prop.major, prop.minor, prop.multiProcessorCount,
              prop.l2CacheSize / 1024, prop.sharedMemPerBlock / 1024);
  std::printf("bench: warmup=%d reps=%d (per-iteration timing, median reported)\n\n", o.warmup,
              o.reps);

  FILE* csv = nullptr;
  if (!o.csv.empty()) {
    const bool exists = [&] {
      FILE* f = std::fopen(o.csv.c_str(), "r");
      if (f) { std::fclose(f); return true; }
      return false;
    }();
    csv = std::fopen(o.csv.c_str(), "a");
    if (!csv) {
      std::fprintf(stderr, "cannot open CSV: %s\n", o.csv.c_str());
      return 1;
    }
    if (!exists)
      std::fprintf(csv,
                   "kernel,M,N,K,alpha,beta,warmup,reps,ms_median,ms_p10,"
                   "ms_p90,ms_min,gflops_median,gflops_best,pct_cublas,"
                   "max_rel,fro_rel,sm_clock_before,sm_clock_after,temp_before,"
                   "temp_after,tag\n");
  }

  int failures = 0;
  for (const Shape& s : o.shapes) {
    const size_t nA = static_cast<size_t>(s.M) * s.K;
    const size_t nB = static_cast<size_t>(s.K) * s.N;
    const size_t nC = static_cast<size_t>(s.M) * s.N;

    auto hA = make_host_matrix(s.M, s.K, 1234u);
    auto hB = make_host_matrix(s.K, s.N, 5678u);
    float* dA = device_alloc(nA);
    float* dB = device_alloc(nB);
    float* dC = device_alloc(nC);
    host_to_device(dA, hA);
    host_to_device(dB, hB);

    // Reference is always K0, on the same inputs and the same device buffers.
    CUDA_CHECK(cudaMemset(dC, 0, nC * sizeof(float)));
    sgemm_k0_cublas(s.M, s.N, s.K, o.alpha, dA, dB, o.beta, dC);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto ref = device_to_host(dC, nC);

    std::printf("=== shape %dx%dx%d ===\n", s.M, s.N, s.K);
    std::printf("%-6s %11s %9s %9s %9s %10s %10s %10s\n", "kernel",
                "ms(median)", "p10", "p90", "GFLOP/s", "%cuBLAS", "max_rel",
                "fro_rel");

    double cublas_gflops = 0.0;
    for (const std::string& kname : o.kernels) {
      const KernelEntry* ke = find_kernel(kname.c_str());
      if (!ke) {
        std::fprintf(stderr, "unknown kernel: %s\n", kname.c_str());
        failures++;
        continue;
      }

      VerifyResult vr;
      bool ok = true;
      if (o.check) {
        CUDA_CHECK(cudaMemset(dC, 0, nC * sizeof(float)));
        ke->fn(s.M, s.N, s.K, o.alpha, dA, dB, o.beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto got = device_to_host(dC, nC);
        vr = verify(ref, got);
        ok = vr.passed(1e-3, 1e-5);
        if (!ok) failures++;
      }

      Timing t;
      GpuState before, after;
      if (o.bench) {
        before = sample_gpu_state();
        t = time_kernel(ke->fn, s, o.alpha, dA, dB, o.beta, dC, o.warmup,
                        o.reps);
        after = sample_gpu_state();
      }
      const double g_med = o.bench ? gflops_of(s, t.median_ms) : 0.0;
      const double g_best = o.bench ? gflops_of(s, t.min_ms) : 0.0;
      if (kname == "k0") cublas_gflops = g_med;
      const double pct =
          (cublas_gflops > 0.0) ? 100.0 * g_med / cublas_gflops : 0.0;

      std::printf("%-6s %11.3f %9.3f %9.3f %9.1f %9.1f%% %10.2e %10.2e  %s\n",
                  ke->name, t.median_ms, t.p10_ms, t.p90_ms, g_med, pct,
                  vr.max_rel, vr.fro_rel, ok ? "" : "<< FAIL");
      if (!ok)
        std::printf("       worst @ %zu: ref=%.6f got=%.6f\n", vr.worst_index,
                    vr.worst_ref, vr.worst_got);

      if (csv)
        std::fprintf(csv,
                     "%s,%d,%d,%d,%g,%g,%d,%d,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,"
                     "%.2f,%.3e,%.3e,%d,%d,%d,%d,%s\n",
                     ke->name, s.M, s.N, s.K, o.alpha, o.beta, o.warmup, o.reps,
                     t.median_ms, t.p10_ms, t.p90_ms, t.min_ms, g_med, g_best,
                     pct, vr.max_rel, vr.fro_rel, before.sm_clock_mhz,
                     after.sm_clock_mhz, before.temp_c, after.temp_c,
                     o.tag.c_str());
    }
    std::printf("\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
  }

  if (csv) std::fclose(csv);
  if (failures) {
    std::printf("!! %d check(s) failed\n", failures);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
