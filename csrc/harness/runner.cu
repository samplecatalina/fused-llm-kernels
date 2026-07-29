// runner - correctness check, timing and CSV output.
// Every kernel is measured through this one path, under identical conditions.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "../common/cuda_utils.h"
#include "../common/matrix.h"
#include "../common/verify.h"
#include "../kernels/kernels.h"
#include "gpu_monitor.h"

namespace {

struct Shape {
  int M, N, K;
};

struct Options {
  std::vector<std::string> kernels;
  std::vector<Shape> shapes;
  int reps = 100;
  double warmup_seconds = 30.0;
  double warmup_max_seconds = 0.0;  // 0 means 4x warmup_seconds
  int warmup_iters = 0;             // only used when warmup_seconds is 0
  bool log_clocks = false;
  double min_power_limit_w = 0.0;   // 0 disables the power-limit check
  bool allow_dirty = false;         // write CSV rows from uncommitted code
  double k0_baseline = 0.0;         // device baseline for k0, GFLOP/s; 0 = none
  double k0_band_pct = 0.0;         // accepted deviation from it, percent
  Shape baseline_shape = {4096, 4096, 4096};
  float alpha = 1.0f;
  float beta = 0.0f;
  bool check = true;
  bool bench = true;
  std::string csv;
  std::string device_tag;
  std::string tag;
};

// Warmup is over once the mean SM clock over kSettleWindow samples (one every
// kSampleInterval seconds) moves by at most kSettleTolerance from the window
// before it.
constexpr size_t kSettleWindow = 5;
constexpr double kSettleTolerance = 0.01;
constexpr double kSampleInterval = 1.0;
// Above this (max - min) / median, a median is not quotable on its own.
constexpr double kSpreadWarnPct = 5.0;
constexpr int kMinPublishReps = 100;

const char* const kCsvHeader =
    "timestamp_utc,source_rev,device_tag,gpu,kernel,M,N,K,alpha,beta,check,"
    "max_rel,fro_rel,warmup_mode,warmup_s_min,warmup_s,warmup_iters,"
    "warmup_settled,warmup_sm_mean_mhz,warmup_sm_range_pct,"
    "reps,ms_median,ms_p10,ms_p90,ms_min,ms_max,ms_max_rep,"
    "spread_pct,"
    "gflops_median,gflops_best,pct_cublas,"
    "pre_sm_mhz,pre_mem_mhz,pre_temp_c,pre_power_w,pre_reasons,"
    "start_sm_mhz,start_mem_mhz,start_temp_c,start_power_w,start_reasons,"
    "start_sample_age_s,"
    "end_sm_mhz,end_mem_mhz,end_temp_c,end_power_w,end_reasons,"
    "timed_s,sw_power_cap_ms,sw_thermal_ms,hw_thermal_ms,hw_power_brake_ms,"
    "tag,pre_power_limit_w,start_power_limit_w,end_power_limit_w,"
    "kernel_desc,baseline_check";

struct Warmup {
  const char* mode = "none";     // seconds | iters | none
  double seconds = 0.0;
  long iters = 0;
  const char* settled = "n/a";   // yes | no | unknown | n/a
  GpuSample last;                // last sample taken while under load
  ClockWindow clock;             // SM clock over the final settle window
};

struct Timing {
  double median_ms = 0, p10_ms = 0, p90_ms = 0, min_ms = 0, max_ms = 0;
  int max_rep = -1;  // which repetition was slowest; locates the outlier
  double start_s = 0, wall_s = 0;
};

double percentile(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) return 0.0;
  const double pos = q * (static_cast<double>(sorted.size()) - 1.0);
  const size_t lo = static_cast<size_t>(pos);
  const size_t hi = std::min(lo + 1, sorted.size() - 1);
  const double frac = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

// Warmup by time, not by count. An iteration count means different things for
// different shapes and devices: 25 iterations of a 3 ms kernel last 75 ms,
// which ends while the GPU is still on a boost clock it cannot sustain, and
// the first timed repetitions then run faster than the rest. Instead the GPU
// is kept loaded for at least warmup_seconds and then until the SM clock
// settles, up to a cap. Whether it settled is recorded, not assumed.
Warmup warm_up(SgemmFn fn, const Shape& s, const Options& o, const float* dA,
               const float* dB, float* dC, const std::string& gpu_id) {
  Warmup w;
  if (o.warmup_seconds > 0.0) {
    w.mode = "seconds";
    const double max_s = o.warmup_max_seconds > 0.0 ? o.warmup_max_seconds
                                                    : 4.0 * o.warmup_seconds;
    ClockSampler sampler(gpu_id, kSampleInterval);
    sampler.start();
    const double t0 = now_s();
    double last_check = t0;
    bool settled = false;
    for (;;) {
      fn(s.M, s.N, s.K, o.alpha, dA, dB, o.beta, dC);
      CUDA_CHECK(cudaDeviceSynchronize());
      ++w.iters;
      const double now = now_s();
      if (now - t0 < o.warmup_seconds || now - last_check < 0.25) continue;
      last_check = now;
      settled = clock_settled(sampler.snapshot(), kSettleWindow,
                              kSettleTolerance);
      if (settled || now - t0 >= max_s) break;
    }
    w.seconds = now_s() - t0;
    sampler.stop();
    const std::vector<GpuSample> samples = sampler.snapshot();
    if (!samples.empty()) w.last = samples.back();
    w.clock = clock_window(samples, kSettleWindow);
    w.settled = samples.empty() ? "unknown" : (settled ? "yes" : "no");
  } else if (o.warmup_iters > 0) {
    w.mode = "iters";
    const double t0 = now_s();
    for (int i = 0; i < o.warmup_iters; ++i)
      fn(s.M, s.N, s.K, o.alpha, dA, dB, o.beta, dC);
    CUDA_CHECK(cudaDeviceSynchronize());
    w.iters = o.warmup_iters;
    w.seconds = now_s() - t0;
  }
  return w;
}

// Time each iteration separately rather than dividing a total by a count: the
// per-iteration distribution exposes clock drift and occasional stalls.
Timing time_kernel(SgemmFn fn, const Shape& s, const Options& o,
                   const float* dA, const float* dB, float* dC) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  Timing t;
  std::vector<double> times;
  times.reserve(o.reps);
  t.start_s = now_s();
  for (int i = 0; i < o.reps; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    fn(s.M, s.N, s.K, o.alpha, dA, dB, o.beta, dC);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    times.push_back(static_cast<double>(ms));
  }
  t.wall_s = now_s() - t.start_s;
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  t.max_rep = static_cast<int>(std::max_element(times.begin(), times.end()) -
                               times.begin());
  std::sort(times.begin(), times.end());
  t.median_ms = percentile(times, 0.50);
  t.p10_ms = percentile(times, 0.10);
  t.p90_ms = percentile(times, 0.90);
  t.min_ms = times.front();
  t.max_ms = times.back();
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
      "  --kernel <name|all>        comma separated, default all\n"
      "  --shape MxNxK              repeatable, default 4096x4096x4096\n"
      "  --preset correctness       three shapes: main / non-divisible / tiny\n"
      "  --preset sweep             square sweep, 512..8192\n"
      "  --reps N                   default 100\n"
      "  --warmup-seconds S         load for >= S s, then until the SM clock\n"
      "                             settles; default 30, 0 disables\n"
      "  --warmup-max-seconds S     cap for the above, default 4x\n"
      "  --warmup N                 count-based warmup, only when\n"
      "                             --warmup-seconds is 0 (profiling runs)\n"
      "  --log-clocks               record clock / power / clock-event state\n"
      "  --device-tag <tag>         device label stored in every CSV row\n"
      "  --allow-dirty              allow CSV rows from uncommitted code (the\n"
      "                             row is still marked -dirty)\n"
      "  --k0-baseline G            device baseline for k0 in GFLOP/s at the\n"
      "                             baseline shape (4096^3); 0 disables\n"
      "  --k0-band P                accepted deviation from it, in percent\n"
      "  --min-power-limit W        refuse to benchmark below this enforced GPU\n"
      "                             power limit; default 0 (no check)\n"
      "  --alpha F --beta F         default 1.0 / 0.0\n"
      "  --no-check                 skip the correctness check\n"
      "  --no-bench                 correctness only\n"
      "  --csv <path>               append rows (header written if new); needs\n"
      "                             --device-tag, --log-clocks, time-based\n"
      "                             warmup and --reps >= %d\n"
      "  --tag <str>                free-form label, stored in the CSV\n",
      kMinPublishReps);
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

std::string shell_line(const char* cmd) {
  FILE* fp = popen(cmd, "r");
  if (!fp) return "";
  char buf[256];
  std::string out;
  if (std::fgets(buf, sizeof buf, fp)) out = buf;
  pclose(fp);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r'))
    out.pop_back();
  return out;
}

// The commit that produced a row, with "-dirty" when kernels, harness or build
// rules differ from it. Without this a CSV row cannot be traced back to code.
std::string source_revision() {
  std::string rev = shell_line("git rev-parse --short=12 HEAD 2>/dev/null");
  if (rev.empty()) return "unknown";
  if (!shell_line("git status --porcelain -- csrc Makefile 2>/dev/null").empty())
    rev += "-dirty";
  return rev;
}

std::string utc_timestamp() {
  std::time_t now = std::time(nullptr);
  std::tm tm_utc{};
  gmtime_r(&now, &tm_utc);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
  return buf;
}

// nvidia-smi selector for the device CUDA is actually running on.
std::string gpu_selector(const cudaDeviceProp& p) {
  const unsigned char* u = reinterpret_cast<const unsigned char*>(p.uuid.bytes);
  char buf[48];
  std::snprintf(buf, sizeof buf,
                "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
                "%02x%02x%02x%02x%02x%02x",
                u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9],
                u[10], u[11], u[12], u[13], u[14], u[15]);
  return buf;
}

bool valid_device_tag(const std::string& t) {
  if (t.empty()) return false;
  for (char c : t)
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-'))
      return false;
  return true;
}

std::string csv_safe(std::string s) {
  std::replace(s.begin(), s.end(), ',', ';');
  return s;
}

std::string fmt_int(long long v) { return v < 0 ? "" : std::to_string(v); }

std::string fmt_num(double v, const char* f) {
  if (!std::isfinite(v) || v < 0.0) return "";
  char buf[64];
  std::snprintf(buf, sizeof buf, f, v);
  return buf;
}

std::string state_cols(const GpuSample& g) {
  if (!g.valid) return ",,,,";
  return fmt_int(g.sm_mhz) + "," + fmt_int(g.mem_mhz) + "," +
         fmt_int(g.temp_c) + "," + fmt_num(g.power_w, "%.2f") + "," +
         g.reasons;
}

// Milliseconds a slowdown reason was active between two samples.
double active_ms(long long before_us, long long after_us) {
  if (before_us < 0 || after_us < 0 || after_us < before_us) return -1.0;
  return static_cast<double>(after_us - before_us) / 1000.0;
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
    } else if (a == "--warmup-seconds") {
      o.warmup_seconds = std::atof(next("--warmup-seconds"));
    } else if (a == "--warmup-max-seconds") {
      o.warmup_max_seconds = std::atof(next("--warmup-max-seconds"));
    } else if (a == "--warmup") {
      o.warmup_iters = std::atoi(next("--warmup"));
    } else if (a == "--log-clocks") {
      o.log_clocks = true;
    } else if (a == "--allow-dirty") {
      o.allow_dirty = true;
    } else if (a == "--k0-baseline") {
      o.k0_baseline = std::atof(next("--k0-baseline"));
    } else if (a == "--k0-band") {
      o.k0_band_pct = std::atof(next("--k0-band"));
    } else if (a == "--min-power-limit") {
      o.min_power_limit_w = std::atof(next("--min-power-limit"));
    } else if (a == "--device-tag") {
      o.device_tag = next("--device-tag");
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
  if (o.reps < 1) {
    std::fprintf(stderr, "--reps must be >= 1\n");
    return 2;
  }
  if (o.warmup_seconds > 0.0 && o.warmup_iters > 0) {
    std::fprintf(stderr, "--warmup N only applies with --warmup-seconds 0\n");
    return 2;
  }
  if (!o.device_tag.empty() && !valid_device_tag(o.device_tag)) {
    std::fprintf(stderr, "--device-tag must match [a-z0-9-]+\n");
    return 2;
  }
  // Rows in a CSV are what published numbers get traced back to, so the
  // conditions that make a row interpretable are enforced here rather than
  // left to whoever invokes the runner.
  if (!o.csv.empty() && o.bench) {
    std::vector<const char*> missing;
    if (o.device_tag.empty()) missing.push_back("--device-tag");
    if (!o.log_clocks) missing.push_back("--log-clocks");
    if (o.warmup_seconds <= 0.0) missing.push_back("--warmup-seconds > 0");
    if (o.reps < kMinPublishReps) missing.push_back("--reps >= 100");
    if (!missing.empty()) {
      std::fprintf(stderr, "--csv requires:");
      for (const char* m : missing) std::fprintf(stderr, " %s", m);
      std::fprintf(stderr, "\n");
      return 2;
    }
  }

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  const std::string gpu_id = gpu_selector(prop);
  const std::string rev = source_revision();
  std::printf("GPU: %s  sm_%d%d  SMs=%d  L2=%dKB  smemPerBlock=%zuKB\n",
              prop.name, prop.major, prop.minor, prop.multiProcessorCount,
              prop.l2CacheSize / 1024, prop.sharedMemPerBlock / 1024);
  std::printf("device tag: %s  source: %s\n",
              o.device_tag.empty() ? "(none)" : o.device_tag.c_str(),
              rev.c_str());
  // A row is only traceable if the code that produced it exists in history.
  const bool dirty =
      rev == "unknown" ||
      (rev.size() > 6 && rev.compare(rev.size() - 6, 6, "-dirty") == 0);
  if (!o.csv.empty() && o.bench && dirty && !o.allow_dirty) {
    std::fprintf(stderr,
                 "refusing to write CSV rows from uncommitted code (source "
                 "%s): commit first, or pass --allow-dirty for an "
                 "exploratory run\n",
                 rev.c_str());
    return 1;
  }
  if (o.bench) {
    if (o.warmup_seconds > 0.0)
      std::printf("bench: warmup >= %.0f s until SM clock settles (cap %.0f s)",
                  o.warmup_seconds,
                  o.warmup_max_seconds > 0.0 ? o.warmup_max_seconds
                                             : 4.0 * o.warmup_seconds);
    else
      std::printf("bench: warmup %d iterations", o.warmup_iters);
    std::printf(", reps=%d, per-iteration timing, median reported\n", o.reps);
  }
  std::printf("\n");

  // A GPU's enforced power limit is not a constant: a laptop part drops to a
  // fraction of it when the host leaves its high-performance power plan, and
  // every number measured afterwards describes a different machine. Checked
  // before any work starts, and again after each timed region.
  if (o.bench && o.min_power_limit_w > 0.0) {
    const GpuSample s = sample_gpu(gpu_id);
    if (!s.valid || s.power_limit_w < 0.0) {
      std::fprintf(stderr, "cannot read the enforced power limit; refusing to "
                           "benchmark with --min-power-limit set\n");
      return 1;
    }
    if (s.power_limit_w < o.min_power_limit_w) {
      std::fprintf(stderr,
                   "enforced power limit %.1f W is below the required %.1f W: "
                   "benchmark conditions not met (check the host power plan "
                   "and power adapter)\n",
                   s.power_limit_w, o.min_power_limit_w);
      return 1;
    }
    std::printf("power limit: %.1f W (required >= %.1f W)\n\n",
                s.power_limit_w, o.min_power_limit_w);
  }

  FILE* csv = nullptr;
  if (!o.csv.empty() && o.bench) {
    std::string first_line;
    if (FILE* f = std::fopen(o.csv.c_str(), "r")) {
      char buf[4096];
      if (std::fgets(buf, sizeof buf, f)) first_line = buf;
      std::fclose(f);
      while (!first_line.empty() &&
             (first_line.back() == '\n' || first_line.back() == '\r'))
        first_line.pop_back();
      if (first_line != kCsvHeader) {
        std::fprintf(stderr,
                     "CSV header mismatch in %s: file was written by a "
                     "different harness version; use a new file\n",
                     o.csv.c_str());
        return 1;
      }
    }
    csv = std::fopen(o.csv.c_str(), "a");
    if (!csv) {
      std::fprintf(stderr, "cannot open CSV: %s\n", o.csv.c_str());
      return 1;
    }
    if (first_line.empty()) std::fprintf(csv, "%s\n", kCsvHeader);
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
    // Whether this run's k0 sits inside the device's run-to-run band. Out of
    // band means the machine is in a different state (thermal, power) from
    // the one the baseline was taken in: ratios inside this run still hold,
    // absolute numbers do not compare across runs.
    const bool baseline_applies =
        o.k0_baseline > 0.0 && s.M == o.baseline_shape.M &&
        s.N == o.baseline_shape.N && s.K == o.baseline_shape.K;
    const char* baseline_state = baseline_applies ? "no-k0-yet" : "n/a";
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
      Warmup w;
      GpuSample pre, start, end;
      if (o.bench) {
        if (o.log_clocks || o.min_power_limit_w > 0.0) pre = sample_gpu(gpu_id);
        w = warm_up(ke->fn, s, o, dA, dB, dC, gpu_id);
        start = w.last;
        if (o.log_clocks && !start.valid) start = sample_gpu(gpu_id);
        t = time_kernel(ke->fn, s, o, dA, dB, dC);
        if (o.log_clocks || o.min_power_limit_w > 0.0) end = sample_gpu(gpu_id);
      }
      const bool power_ok =
          !o.bench || o.min_power_limit_w <= 0.0 ||
          (end.valid && end.power_limit_w >= o.min_power_limit_w);
      if (!power_ok) {
        std::printf("       ! enforced power limit dropped to %.1f W during "
                    "the run (required >= %.1f W)\n",
                    end.power_limit_w, o.min_power_limit_w);
        failures++;
      }
      const double g_med = o.bench ? gflops_of(s, t.median_ms) : 0.0;
      const double g_best = o.bench ? gflops_of(s, t.min_ms) : 0.0;
      if (kname == "k0") {
        cublas_gflops = g_med;
        if (baseline_applies && o.bench) {
          const double dev_pct = 100.0 * (g_med / o.k0_baseline - 1.0);
          const bool in_band = std::fabs(dev_pct) <= o.k0_band_pct;
          baseline_state = in_band ? "in-band" : "out-of-band";
          if (!in_band)
            std::printf("       ! k0 is %+.2f%% from the device baseline "
                        "%.1f (band +-%.1f%%): absolute numbers from this run "
                        "do not compare across runs; ratios within it still "
                        "do\n",
                        dev_pct, o.k0_baseline, o.k0_band_pct);
        }
      }
      const double pct =
          (cublas_gflops > 0.0) ? 100.0 * g_med / cublas_gflops : 0.0;
      const double spread =
          (t.median_ms > 0.0) ? 100.0 * (t.max_ms - t.min_ms) / t.median_ms : 0.0;

      std::printf("%-6s %11.3f %9.3f %9.3f %9.1f %9.1f%% %10.2e %10.2e  %s\n",
                  ke->name, t.median_ms, t.p10_ms, t.p90_ms, g_med, pct,
                  vr.max_rel, vr.fro_rel, ok ? "" : "<< FAIL");
      if (!ok)
        std::printf("       worst @ %zu: ref=%.6f got=%.6f\n", vr.worst_index,
                    vr.worst_ref, vr.worst_got);

      if (o.bench) {
        if (std::strcmp(w.mode, "seconds") == 0)
          std::printf("       warmup %.1f s, %ld iters, SM clock %s: "
                      "mean %.0f MHz, range %.1f%% over the last %zu s\n",
                      w.seconds, w.iters,
                      std::strcmp(w.settled, "yes") == 0   ? "settled"
                      : std::strcmp(w.settled, "no") == 0 ? "did NOT settle"
                                                          : "unknown",
                      w.clock.mean_mhz, w.clock.range_pct,
                      static_cast<size_t>(kSettleWindow * kSampleInterval));
        if (o.log_clocks && start.valid && end.valid) {
          const double cap_ms =
              active_ms(start.us_sw_power_cap, end.us_sw_power_cap);
          std::printf("       timed %.1f s, SM clock %d -> %d MHz, %d -> %d C, "
                      "SW power cap active %.0f ms\n",
                      t.wall_s, start.sm_mhz, end.sm_mhz, start.temp_c,
                      end.temp_c, cap_ms);
        }
        if (spread > kSpreadWarnPct)
          std::printf("       ! spread (max-min)/median = %.1f%% > %.0f%% "
                      "(slowest: rep %d of %d): explain before quoting\n",
                      spread, kSpreadWarnPct, t.max_rep, o.reps);
      }

      if (csv && !ok)
        std::printf("       row not written: correctness check failed\n");
      if (csv && ok && !power_ok)
        std::printf("       row not written: power limit below the minimum\n");
      if (csv && ok && power_ok) {
        const double age = (start.valid && t.start_s > 0.0)
                               ? t.start_s - start.t_s
                               : -1.0;
        std::string row;
        row += utc_timestamp() + "," + rev + "," + o.device_tag + "," +
               csv_safe(prop.name) + "," + ke->name + ",";
        row += std::to_string(s.M) + "," + std::to_string(s.N) + "," +
               std::to_string(s.K) + ",";
        char buf[512];
        std::snprintf(buf, sizeof buf, "%g,%g,%s,%s,%s,", o.alpha, o.beta,
                      o.check ? "pass" : "skipped",
                      o.check ? fmt_num(vr.max_rel, "%.3e").c_str() : "",
                      o.check ? fmt_num(vr.fro_rel, "%.3e").c_str() : "");
        row += buf;
        row += std::string(w.mode) + "," + fmt_num(o.warmup_seconds, "%.1f") +
               "," + fmt_num(w.seconds, "%.2f") + "," +
               std::to_string(w.iters) + "," + w.settled + "," +
               fmt_num(w.clock.mean_mhz, "%.0f") + "," +
               fmt_num(w.clock.range_pct, "%.2f") + ",";
        std::snprintf(buf, sizeof buf,
                      "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.2f,%.3f,%.3f,", o.reps,
                      t.median_ms, t.p10_ms, t.p90_ms, t.min_ms, t.max_ms,
                      t.max_rep, spread, g_med, g_best);
        row += buf;
        row += fmt_num(cublas_gflops > 0.0 ? pct : -1.0, "%.2f") + ",";
        row += state_cols(pre) + "," + state_cols(start) + "," +
               fmt_num(age, "%.2f") + "," + state_cols(end) + ",";
        row += fmt_num(t.wall_s, "%.2f") + ",";
        row += fmt_num(active_ms(start.us_sw_power_cap, end.us_sw_power_cap),
                       "%.1f") + ",";
        row += fmt_num(active_ms(start.us_sw_thermal, end.us_sw_thermal),
                       "%.1f") + ",";
        row += fmt_num(active_ms(start.us_hw_thermal, end.us_hw_thermal),
                       "%.1f") + ",";
        row += fmt_num(active_ms(start.us_hw_power_brake,
                                 end.us_hw_power_brake),
                       "%.1f") + ",";
        row += csv_safe(o.tag) + ",";
        row += fmt_num(pre.power_limit_w, "%.2f") + "," +
               fmt_num(start.power_limit_w, "%.2f") + "," +
               fmt_num(end.power_limit_w, "%.2f") + ",";
        row += csv_safe(ke->desc) + ",";
        row += baseline_state;
        std::fprintf(csv, "%s\n", row.c_str());
        std::fflush(csv);
      }
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
