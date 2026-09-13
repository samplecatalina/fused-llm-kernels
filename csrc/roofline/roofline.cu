// roofline - measured ceilings for the roofline model.
//
// A roofline plot bounds achievable throughput by two roofs: the memory
// bandwidth the part can actually deliver (a slope: FLOP/s = bandwidth x
// arithmetic intensity) and the FLOP rate its SMs can actually sustain (a
// flat roof). Datasheet figures do not work on a mobile part whose clocks and
// power budget move, so both roofs are measured here, under the same
// discipline as the SGEMM benchmarks: time-based warmup until the SM clock
// settles, per-iteration timing, clock and power state recorded, and no rows
// from uncommitted code.
//
// Benchmarks:
//   copy_f4   dst = src, four floats per load/store    bandwidth roof
//   copy_f1   dst = src, one float per load/store      bandwidth, scalar
//   triad_f4  a = b + s*c, four floats per access      a point deep in the
//                                                      memory-bound region
//   fma_f4    x = x*a + b in registers, four lanes     compute roof
//   fma_f1    the same with one lane                   compute, scalar
//   gemm_f16  cuBLAS FP16 GEMM, FP32 accumulate         tensor-core roof,
//                                                       the ceiling the
//                                                       attention kernel is
//                                                       reported against
//
// Traffic is counted as bytes loaded plus bytes stored. FLOPs count every
// scalar multiply and add.
#include <algorithm>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../common/cuda_utils.h"
#include "../common/matrix.h"
#include "../harness/gpu_monitor.h"
#include "../harness/provenance.h"

namespace {

constexpr int kThreadsPerBlock = 1024;
// 64M floats = 256 MiB per array, eight times this part's L2, so the copy
// and triad traffic has to come from global memory.
constexpr int kMemElems = 64 * 1024 * 1024;
constexpr int kFmaThreads = 64 * kThreadsPerBlock;
// Enough iterations that launch overhead is negligible and throughput has
// reached its plateau. With 1024 a call took about 70 us and read 7.0
// TFLOP/s, below cuBLAS itself; throughput stops rising by about 262144
// (within ~1% of a million iterations for both variants) while a call still
// lasts only 3 to 11 ms.
constexpr int kFmaItersDefault = 262144;
int g_fma_iters = kFmaItersDefault;
// FP16 GEMM size. Throughput still rises with N here (21.5 TFLOP/s at 2048,
// 28.1 at 4096); 8192 is the largest square that fits alongside the other
// buffers in 8 GB and is where it flattens.
constexpr int kGemmN = 8192;

dim3 grid_for(int threads) {
  return dim3((threads + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

}  // namespace

__global__ void copy_f4_kernel(const float* __restrict__ src,
                               float* __restrict__ dst, int n4) {
  const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= n4) return;
  *reinterpret_cast<float4*>(&dst[4 * t]) =
      *reinterpret_cast<const float4*>(&src[4 * t]);
}

__global__ void copy_f1_kernel(const float* __restrict__ src,
                               float* __restrict__ dst, int n) {
  const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= n) return;
  dst[t] = src[t];
}

__global__ void triad_f4_kernel(float* __restrict__ a,
                                const float* __restrict__ b,
                                const float* __restrict__ c, float s, int n4) {
  const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= n4) return;
  const float4 bb = *reinterpret_cast<const float4*>(&b[4 * t]);
  const float4 cc = *reinterpret_cast<const float4*>(&c[4 * t]);
  *reinterpret_cast<float4*>(&a[4 * t]) =
      make_float4(bb.x + s * cc.x, bb.y + s * cc.y, bb.z + s * cc.z,
                  bb.w + s * cc.w);
}

// Seeds lie in [0.6, 0.9]; x converges to 0.1 / (1 - a), so the loop never
// reaches denormals or infinities, which would change its speed.
__global__ void fma_f4_kernel(const float* __restrict__ seed,
                              float* __restrict__ out, int threads, int iters) {
  const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= threads) return;
  const float4 a = *reinterpret_cast<const float4*>(&seed[4 * t]);
  float x0 = a.x, x1 = a.y, x2 = a.z, x3 = a.w;
  const float b = 0.1f;
  for (int i = 0; i < iters; ++i) {
    x0 = x0 * a.x + b;
    x1 = x1 * a.y + b;
    x2 = x2 * a.z + b;
    x3 = x3 * a.w + b;
  }
  *reinterpret_cast<float4*>(&out[4 * t]) = make_float4(x0, x1, x2, x3);
}

__global__ void fma_f1_kernel(const float* __restrict__ seed,
                              float* __restrict__ out, int threads, int iters) {
  const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= threads) return;
  const float a = seed[4 * t];
  float x = a;
  const float b = 0.1f;
  for (int i = 0; i < iters; ++i) x = x * a + b;
  out[4 * t] = x;
}

namespace {

struct Bench {
  const char* name;
  double bytes_per_call;
  double flops_per_call;
  long long elements;
  int threads;
  int iters;
};

struct Buffers {
  float* x = nullptr;  // src / b / seed
  float* y = nullptr;  // dst / a / out
  float* z = nullptr;  // c
};

// FP16 operands with FP32 accumulation: what a tensor-core matmul does, and
// what the attention kernel's dots do.
struct GemmBuffers {
  __half* a = nullptr;
  __half* b = nullptr;
  __half* c = nullptr;
  cublasHandle_t handle = nullptr;
};

GemmBuffers g_gemm;

void run_gemm_f16() {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(g_gemm.handle, CUBLAS_OP_N, CUBLAS_OP_N, kGemmN,
                            kGemmN, kGemmN, &alpha, g_gemm.a, CUDA_R_16F,
                            kGemmN, g_gemm.b, CUDA_R_16F, kGemmN, &beta,
                            g_gemm.c, CUDA_R_16F, kGemmN, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void run_once(const std::string& name, const Buffers& d) {
  const int n4 = kMemElems / 4;
  dim3 block(kThreadsPerBlock);
  if (name == "gemm_f16") {
    run_gemm_f16();
    CUDA_CHECK(cudaDeviceSynchronize());
    return;
  }
  if (name == "copy_f4") {
    copy_f4_kernel<<<grid_for(n4), block>>>(d.x, d.y, n4);
  } else if (name == "copy_f1") {
    copy_f1_kernel<<<grid_for(kMemElems), block>>>(d.x, d.y, kMemElems);
  } else if (name == "triad_f4") {
    triad_f4_kernel<<<grid_for(n4), block>>>(d.y, d.x, d.z, 0.5f, n4);
  } else if (name == "fma_f4") {
    fma_f4_kernel<<<grid_for(kFmaThreads), block>>>(d.x, d.y, kFmaThreads,
                                                    g_fma_iters);
  } else {
    fma_f1_kernel<<<grid_for(kFmaThreads), block>>>(d.x, d.y, kFmaThreads,
                                                    g_fma_iters);
  }
  check_launch(name.c_str());
  CUDA_CHECK(cudaDeviceSynchronize());
}

Bench describe(const std::string& name) {
  const double f = 4.0;  // bytes per float
  if (name == "copy_f4" || name == "copy_f1")
    return {nullptr, 2.0 * kMemElems * f, 0.0, kMemElems,
            name == "copy_f4" ? kMemElems / 4 : kMemElems, 0};
  if (name == "triad_f4")
    return {nullptr, 3.0 * kMemElems * f, 2.0 * kMemElems, kMemElems,
            kMemElems / 4, 0};
  if (name == "gemm_f16") {
    const double n = kGemmN;
    // Two FP16 reads and one FP16 write of an n x n matrix; 2 n^3 FLOPs.
    return {nullptr, 3.0 * n * n * 2.0, 2.0 * n * n * n,
            static_cast<long long>(n * n), 0, 0};
  }
  const double lanes = (name == "fma_f4") ? 4.0 : 1.0;
  // one load and one store of a float4 per thread
  return {nullptr, 2.0 * kFmaThreads * 16.0,
          2.0 * lanes * kFmaThreads * g_fma_iters, 0, kFmaThreads, g_fma_iters};
}

double percentile(const std::vector<double>& v, double q) {
  const double pos = q * (static_cast<double>(v.size()) - 1.0);
  const size_t lo = static_cast<size_t>(pos);
  const size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (v[hi] - v[lo]) * (pos - static_cast<double>(lo));
}

constexpr size_t kSettleWindow = 5;
constexpr double kSettleTolerance = 0.01;

const char* const kHeader =
    "timestamp_utc,source_rev,device_tag,gpu,bench,elements,threads,iters,"
    "reps,warmup_s,warmup_settled,warmup_sm_mean_mhz,warmup_sm_range_pct,"
    "ms_median,ms_p10,ms_p90,spread_pct,bytes_per_call,flops_per_call,"
    "gb_per_s,gflops,arith_intensity,start_sm_mhz,start_temp_c,start_power_w,"
    "start_power_limit_w,end_sm_mhz,end_temp_c,end_power_w,end_power_limit_w,"
    "tag";

void usage() {
  std::printf(
      "usage: roofline [options]\n"
      "  --bench <list>          comma separated from copy_f4,copy_f1,triad_f4,\n"
      "                          fma_f4,fma_f1,gemm_f16; default all but\n"
      "                          gemm_f16\n"
      "  --reps N                default 100\n"
      "  --fma-iters N           iterations per thread in fma_*, default 262144\n"
      "  --warmup-seconds S      load for >= S s, then until the SM clock\n"
      "                          settles (cap 4x); default 30\n"
      "  --min-power-limit W     refuse to run below this enforced power limit\n"
      "  --device-tag <tag>      required with --csv\n"
      "  --csv <path>            append rows; refused for uncommitted code\n"
      "  --allow-dirty           permit rows from uncommitted code\n"
      "  --tag <str>             free-form label\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> benches;
  int reps = 100;
  double warmup_s = 30.0, min_power = 0.0;
  bool allow_dirty = false;
  std::string csv_path, device_tag, tag;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
      return argv[++i];
    };
    if (a == "--bench") {
      std::string v = next();
      size_t p = 0;
      while (p <= v.size()) {
        size_t c = v.find(',', p);
        if (c == std::string::npos) c = v.size();
        if (c > p) benches.push_back(v.substr(p, c - p));
        p = c + 1;
      }
    } else if (a == "--reps") reps = std::atoi(next());
    else if (a == "--fma-iters") g_fma_iters = std::atoi(next());
    else if (a == "--warmup-seconds") warmup_s = std::atof(next());
    else if (a == "--min-power-limit") min_power = std::atof(next());
    else if (a == "--device-tag") device_tag = next();
    else if (a == "--csv") csv_path = next();
    else if (a == "--allow-dirty") allow_dirty = true;
    else if (a == "--tag") tag = next();
    else if (a == "-h" || a == "--help") { usage(); return 0; }
    else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(); return 2; }
  }
  if (benches.empty()) benches = {"copy_f4", "copy_f1", "triad_f4", "fma_f4", "fma_f1"};
  for (const auto& b : benches) {
    if (b != "copy_f4" && b != "copy_f1" && b != "triad_f4" && b != "fma_f4" &&
        b != "fma_f1" && b != "gemm_f16") {
      std::fprintf(stderr, "unknown bench: %s\n", b.c_str());
      return 2;
    }
  }
  if (!csv_path.empty()) {
    if (!valid_device_tag(device_tag) || warmup_s <= 0.0 || reps < 100) {
      std::fprintf(stderr, "--csv requires --device-tag, --warmup-seconds > 0 and --reps >= 100\n");
      return 2;
    }
  }

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  const std::string gpu_id = gpu_selector(prop);
  const std::string rev = source_revision();
  std::printf("GPU: %s  source: %s\n", prop.name, rev.c_str());
  if (!csv_path.empty() && is_untraceable(rev) && !allow_dirty) {
    std::fprintf(stderr, "refusing to write rows from uncommitted code (source %s)\n", rev.c_str());
    return 1;
  }
  if (min_power > 0.0) {
    const GpuSample s = sample_gpu(gpu_id);
    if (!s.valid || s.power_limit_w < min_power) {
      std::fprintf(stderr, "enforced power limit %.1f W is below the required %.1f W\n",
                   s.power_limit_w, min_power);
      return 1;
    }
  }

  // Inputs: random floats for the memory benchmarks, seeds in [0.6, 0.9]
  // for the compute ones.
  Buffers mem, fma;
  {
    std::mt19937 gen(99);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f), s(0.6f, 0.9f);
    std::vector<float> h(kMemElems);
    mem.x = device_alloc(kMemElems);
    mem.y = device_alloc(kMemElems);
    mem.z = device_alloc(kMemElems);
    for (auto& v : h) v = u(gen);
    host_to_device(mem.x, h);
    for (auto& v : h) v = u(gen);
    host_to_device(mem.z, h);
    CUDA_CHECK(cudaMemset(mem.y, 0, static_cast<size_t>(kMemElems) * sizeof(float)));
    std::vector<float> seeds(4 * kFmaThreads);
    for (auto& v : seeds) v = s(gen);
    fma.x = device_alloc(seeds.size());
    fma.y = device_alloc(seeds.size());
    host_to_device(fma.x, seeds);
  }

  const bool want_gemm =
      std::find(benches.begin(), benches.end(), "gemm_f16") != benches.end();
  if (want_gemm) {
    const size_t n2 = static_cast<size_t>(kGemmN) * kGemmN;
    std::vector<__half> h(n2);
    std::mt19937 gen(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    CUDA_CHECK(cudaMalloc(&g_gemm.a, n2 * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&g_gemm.b, n2 * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&g_gemm.c, n2 * sizeof(__half)));
    for (auto& v : h) v = __float2half(u(gen));
    CUDA_CHECK(cudaMemcpy(g_gemm.a, h.data(), n2 * sizeof(__half),
                          cudaMemcpyHostToDevice));
    for (auto& v : h) v = __float2half(u(gen));
    CUDA_CHECK(cudaMemcpy(g_gemm.b, h.data(), n2 * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUBLAS_CHECK(cublasCreate(&g_gemm.handle));
  }

  FILE* csv = nullptr;
  if (!csv_path.empty()) {
    std::string first;
    if (FILE* f = std::fopen(csv_path.c_str(), "r")) {
      char buf[2048];
      if (std::fgets(buf, sizeof buf, f)) first = buf;
      std::fclose(f);
      while (!first.empty() && (first.back() == '\n' || first.back() == '\r')) first.pop_back();
      if (first != kHeader) { std::fprintf(stderr, "CSV header mismatch in %s\n", csv_path.c_str()); return 1; }
    }
    csv = std::fopen(csv_path.c_str(), "a");
    if (!csv) { std::fprintf(stderr, "cannot open %s\n", csv_path.c_str()); return 1; }
    if (first.empty()) std::fprintf(csv, "%s\n", kHeader);
  }

  std::printf("%-9s %10s %9s %9s %10s %10s %8s\n", "bench", "ms(median)", "p10", "p90",
              "GB/s", "GFLOP/s", "spread");
  for (const std::string& name : benches) {
    const Buffers& d = (name.rfind("fma", 0) == 0) ? fma : mem;
    const Bench b = describe(name);

    // Time-based warmup, same criterion as the SGEMM runner.
    ClockSampler sampler(gpu_id, 1.0);
    sampler.start();
    const double t0 = now_s();
    double last_check = t0;
    bool settled = false;
    for (;;) {
      run_once(name, d);
      const double now = now_s();
      if (now - t0 < warmup_s || now - last_check < 0.25) continue;
      last_check = now;
      settled = clock_settled(sampler.snapshot(), kSettleWindow, kSettleTolerance);
      if (settled || now - t0 >= 4.0 * warmup_s) break;
    }
    const double warm = now_s() - t0;
    sampler.stop();
    const std::vector<GpuSample> samples = sampler.snapshot();
    const GpuSample start = samples.empty() ? sample_gpu(gpu_id) : samples.back();
    const ClockWindow cw = clock_window(samples, kSettleWindow);

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    std::vector<double> ms;
    for (int r = 0; r < reps; ++r) {
      CUDA_CHECK(cudaEventRecord(e0));
      run_once(name, d);
      CUDA_CHECK(cudaEventRecord(e1));
      CUDA_CHECK(cudaEventSynchronize(e1));
      float v = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&v, e0, e1));
      ms.push_back(v);
    }
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    const GpuSample end = sample_gpu(gpu_id);

    std::sort(ms.begin(), ms.end());
    const double med = percentile(ms, 0.5), p10 = percentile(ms, 0.1), p90 = percentile(ms, 0.9);
    const double spread = 100.0 * (ms.back() - ms.front()) / med;
    const double gbps = b.bytes_per_call / (med * 1e-3) / 1e9;
    const double gflops = b.flops_per_call / (med * 1e-3) / 1e9;
    const double ai = b.bytes_per_call > 0.0 ? b.flops_per_call / b.bytes_per_call : -1.0;
    std::printf("%-9s %10.3f %9.3f %9.3f %10.2f %10.1f %7.1f%%  (warmup %.1f s, %s, SM %.0f MHz)\n",
                name.c_str(), med, p10, p90, gbps, gflops, spread, warm,
                settled ? "settled" : "not settled", cw.mean_mhz);

    const bool power_ok = min_power <= 0.0 || (end.valid && end.power_limit_w >= min_power);
    if (!power_ok) std::printf("          ! power limit dropped below %.1f W: row not written\n", min_power);
    if (csv && power_ok) {
      std::string row = utc_timestamp() + "," + rev + "," + device_tag + "," +
                        csv_safe(prop.name) + "," + name + "," + fmt_int(b.elements) + "," +
                        std::to_string(b.threads) + "," + std::to_string(b.iters) + "," +
                        std::to_string(reps) + "," + fmt_num(warm, "%.2f") + "," +
                        (samples.empty() ? "unknown" : (settled ? "yes" : "no")) + "," +
                        fmt_num(cw.mean_mhz, "%.0f") + "," + fmt_num(cw.range_pct, "%.2f") + "," +
                        fmt_num(med, "%.6f") + "," + fmt_num(p10, "%.6f") + "," +
                        fmt_num(p90, "%.6f") + "," + fmt_num(spread, "%.2f") + "," +
                        fmt_num(b.bytes_per_call, "%.0f") + "," + fmt_num(b.flops_per_call, "%.0f") + "," +
                        fmt_num(gbps, "%.3f") + "," + fmt_num(gflops, "%.3f") + "," +
                        fmt_num(ai, "%.6f") + "," + fmt_int(start.sm_mhz) + "," +
                        fmt_int(start.temp_c) + "," + fmt_num(start.power_w, "%.2f") + "," +
                        fmt_num(start.power_limit_w, "%.2f") + "," + fmt_int(end.sm_mhz) + "," +
                        fmt_int(end.temp_c) + "," + fmt_num(end.power_w, "%.2f") + "," +
                        fmt_num(end.power_limit_w, "%.2f") + "," + csv_safe(tag);
      std::fprintf(csv, "%s\n", row.c_str());
      std::fflush(csv);
    }
  }
  if (csv) std::fclose(csv);
  for (float* p : {mem.x, mem.y, mem.z, fma.x, fma.y}) CUDA_CHECK(cudaFree(p));
  if (g_gemm.handle) {
    CUBLAS_CHECK(cublasDestroy(g_gemm.handle));
    for (__half* p : {g_gemm.a, g_gemm.b, g_gemm.c}) CUDA_CHECK(cudaFree(p));
  }
  return 0;
}
