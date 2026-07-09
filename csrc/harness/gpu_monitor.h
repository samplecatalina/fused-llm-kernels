#pragma once
// GPU clock, power and clock-event sampling through nvidia-smi.
//
// A benchmark number is only interpretable next to the clock it ran at. Under
// sustained load neither a laptop part nor a datacenter part holds its boost
// SM clock, and clocks cannot be locked from user space on either. These
// helpers record that state; they never try to control it.
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct GpuSample {
  bool valid = false;
  double t_s = 0.0;  // steady-clock seconds, same origin as now_s()
  int sm_mhz = -1;
  int mem_mhz = -1;
  int temp_c = -1;
  double power_w = -1.0;  // -1 when not reported
  std::string reasons;    // clocks_event_reasons.active, hex bitmask
  double power_limit_w = -1.0;  // enforced power limit, -1 when not reported
  // Cumulative microseconds each slowdown reason has been active since the
  // driver loaded. Only the difference between two samples means anything.
  long long us_sw_power_cap = -1;
  long long us_sw_thermal = -1;
  long long us_hw_thermal = -1;
  long long us_hw_power_brake = -1;
};

inline double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

namespace gpu_monitor_detail {

inline std::string trim(const std::string& s) {
  const size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  const size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

// Fields nvidia-smi cannot read come back as "[N/A]" or similar; those map to
// -1 rather than failing the whole sample.
inline long long to_ll(const std::string& s) {
  char* end = nullptr;
  const long long v = std::strtoll(s.c_str(), &end, 10);
  return (!s.empty() && end && *end == '\0') ? v : -1;
}

inline double to_d(const std::string& s) {
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  return (!s.empty() && end && *end == '\0') ? v : -1.0;
}

}  // namespace gpu_monitor_detail

// `selector` is an nvidia-smi --id value (GPU UUID or PCI bus id). It matters
// on multi-GPU nodes, where the device CUDA runs on is not necessarily index 0.
inline GpuSample sample_gpu(const std::string& selector) {
  using namespace gpu_monitor_detail;
  std::string cmd =
      "nvidia-smi --query-gpu=clocks.current.sm,clocks.current.memory,"
      "temperature.gpu,power.draw,clocks_event_reasons.active,"
      "clocks_event_reasons_counters.sw_power_cap,"
      "clocks_event_reasons_counters.sw_thermal_slowdown,"
      "clocks_event_reasons_counters.hw_thermal_slowdown,"
      "clocks_event_reasons_counters.hw_power_brake_slowdown,"
      "enforced.power.limit "
      "--format=csv,noheader,nounits";
  if (!selector.empty()) cmd += " --id=" + selector;
  cmd += " 2>/dev/null";

  GpuSample s;
  s.t_s = now_s();
  FILE* fp = popen(cmd.c_str(), "r");
  if (!fp) return s;
  char line[512];
  const bool got = std::fgets(line, sizeof line, fp) != nullptr;
  pclose(fp);
  if (!got) return s;

  std::vector<std::string> f;
  std::string cur;
  for (const char* p = line; *p; ++p) {
    if (*p == ',') {
      f.push_back(trim(cur));
      cur.clear();
    } else {
      cur += *p;
    }
  }
  f.push_back(trim(cur));
  if (f.size() != 10) return s;

  s.sm_mhz = static_cast<int>(to_ll(f[0]));
  s.mem_mhz = static_cast<int>(to_ll(f[1]));
  s.temp_c = static_cast<int>(to_ll(f[2]));
  s.power_w = to_d(f[3]);
  s.reasons = (f[4].rfind("0x", 0) == 0) ? f[4] : "";
  s.us_sw_power_cap = to_ll(f[5]);
  s.us_sw_thermal = to_ll(f[6]);
  s.us_hw_thermal = to_ll(f[7]);
  s.us_hw_power_brake = to_ll(f[8]);
  s.power_limit_w = to_d(f[9]);
  s.valid = s.sm_mhz >= 0;
  return s;
}

// Samples in a background thread so the GPU stays loaded while sampling: one
// nvidia-smi call takes tens of milliseconds, long enough for an idle GPU to
// start changing clock state.
class ClockSampler {
 public:
  ClockSampler(std::string selector, double interval_s)
      : selector_(std::move(selector)), interval_s_(interval_s) {}
  ~ClockSampler() { stop(); }
  ClockSampler(const ClockSampler&) = delete;
  ClockSampler& operator=(const ClockSampler&) = delete;

  void start() {
    stop();
    {
      std::lock_guard<std::mutex> lk(mu_);
      samples_.clear();
      stop_ = false;
    }
    thread_ = std::thread([this] { run(); });
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
  }

  std::vector<GpuSample> snapshot() const {
    std::lock_guard<std::mutex> lk(mu_);
    return samples_;
  }

 private:
  void run() {
    std::unique_lock<std::mutex> lk(mu_);
    while (!stop_) {
      lk.unlock();
      GpuSample s = sample_gpu(selector_);
      lk.lock();
      if (s.valid) samples_.push_back(s);
      cv_.wait_for(lk, std::chrono::duration<double>(interval_s_),
                   [this] { return stop_; });
    }
  }

  std::string selector_;
  double interval_s_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::vector<GpuSample> samples_;
  bool stop_ = true;
  std::thread thread_;
};

// True once the SM clock has stopped trending: the mean of the last `window`
// samples is within `tol` of the mean of the `window` samples before them.
// Means are compared rather than requiring every sample to agree because a
// power-capped part dithers by several percent around a steady operating point
// indefinitely; a per-sample criterion would never be met there, while a decay
// from a boost clock still shows up as a trend in the means.
inline bool clock_settled(const std::vector<GpuSample>& v, size_t window,
                          double tol) {
  if (window == 0 || v.size() < 2 * window) return false;
  double earlier = 0.0, recent = 0.0;
  for (size_t i = v.size() - 2 * window; i < v.size() - window; ++i)
    earlier += v[i].sm_mhz;
  for (size_t i = v.size() - window; i < v.size(); ++i) recent += v[i].sm_mhz;
  earlier /= static_cast<double>(window);
  recent /= static_cast<double>(window);
  return recent > 0.0 && std::fabs(recent - earlier) <= tol * recent;
}

// Mean SM clock over the last `window` samples and its peak-to-peak range as a
// percentage of that mean. The range is the clock dither that per-iteration
// timings inherit, so it is recorded next to them.
struct ClockWindow {
  double mean_mhz = -1.0;
  double range_pct = -1.0;
};

inline ClockWindow clock_window(const std::vector<GpuSample>& v,
                                size_t window) {
  ClockWindow w;
  const size_t n = std::min(window, v.size());
  if (n == 0) return w;
  int lo = INT_MAX, hi = 0;
  double sum = 0.0;
  for (size_t i = v.size() - n; i < v.size(); ++i) {
    lo = std::min(lo, v[i].sm_mhz);
    hi = std::max(hi, v[i].sm_mhz);
    sum += v[i].sm_mhz;
  }
  w.mean_mhz = sum / static_cast<double>(n);
  if (w.mean_mhz > 0.0) w.range_pct = 100.0 * (hi - lo) / w.mean_mhz;
  return w;
}
