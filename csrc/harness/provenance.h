#pragma once
// Provenance helpers shared by every program that writes result rows: which
// commit produced a row, when, and on which GPU. A row that cannot be traced
// back to committed code is not written at all.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <string>

#include <cuda_runtime.h>

inline std::string shell_line(const char* cmd) {
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
// rules differ from it.
inline std::string source_revision() {
  std::string rev = shell_line("git rev-parse --short=12 HEAD 2>/dev/null");
  if (rev.empty()) return "unknown";
  if (!shell_line("git status --porcelain -- csrc Makefile 2>/dev/null").empty())
    rev += "-dirty";
  return rev;
}

inline bool is_untraceable(const std::string& rev) {
  return rev == "unknown" ||
         (rev.size() > 6 && rev.compare(rev.size() - 6, 6, "-dirty") == 0);
}

inline std::string utc_timestamp() {
  std::time_t now = std::time(nullptr);
  std::tm tm_utc{};
  gmtime_r(&now, &tm_utc);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
  return buf;
}

// nvidia-smi selector for the device CUDA is actually running on.
inline std::string gpu_selector(const cudaDeviceProp& p) {
  const unsigned char* u = reinterpret_cast<const unsigned char*>(p.uuid.bytes);
  char buf[48];
  std::snprintf(buf, sizeof buf,
                "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
                "%02x%02x%02x%02x%02x%02x",
                u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9],
                u[10], u[11], u[12], u[13], u[14], u[15]);
  return buf;
}

inline bool valid_device_tag(const std::string& t) {
  if (t.empty()) return false;
  for (char c : t)
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-'))
      return false;
  return true;
}

inline std::string csv_safe(std::string s) {
  std::replace(s.begin(), s.end(), ',', ';');
  return s;
}

inline std::string fmt_int(long long v) { return v < 0 ? "" : std::to_string(v); }

inline std::string fmt_num(double v, const char* f) {
  if (!std::isfinite(v) || v < 0.0) return "";
  char buf[64];
  std::snprintf(buf, sizeof buf, f, v);
  return buf;
}
