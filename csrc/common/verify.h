#pragma once
#include <cmath>
#include <cstddef>
#include <vector>

// Two tolerances, both of which must hold:
//   max_rel - largest element-wise relative error. Catches "one element is
//             completely wrong", i.e. indexing and boundary bugs.
//   fro_rel - relative Frobenius-norm error. Catches a systematic drift across
//             the whole matrix.
// Neither is sufficient alone: max_rel is inflated by elements near zero, and
// fro_rel averages away a handful of badly wrong elements - which is exactly
// what a boundary bug produces.
struct VerifyResult {
  double max_rel = 0.0;
  double fro_rel = 0.0;
  size_t worst_index = 0;
  double worst_ref = 0.0;
  double worst_got = 0.0;
  bool passed(double max_tol, double fro_tol) const {
    return std::isfinite(max_rel) && std::isfinite(fro_rel) &&
           max_rel < max_tol && fro_rel < fro_tol;
  }
};

inline VerifyResult verify(const std::vector<float>& ref,
                           const std::vector<float>& got) {
  VerifyResult r;
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double a = ref[i], b = got[i];
    const double d = a - b;
    num += d * d;
    den += a * a;
    const double rel = std::fabs(d) / (std::fabs(a) + 1e-5);
    if (!(rel <= r.max_rel)) {  // written this way so NaN also trips it
      r.max_rel = rel;
      r.worst_index = i;
      r.worst_ref = a;
      r.worst_got = b;
    }
  }
  r.fro_rel = (den > 0.0) ? std::sqrt(num) / std::sqrt(den) : std::sqrt(num);
  return r;
}
