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
//
// max_rel divides by |ref| plus a floor of 5% of the RMS of ref. A fixed
// absolute floor does not work: two correct implementations that sum in a
// different order differ by floating-point rounding of up to about 2e-6 of
// the RMS, and on an element close to zero that rounding reads as a large
// relative error (at 1024^3 a fixed 1e-5 floor rejected every kernel here).
// Measured on 4096^2-element synthetic matrices, the 5% floor keeps rounding
// noise 5x under the tolerance at every size, and a floor of 1% did not
// (1.1x). A single wrong A*B term (about 0.25 for these inputs) is still
// rejected: by orders of magnitude on small elements, and by 1.5x even on the
// single largest element of an 8192^3 result, where the floor plays no part.
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

constexpr double kMaxRelFloorOfRms = 5e-2;

inline VerifyResult verify(const std::vector<float>& ref,
                           const std::vector<float>& got) {
  VerifyResult r;
  double den = 0.0;
  for (float a : ref) den += static_cast<double>(a) * a;
  const double rms = ref.empty() ? 0.0 : std::sqrt(den / ref.size());
  const double floor = rms > 0.0 ? kMaxRelFloorOfRms * rms : 1e-5;

  double num = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double a = ref[i], b = got[i];
    const double d = a - b;
    num += d * d;
    const double rel = std::fabs(d) / (std::fabs(a) + floor);
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
