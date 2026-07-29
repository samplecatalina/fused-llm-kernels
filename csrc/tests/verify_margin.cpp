// Margin test for the correctness tolerance in common/verify.h.
//
// Builds full-size (4096^2-element) synthetic results with the RMS a K-term
// sum of U(-1,1) products has (sqrt(K)/3), then checks two things:
//   - rounding noise of 2e-6 of the RMS, the largest difference observed
//     between two correct implementations that sum in a different order,
//     passes with margin;
//   - a single wrong A*B term (0.25) is rejected, both on the element nearest
//     zero and on the largest element, where the RMS floor plays no part.
// Exits non-zero if either property fails at any size.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../common/verify.h"

int main() {
  constexpr double kTolMaxRel = 1e-3, kTolFroRel = 1e-5;
  std::mt19937_64 gen(11);
  int wrong = 0;
  for (int K : {512, 1024, 4096, 8192}) {
    const size_t n = 4096ull * 4096ull;
    const double rms = std::sqrt(static_cast<double>(K)) / 3.0;
    std::normal_distribution<double> ref_dist(0.0, rms), noise(0.0, 2e-6 * rms);
    std::vector<float> ref(n), noisy(n);
    for (size_t i = 0; i < n; ++i) {
      ref[i] = static_cast<float>(ref_dist(gen));
      noisy[i] = ref[i] + static_cast<float>(noise(gen));
    }
    size_t lo = 0, hi = 0;
    for (size_t i = 1; i < n; ++i) {
      if (std::fabs(ref[i]) < std::fabs(ref[lo])) lo = i;
      if (std::fabs(ref[i]) > std::fabs(ref[hi])) hi = i;
    }
    const VerifyResult rn = verify(ref, noisy);
    std::vector<float> bug = ref;
    bug[lo] += 0.25f;
    const VerifyResult rlo = verify(ref, bug);
    bug = ref;
    bug[hi] += 0.25f;
    const VerifyResult rhi = verify(ref, bug);

    const bool ok = rn.passed(kTolMaxRel, kTolFroRel) &&
                    !rlo.passed(kTolMaxRel, kTolFroRel) &&
                    !rhi.passed(kTolMaxRel, kTolFroRel);
    std::printf("K=%5d rms=%6.2f  noise max_rel %.2e (%.1fx under)  "
                "wrong term: near zero %.2e, largest element %.2e (%.1fx over)  %s\n",
                K, rms, rn.max_rel, kTolMaxRel / rn.max_rel, rlo.max_rel,
                rhi.max_rel, rhi.max_rel / kTolMaxRel, ok ? "ok" : "WRONG");
    wrong += !ok;
  }
  return wrong;
}
