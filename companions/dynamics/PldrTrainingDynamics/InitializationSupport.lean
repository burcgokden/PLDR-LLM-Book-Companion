/- Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.
Scalar checks of the normalized rank-three uniform tensor law. These do
not verify a random number generator or an infinite-width limit. -/
import Mathlib
namespace PldrTrainingDynamics.InitializationSupport

theorem variance_rescaling {N d : ℝ} (hN : 1 ≤ N) (hd : 2 ≤ d) :
    (Real.sqrt ((d+N)/(d+2)))^2 * (2/(d*(d+N))) = 2/(d*(d+2)) := by
  rw [Real.sq_sqrt (by positivity)]
  field_simp

theorem normalized_support_le_one {d : ℝ} (hd : 2 ≤ d) :
    Real.sqrt (6/(d*(d+2))) ≤ 1 := by
  apply (Real.sqrt_le_one).2
  apply (div_le_iff₀ (by positivity : 0 < d*(d+2))).2
  nlinarith

theorem uniform_support_rescaling {N d : ℝ} (hN : 1 ≤ N) (hd : 2 ≤ d) :
    Real.sqrt ((d+N)/(d+2)) * Real.sqrt (6/(d*(d+N))) =
      Real.sqrt (6/(d*(d+2))) := by
  rw [← Real.sqrt_mul (by positivity)]
  congr 1
  field_simp
end PldrTrainingDynamics.InitializationSupport
