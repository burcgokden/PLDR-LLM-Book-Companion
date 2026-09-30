/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# Deterministic analytic core of the quantitative RoPE twirl

In a torus eigenbasis, a fixed matrix's off-commutant entries receive
geometric position sums at nonzero difference frequencies. This file checks:

* `PldrLlm.norm_one_sub_exp`: the chord identity
  `‖1 - exp(iω)‖ = 2 |sin(ω/2)|`;
* `PldrLlm.geom_orbit_bound`: the geometric-sum bound
  `‖Σ_{n=1}^S exp(iωn)‖ ≤ 1 / |sin(ω/2)|` for `sin(ω/2) ≠ 0`.

For positive S, division by S gives the scalar O(1/S) remainder.
The matrix-level commutant projection assembly, multi-plane nonresonance,
frequency-resolved constants and stochastic application are not formalized
here. In particular this module does not prove concentration of the normalized
query Gram or a LayerNorm input law. Book numbers belong to the correspondence.
-/
import Mathlib

namespace PldrLlm

open Complex Finset

/-- Chord length of the unit circle: `‖1 − e^{iω}‖ = 2 |sin (ω/2)|`. -/
theorem norm_one_sub_exp (ω : ℝ) :
    ‖1 - Complex.exp (ω * Complex.I)‖ = 2 * |Real.sin (ω / 2)| := by
  have hz : 1 - Complex.exp (ω * Complex.I) =
      Complex.ofReal (1 - Real.cos ω) +
        Complex.ofReal (-Real.sin ω) * Complex.I := by
    rw [Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin]
    push_cast
    ring
  have h1 : Real.sin (ω / 2) ^ 2 = 1 / 2 - Real.cos ω / 2 := by
    have h := Real.sin_sq_eq_half_sub (ω / 2)
    rwa [show 2 * (ω / 2) = ω by ring] at h
  have h2 : Real.sin ω ^ 2 + Real.cos ω ^ 2 = 1 :=
    Real.sin_sq_add_cos_sq ω
  have habs : |Real.sin (ω / 2)| ^ 2 = Real.sin (ω / 2) ^ 2 :=
    sq_abs _
  have hs : (1 - Real.cos ω) ^ 2 + (-Real.sin ω) ^ 2 =
      (2 * |Real.sin (ω / 2)|) ^ 2 := by
    linear_combination h2 - 4 * habs - 4 * h1
  rw [hz, Complex.norm_add_mul_I, hs,
    Real.sqrt_sq (by positivity)]

/-- Bound for the ergodic (geometric) position sum at a nonzero
frequency: `‖Σ_{n=1}^S e^{iωn}‖ ≤ 1 / |sin (ω/2)|`.  Dividing by `S`
gives the `O(1/S)` suppression of every off-commutant component of the
fixed-matrix position average, after the written matrix assembly. -/
theorem geom_orbit_bound {ω : ℝ} (hω : Real.sin (ω / 2) ≠ 0) (S : ℕ) :
    ‖∑ n ∈ Finset.range S, Complex.exp (ω * Complex.I) ^ (n + 1)‖ ≤
      1 / |Real.sin (ω / 2)| := by
  set z := Complex.exp (ω * Complex.I) with hzdef
  have habs : (0 : ℝ) < |Real.sin (ω / 2)| := abs_pos.mpr hω
  have hz1 : z ≠ 1 := by
    intro h
    have h0 : ‖(1 : ℂ) - z‖ = 0 := by rw [h]; simp
    rw [hzdef, norm_one_sub_exp] at h0
    linarith
  have hnorm : ‖z‖ = 1 := by
    rw [hzdef]; exact Complex.norm_exp_ofReal_mul_I ω
  have hgeom : ∑ n ∈ Finset.range S, z ^ (n + 1) =
      z * ((z ^ S - 1) / (z - 1)) := by
    rw [← geom_sum_eq hz1 S, Finset.mul_sum]
    exact Finset.sum_congr rfl fun n _ => pow_succ' z n
  have hub : ‖z ^ S - 1‖ ≤ 2 := by
    calc ‖z ^ S - 1‖ ≤ ‖z ^ S‖ + ‖(1 : ℂ)‖ := norm_sub_le _ _
      _ = 2 := by rw [norm_pow, hnorm, one_pow, norm_one]; norm_num
  have hden : ‖z - 1‖ = 2 * |Real.sin (ω / 2)| := by
    rw [← norm_neg, neg_sub, hzdef]
    exact norm_one_sub_exp ω
  have h2 : (0 : ℝ) < 2 * |Real.sin (ω / 2)| := by linarith
  rw [hgeom, norm_mul, hnorm, one_mul, norm_div, hden]
  calc ‖z ^ S - 1‖ / (2 * |Real.sin (ω / 2)|)
      ≤ 2 / (2 * |Real.sin (ω / 2)|) := by gcongr
    _ = 1 / |Real.sin (ω / 2)| := by
        field_simp

end PldrLlm
