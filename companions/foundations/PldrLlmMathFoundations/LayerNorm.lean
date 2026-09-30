/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# LayerNorm identities and conditional target transfer

The implemented row map is
`LNε(r) = γ ⊙ (r - mean r) / sqrt (var r + ε) + β`, with positive offset.
This module checks the following selected clauses of the LayerNorm analysis:

* `PldrLlm.rowVar_nonneg`: nonnegative row variance;
* `PldrLlm.lnEps_shift`: exact common-shift invariance;
* `PldrLlm.lnEps_scale_eps_zero`: positive-scale invariance at zero offset
  on positive-variance rows;
* `PldrLlm.sum_sq_normalized`: the exact squared-norm identity
  `d * var r / (var r + ε)` for positive offset;
* `PldrLlm.sum_sq_normalized_lt_dim`: the strict squared-norm bound
  for positive dimension and positive offset;
* `PldrLlm.lnEps_scale_offset`: exact positive-scale covariance when the
  offset changes from `ε` to `ε / c^2`;
* `PldrLlm.lnEps_constant`: a constant row gives the affine bias;
* `PldrLlm.ln_target_transfer`: a Euclidean triangle estimate that assumes
  a `LipschitzWith` premise and retains the scale discrepancy.

For positive dimension and offset the normalized image is the relative open
ball of radius sqrt(d) in the centered subspace. The closed ball in that
subspace is compact; its continuous affine image for fixed gain and bias is
a compact containing set. The actual affine image need not be compact (unit
gain, zero bias, d >= 2), but can be a singleton (d = 1 or zero gain).
These topological, image and surjectivity statements, and the radial Lipschitz
argument, are written mathematics outside the scalar checks in this module.
The target-transfer theorem does not derive the LayerNorm Lipschitz constant,
identify the normalized-Gram target, or prove statistical concentration.
Matrix Bernstein, common pre-rotation query moments and independence remain
written hypotheses and arguments. Book numbers belong to the correspondence.
-/
import Mathlib

namespace PldrLlm

open Finset

variable {d : ℕ}

/-- Mean of the entries of a row `r : Fin d → ℝ`. -/
noncomputable def rowMean (r : Fin d → ℝ) : ℝ := (∑ i, r i) / d

/-- Biased variance of the entries of a row. -/
noncomputable def rowVar (r : Fin d → ℝ) : ℝ :=
  (∑ i, (r i - rowMean r) ^ 2) / d

/-- The ε-regularized LayerNorm row map with learned affine parameters. -/
noncomputable def lnEps (ε : ℝ) (γ β : Fin d → ℝ) (r : Fin d → ℝ) :
    Fin d → ℝ :=
  fun i => γ i * ((r i - rowMean r) / Real.sqrt (rowVar r + ε)) + β i

lemma rowVar_nonneg (r : Fin d → ℝ) : 0 ≤ rowVar r :=
  div_nonneg (Finset.sum_nonneg fun _ _ => sq_nonneg _) (Nat.cast_nonneg d)

lemma rowMean_shift [NeZero d] (r : Fin d → ℝ) (c : ℝ) :
    rowMean (fun i => r i + c) = rowMean r + c := by
  have hd : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne d)
  unfold rowMean
  rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, add_div, nsmul_eq_mul, mul_comm,
    mul_div_assoc, div_self hd, mul_one]

lemma rowVar_shift [NeZero d] (r : Fin d → ℝ) (c : ℝ) :
    rowVar (fun i => r i + c) = rowVar r := by
  unfold rowVar
  rw [rowMean_shift]
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  ring_nf

/-- Exact shift invariance of the implemented (ε-regularized) LayerNorm:
valid also at positive offset. Fixed-offset scale invariance generally fails. -/
theorem lnEps_shift [NeZero d] (ε : ℝ) (γ β : Fin d → ℝ)
    (r : Fin d → ℝ) (c : ℝ) :
    lnEps ε γ β (fun i => r i + c) = lnEps ε γ β r := by
  funext i
  unfold lnEps
  rw [rowMean_shift, rowVar_shift]
  ring_nf

lemma rowMean_smul [NeZero d] (r : Fin d → ℝ) (c : ℝ) :
    rowMean (fun i => c * r i) = c * rowMean r := by
  unfold rowMean
  rw [← Finset.mul_sum, mul_div_assoc]

lemma rowVar_smul [NeZero d] (r : Fin d → ℝ) (c : ℝ) :
    rowVar (fun i => c * r i) = c ^ 2 * rowVar r := by
  unfold rowVar
  rw [rowMean_smul, ← mul_div_assoc]
  congr 1
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  ring

/-- Exact positive-scale invariance holds for the idealized `ε = 0` map
on rows of positive variance.  For the implemented `ε = 10⁻⁶ > 0` it
generally fails (the written argument quantifies the error); this lemma isolates what
the idealization provides. -/
theorem lnEps_scale_eps_zero [NeZero d] (γ β : Fin d → ℝ)
    (r : Fin d → ℝ) {c : ℝ} (hc : 0 < c) (hv : 0 < rowVar r) :
    lnEps 0 γ β (fun i => c * r i) = lnEps 0 γ β r := by
  funext i
  unfold lnEps
  rw [rowMean_smul, rowVar_smul, add_zero, add_zero]
  have hsq : Real.sqrt (c ^ 2 * rowVar r) = c * Real.sqrt (rowVar r) := by
    rw [Real.sqrt_mul (by positivity), Real.sqrt_sq hc.le]
  rw [hsq]
  have hs : Real.sqrt (rowVar r) ≠ 0 := by
    positivity
  have : c * r i - c * rowMean r = c * (r i - rowMean r) := by ring
  rw [this, mul_div_mul_left _ _ (ne_of_gt hc)]

/-- Exact norm identity for the normalized vector of the implemented
map: `Σ ((r i - mean)/sqrt (var + ε))² = d·var/(var + ε)`.  For
`ε > 0` the right side is `< d`, a relative open-ball bound. Its affine
closed-ball enclosure uses fixed gain and bias. Positive variance can
approach the sphere as `var/ε → ∞`; dimension one has only zero variance. -/
theorem sum_sq_normalized [NeZero d] (r : Fin d → ℝ) {ε : ℝ}
    (hε : 0 < ε) :
    ∑ i, ((r i - rowMean r) / Real.sqrt (rowVar r + ε)) ^ 2
      = d * rowVar r / (rowVar r + ε) := by
  have hpos : 0 < rowVar r + ε := by
    have := rowVar_nonneg r
    linarith
  have hd : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne d)
  have hsum : ∑ i, (r i - rowMean r) ^ 2 = d * rowVar r := by
    unfold rowVar
    field_simp
  calc ∑ i, ((r i - rowMean r) / Real.sqrt (rowVar r + ε)) ^ 2
      = ∑ i, (r i - rowMean r) ^ 2 / (rowVar r + ε) := by
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [div_pow, Real.sq_sqrt hpos.le]
    _ = (∑ i, (r i - rowMean r) ^ 2) / (rowVar r + ε) := by
        rw [Finset.sum_div]
    _ = d * rowVar r / (rowVar r + ε) := by rw [hsum]


/-- Positive offset keeps the squared normalized norm strictly below the dimension.
This is a scalar bound, not a formalization of the topological image. -/
theorem sum_sq_normalized_lt_dim [NeZero d] (r : Fin d → ℝ) {ε : ℝ}
    (hε : 0 < ε) :
    ∑ i, ((r i - rowMean r) / Real.sqrt (rowVar r + ε)) ^ 2 < (d : ℝ) := by
  rw [sum_sq_normalized r hε]
  have hd : 0 < (d : ℝ) := Nat.cast_pos.mpr (Nat.pos_of_ne_zero (NeZero.ne d))
  have hv : 0 ≤ rowVar r := rowVar_nonneg r
  have hpos : 0 < rowVar r + ε := by linarith
  apply (div_lt_iff₀ hpos).2
  nlinarith [mul_pos hd hε]


/-- Exact scale covariance of the positive-offset normalization. -/
theorem lnEps_scale_offset [NeZero d] (γ β : Fin d → ℝ)
    (r : Fin d → ℝ) {c ε : ℝ} (hc : 0 < c) (_hε : 0 < ε) :
    lnEps ε γ β (fun i => c * r i) = lnEps (ε / c^2) γ β r := by
  funext i
  unfold lnEps
  rw [rowMean_smul, rowVar_smul]
  have hid : c^2 * rowVar r + ε = c^2 * (rowVar r + ε / c^2) := by
    field_simp
  have hs : Real.sqrt (c^2 * rowVar r + ε) =
      c * Real.sqrt (rowVar r + ε / c^2) := by
    rw [hid, Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq hc.le]
  rw [hs, show c * r i - c * rowMean r = c * (r i - rowMean r) by ring,
    mul_div_mul_left _ _ hc.ne']

/-- Every exactly constant row has affine output β, including at zero offset. -/
theorem lnEps_constant [NeZero d] (ε b : ℝ) (γ β : Fin d → ℝ) :
    lnEps ε γ β (fun _ => b) = β := by
  funext i
  simp [lnEps, rowMean, NeZero.ne d]

/-- Euclidean target-transfer kernel. The Lipschitz premise is supplied;
this lemma does not derive it or a matrix concentration inequality. -/
theorem ln_target_transfer {d : ℕ} (f : EuclideanSpace ℝ (Fin d) → EuclideanSpace ℝ (Fin d))
    (L : NNReal) (hf : LipschitzWith L f) (u r m : EuclideanSpace ℝ (Fin d)) :
    ‖u - f m‖ ≤ ‖u - f r‖ + (L : ℝ) * ‖r - m‖ := by
  calc
    ‖u - f m‖ = ‖(u - f r) + (f r - f m)‖ := by congr 1; abel
    _ ≤ ‖u - f r‖ + ‖f r - f m‖ := norm_add_le _ _
    _ ≤ ‖u - f r‖ + (L : ℝ) * ‖r - m‖ := by
      exact add_le_add_right (hf.norm_sub_le r m) _

end PldrLlm
