/- Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.
Matrix identities for the fixed rotary family. No multi-plane dimension
count or full-decoder expressivity claim is encoded here. -/
import PldrLlmMathFoundations.Rope

namespace PldrLlm
open Matrix

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The chosen projections A = Gᵀ and B = 1 satisfy every score-matrix identity. -/
theorem rope_chosen_absorption (R : ℤ → Matrix ι ι ℝ)
    (hmul : ∀ n m, R (n + m) = R n * R m)
    (G : Matrix ι ι ℝ) (hG : ∀ n, G * R n = R n * G) (n m : ℤ) :
    G * R (m - n) * 1 = R (-n) * G * R m := by
  rw [Matrix.mul_one, ← hG (-n), Matrix.mul_assoc, ← hmul]
  congr 2
  omega

/-- Equality of all score matrices forces commutation; no invertibility
of either projection is assumed. The rotary group law supplies the inverse. -/
theorem rope_absorption_necessary (R : ℤ → Matrix ι ι ℝ)
    (hzero : R 0 = 1) (hmul : ∀ n m, R (n + m) = R n * R m)
    (A B G : Matrix ι ι ℝ)
    (h : ∀ n m, Aᵀ * R (m - n) * B = R (-n) * G * R m) :
    ∀ n, G * R n = R n * G := by
  have hprod : Aᵀ * B = G := by simpa [hzero] using h 0 0
  intro n
  have hd : G = R (-n) * G * R n := by
    simpa [hzero, hprod] using h n n
  calc
    G * R n = (R n * R (-n)) * G * R n := by
      rw [← hmul, add_neg_cancel, hzero, Matrix.one_mul]
    _ = R n * (R (-n) * G * R n) := by simp only [Matrix.mul_assoc]
    _ = R n * G := by rw [← hd]

def ropeD : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 1]
noncomputable def ropeDInv : Matrix (Fin 2) (Fin 2) ℝ := !![1/2, 0; 0, 1]

theorem rope_factor_product : ropeD * ropeDInv = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [ropeD, ropeDInv, Matrix.mul_apply, Fin.sum_univ_two]

theorem rope_factor_fails : ropeD * rotJ * ropeDInv ≠ rotJ := by
  intro h
  have h01 := congrFun (congrFun h 0) 1
  norm_num [ropeD, ropeDInv, rotJ, Matrix.mul_apply, Fin.sum_univ_two] at h01

/-- Sum of squared entries, hence the squared Frobenius discrepancy. -/
theorem rope_factor_error_sq :
    ∑ i : Fin 2, ∑ j : Fin 2, ((ropeD * rotJ * ropeDInv - rotJ) i j)^2 = 5/4 := by
  norm_num [ropeD, ropeDInv, rotJ, Matrix.mul_apply, Fin.sum_univ_two]
end PldrLlm
