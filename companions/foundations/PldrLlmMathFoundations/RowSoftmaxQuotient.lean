/- Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.
Finite real logits, positive row/column counts, and ordinary row softmax.
Centering is defined entrywise with the exact arithmetic-mean convention.
No Jacobian bound, variable causal support, or decoder transfer is formalized. -/
import PldrLlmMathFoundations.Softmax
import PldrLlmMathFoundations.LayerNorm

namespace PldrLlm
open Matrix Finset
variable {r c : ℕ} [NeZero r] [NeZero c]

def RowsEqual (Z : Matrix (Fin r) (Fin c) ℝ) : Prop := ∀ i k, Z i = Z k
noncomputable def rowSoftmax (Z : Matrix (Fin r) (Fin c) ℝ) : Matrix (Fin r) (Fin c) ℝ := fun i => softmax (Z i)
noncomputable def centerCols (Z : Matrix (Fin r) (Fin c) ℝ) : Matrix (Fin r) (Fin c) ℝ :=
  fun i j => Z i j - rowMean (Z i)
noncomputable def centerRows (Z : Matrix (Fin r) (Fin c) ℝ) : Matrix (Fin r) (Fin c) ℝ :=
  fun i j => Z i j - (∑ k, Z k j) / r

/-- The softmax rows coincide exactly when logits are a common row plus row shifts. -/
theorem rowSoftmax_eq_iff_shifts (Z : Matrix (Fin r) (Fin c) ℝ) :
    RowsEqual (rowSoftmax Z) ↔
      ∃ (v : Fin c → ℝ) (α : Fin r → ℝ), ∀ i j, Z i j = v j + α i := by
  classical
  constructor
  · intro h
    have hs : ∀ i, ∃ a : ℝ, ∀ j, Z i j = Z 0 j + a := fun i =>
      (softmax_eq_iff_shift (Z 0) (Z i)).mp (h 0 i)
    choose α hα using hs
    exact ⟨Z 0, α, hα⟩
  · rintro ⟨v, α, h⟩ i k
    change softmax (Z i) = softmax (Z k)
    have hi : Z i = fun j => v j + α i := funext (h i)
    have hk : Z k = fun j => v j + α k := funext (h k)
    rw [hi, hk, softmax_shift, softmax_shift]

omit [NeZero c] in
theorem rowsEqual_iff_centerRows_zero (Z : Matrix (Fin r) (Fin c) ℝ) :
    RowsEqual Z ↔ centerRows Z = 0 := by
  constructor
  · intro h
    ext i j
    have hs : (∑ k, Z k j) = (r : ℝ) * Z i j := by
      simp only [show (fun k => Z k j) = (fun _ => Z i j) from
        funext (fun k => congrFun (h k i) j)]
      simp
    simp [centerRows, hs, NeZero.ne r]
  · intro h i k
    funext j
    have hi := congrFun (congrFun h i) j
    have hk := congrFun (congrFun h k) j
    simp only [centerRows, Matrix.zero_apply] at hi hk
    linarith

theorem centeredRows_iff_shifts (Z : Matrix (Fin r) (Fin c) ℝ) :
    RowsEqual (centerCols Z) ↔
      ∃ (v : Fin c → ℝ) (α : Fin r → ℝ), ∀ i j, Z i j = v j + α i := by
  constructor
  · intro h
    refine ⟨Z 0, fun i => rowMean (Z i) - rowMean (Z 0), ?_⟩
    intro i j
    have hij := congrFun (h i 0) j
    simp only [centerCols] at hij
    linarith
  · rintro ⟨v, α, h⟩ i k
    have hi : Z i = fun j => v j + α i := funext (h i)
    have hk : Z k = fun j => v j + α k := funext (h k)
    funext j
    simp only [centerCols, hi, hk, rowMean_shift]
    ring

theorem rowSoftmax_eq_iff_doubleCenter (Z : Matrix (Fin r) (Fin c) ℝ) :
    centerRows (rowSoftmax Z) = 0 ↔ centerRows (centerCols Z) = 0 := by
  rw [← rowsEqual_iff_centerRows_zero, ← rowsEqual_iff_centerRows_zero,
    rowSoftmax_eq_iff_shifts, centeredRows_iff_shifts]

noncomputable def centering (n : ℕ) : Matrix (Fin n) (Fin n) ℝ :=
  1 - (1 / (n : ℝ)) • (Matrix.of fun (_ : Fin n) (_ : Fin n) => (1 : ℝ))

omit [NeZero r] [NeZero c] in
theorem centerRows_matrix (Z : Matrix (Fin r) (Fin c) ℝ) :
    centerRows Z = centering r * Z := by
  unfold centering
  rw [Matrix.sub_mul, Matrix.one_mul, Matrix.smul_mul]
  ext i j
  simp [Matrix.mul_apply, centerRows, div_eq_mul_inv, mul_comm]

omit [NeZero r] [NeZero c] in
theorem centerCols_matrix (Z : Matrix (Fin r) (Fin c) ℝ) :
    centerCols Z = Z * centering c := by
  unfold centering
  rw [Matrix.mul_sub, Matrix.mul_one, Matrix.mul_smul]
  ext i j
  simp [Matrix.mul_apply, centerCols, rowMean, div_eq_mul_inv, mul_comm]

/-- Literal matrix form C_r S(Z)=0 iff C_r Z C_c=0. -/
theorem rowSoftmax_matrix_criterion (Z : Matrix (Fin r) (Fin c) ℝ) :
    centering r * rowSoftmax Z = 0 ↔ centering r * Z * centering c = 0 := by
  simpa only [centerRows_matrix, centerCols_matrix, Matrix.mul_assoc] using
    rowSoftmax_eq_iff_doubleCenter Z

def softmaxErasure : Matrix (Fin 2) (Fin 2) ℝ := !![1, 1; 2, 2]

theorem softmaxErasure_centered_energy :
    ∑ i : Fin 2, ∑ j : Fin 2, (centerRows softmaxErasure i j)^2 = 1 := by
  norm_num [centerRows, softmaxErasure, Fin.sum_univ_two]

theorem softmaxErasure_uniform : rowSoftmax softmaxErasure = fun _ _ => (1/2 : ℝ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [rowSoftmax, softmaxErasure, softmax, Fin.sum_univ_two] <;>
    field_simp <;> ring
end PldrLlm
