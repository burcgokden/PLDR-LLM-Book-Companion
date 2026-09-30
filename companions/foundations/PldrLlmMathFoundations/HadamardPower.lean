/-
Copyright 2026 Burc Gokden. Released under the Apache 2.0 license.

# The elementwise (Hadamard) power law A ↦ A^{⊙P}

The potential tensor of PLGA is `A_P = A_LM^{⊙P}`, the elementwise power
with a learned exponent matrix `P`.  For entrywise-positive `A` this file
checks entrywise positivity and the one-parameter group law in the book:

* `hadPow_pos` : a positive base gives strictly positive real powers;
* `hadPow_zero` : zero exponents give the all-ones matrix;
* `hadPow_add`  : `A^{⊙(s+t)P} = A^{⊙sP} ⊙ A^{⊙tP}` (Hadamard product),
  i.e. `t ↦ A^{⊙tP}` is a multiplicative one-parameter group, the
  discrete-time form of the linear log-space flow described in the book.

The architectural base is `A_LM`. Positivity of `A_P` gives no common
entry floor over unrestricted exponents, no positive-definiteness claim,
and no positivity claim for the signed final operator `G = a A_P + b_a`
with unrestricted real matrix coupling and bias.
-/
import Mathlib

namespace PldrLlm

open Matrix

variable {d : ℕ}

/-- Elementwise (Hadamard) real power: `(hadPow A P) i j = (A i j) ^ (P i j)`
with `x ^ y` the real power `Real.rpow`. -/
noncomputable def hadPow (A P : Matrix (Fin d) (Fin d) ℝ) :
    Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun i j => A i j ^ P i j

@[simp]
lemma hadPow_apply (A P : Matrix (Fin d) (Fin d) ℝ) (i j : Fin d) :
    hadPow A P i j = A i j ^ P i j := rfl

/-- Positive real powers of a positive base, entry by entry. This does
not assert an exponent-independent floor or positivity of the final score. -/
theorem hadPow_pos (A P : Matrix (Fin d) (Fin d) ℝ)
    (hA : ∀ i j, 0 < A i j) : ∀ i j, 0 < hadPow A P i j := by
  intro i j
  exact Real.rpow_pos_of_pos (hA i j) (P i j)

/-- Zero exponent matrix gives the all-ones matrix (the identity of the
Hadamard product). -/
theorem hadPow_zero (A : Matrix (Fin d) (Fin d) ℝ) :
    hadPow A 0 = Matrix.of fun _ _ => (1 : ℝ) := by
  ext i j
  simp [hadPow]

/-- One-parameter multiplicative group law of the power law stage:
for entrywise-positive `A`,
`A^{⊙(s+t)P} = A^{⊙sP} ⊙ A^{⊙tP}` entrywise. -/
theorem hadPow_add (A P : Matrix (Fin d) (Fin d) ℝ)
    (hA : ∀ i j, 0 < A i j) (s t : ℝ) :
    hadPow A ((s + t) • P) =
      (hadPow A (s • P)).hadamard (hadPow A (t • P)) := by
  ext i j
  simp only [hadPow_apply, Matrix.hadamard_apply, Matrix.smul_apply,
    smul_eq_mul, add_mul]
  exact Real.rpow_add (hA i j) _ _

end PldrLlm
