# Formal correspondence

The book contains standalone mathematical proofs. Lean supplies independent checks
of the specific clauses listed in Appendix D and `provenance/book-correspondence.json`.
Appendix D groups the presentation by proof family, with separate checked-scope
and remaining-obligation columns. A shortened printed declaration list does not
change coverage: the complete book-owned map is the authoritative declaration
inventory, and the unified checker resolves its full types. Table reflow adds
no new formal theorem. The complete included source mappings remain in
`companions/dynamics/provenance/statement-manifest.json` and
`companions/foundations/README.md`. Their source-work numbers are provenance;
book labels and compiled numbering belong to the book-owned map.

For navigation in the source monograph, the included Dynamics manifest also
records `published_anchor` and `published_pdf_page` bound to the exact arXiv
v1 PDF. The original-build anchor metadata remains available for provenance.
These fields identify source-monograph statements; the book-owned map keeps
its own statement identities. See the included
[publication guide](../companions/dynamics/docs/CHAPTERS.md#published-pdf-destinations)
for PDF acquisition and the optional parser dependency. From this repository root:

```sh
python3 companions/dynamics/scripts/check_published_pdf.py --pdf PDF --data-repo DATA-REPO --output validation/monograph-pdf.json
```

`RankOne.lean` assumes identical rows and checks finite real matrix algebra;
it does not assert that every trained model reaches that configuration.
`rowClone_mulVec` gives the action on an arbitrary vector.
`rowClone_range_eq_span` uses a nonzero coordinate to prove the full column
range is the real span of ones; `rowClone_range_zero` gives the zero-profile
range. `rowClone_range_eq_span_iff` explicitly assumes positive dimension,
since the all-ones vector is zero in empty dimension. Existing rank, zero,
determinant and power declarations retain their types. The characteristic
polynomial, spectral-radius interpretation and singular-value perturbation
bound remain written obligations. The implication from A² = s A to
idempotency of A/s requires s != 0.

`Iswiglu.lean` checks the base floor and strict positivity at positive offset.
`HadamardPower.lean` additionally checks `PldrLlm.hadPow_pos`: every entry of
a positive base raised to any real exponent stays positive in exact real
arithmetic. This proves no common entry floor over unrestricted exponents,
positive definiteness, or numerical underflow guarantee. The final coupling
and bias are unrestricted real matrices. The book's exact choice a = 0,
bias = -I gives G = -I by a standalone written calculation; positive-base
lemmas do not formalize that signed-score witness.

The matrix-softmax statement uses literal left/right centering matrices and
positive row/column counts. It does not formalize the Jacobian operator bound or
different causal supports. RoPE absorption checks a normalized integer rotation
group, a chosen projection pair, and necessity from equality of all score
matrices. The written criterion quantifies over all query/key vectors and
positions with position-independent factors and the prescribed orthogonal
group fixed. Ambient-torus membership additionally uses distinct plane angles
up to sign modulo 2 pi and excludes angles 0 modulo pi. Restricted reachable
states can admit further coincidences. The multipane nonresonance argument,
dimension count and full-model interpretation remain written obligations. LayerNorm scale covariance is exact at positive scale/offset. Its
target-transfer lemma `PldrLlm.ln_target_transfer` uses Euclidean spaces and
assumes `LipschitzWith`; it retains the scale discrepancy and does not derive
the LayerNorm constant, identify the target or prove concentration. Constant
rows give the affine bias. The positive-offset image is distinguished from
its fixed compact containing set. The normalized image is a relative open
ball in the centered subspace; dimension one or zero gain can give a singleton. `PldrLlm.sum_sq_normalized` gives the exact
norm identity and `PldrLlm.sum_sq_normalized_lt_dim` checks its strict
upper bound under positive dimension and offset. Compactness, the image
characterization, surjectivity and the radial Lipschitz proof remain written
arguments.

`TwirlBound.lean` checks only `PldrLlm.norm_one_sub_exp` and
`PldrLlm.geom_orbit_bound`, the deterministic chord and geometric-sum core.
The matrix projection assembly and every stochastic application remain written
obligations. The book's normalized-query-Gram concentration proposition has
no concentration implementation. It models independent **pre-rotation** queries
with a common second moment and requires a separate LayerNorm target transfer.
Initializer checks establish scalar variance/support arithmetic with `N >= 1`
and `d >= 2`; they do not verify randomness or a trained infinite-width limit.

The unified check freshly compiles every owned module, resolves mapped declarations
and exports their full types. It audits every owned declaration's transitive axioms,
including generated declarations, and rejects any owned project axiom even if
unused. Only `propext`, `Classical.choice` and `Quot.sound` are admitted. Negative
controls exercise both an unfinished proof and an unused project axiom.

Successful compilation, the axiom boundary and semantic coverage are separate
checks. None establishes empirical assumptions, full native-program smoothness,
statistical concentration or universal criticality. The book mapping records those
boundaries instead of reporting a percentage of book proofs formalized.

The native checkpoint schema is checked by executable schema and restoration
tests. The algebraic AdamW formal library does not certify PyTorch serialization,
optimizer coordinate mapping, RNG formats or historical execution.

## Spectral, heat-trace and categorical boundaries

The reversible-family spectral dictionary, including positive-time extinction
at spectral envelope r = 0, has a standalone written proof and no claimed
general spectral formalization. Its two-state rational witness is an example.

`DagLoss.lean` checks `PldrLlm.trace_pow_nonneg`,
`PldrLlm.trace_exp_ge_card`, `PldrLlm.trace_exp_ge_card_add_trace`, and
`PldrLlm.trace_exp_ge_of_diag_bound`: exact-real power and trace inequalities.
Applying the last bound to N = M elementwise-squared gives h(M) >= d eps².
The normalized floor log(1 + eps²) additionally uses positive dimension and
monotonicity of log in the written proof. The full equality-iff-acyclic
statement, binary64 rounding, cancellation and implementation error are
outside these declarations. Strict positivity of A_P does not give it the
same entry floor as A_LM under arbitrary learned exponents.

The categorical KL Hessian argument is also a written proof. The book's
one-half estimate is conservative; the one-quarter Euclidean bound follows
from the Hessian norm at most one-half and the integral remainder. Existing
softmax normalization and common-shift declarations do not verify that bound.
