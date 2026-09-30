# Derived numerical summaries

The supported book audit is the included `companions/foundations/audit/`
implementation, pinned by the release payload printed in Appendix C.
[Supported release](SUPPORTED_RELEASE.md) identifies its relationship to the
separate foundations checkout. It compares derived quantities while retaining exact
input integrity. It never rewrites a retained array or scientific summary.

Run from the book companion root:

```sh
python3 -B -m pytest -q companions/foundations/audit
python3 -B companions/foundations/audit/reconstruct_summaries.py --output validation/sequential-reconstruction.json
```

The six raw/result files are pinned in
`companions/foundations/audit/retained-inputs.json`. Their hashes, shapes, key sets,
counts, integer indices, categories and Boolean decisions are exact. Booleans are
checked before numeric coercion. No NaN or infinity sentinel is admitted in the
sequential summaries. Unregistered fields fail even if both objects contain them.
Count-derived accuracy fractions are exact. Signs and exact zeros are also exact.

For binary64 unit roundoff `u = 2^-53`, put `gamma(n) = n*u/(1-n*u)`.
The policy rejects reduction sizes with `n*u >= 0.01`. Let `T` be token count,
`I` item count, `L` maximum candidate token length, `K` maximum candidate count,
`M` the maximum absolute primitive log probability, and `A` the larger mean
absolute log probability of the two scoring protocols. Define:

- `e_token = 4*u*M`, allowing two subtraction paths;
- `e_candidate = 4*gamma(L+2)*L*M`, covering the candidate reductions and subtraction;
- `e_mass = 2*e_candidate + 4*gamma(K+8)`, propagating score errors through a
  probability mass, shifted exponentials, summation and division;
- `r = 2*gamma(max(T,I)+8)`, covering the final reduction/comparison paths.

The exponentiation allowance assumes at most four units of relative roundoff
per finite exponential in the supported stack. These are conservative operation
budgets, not platform-independent correctly-rounded-libm theorems. Retained
finite arrays are the domain. A new dtype, nonfinite policy, reduction formula or
unsupported library law requires requalification.

| Quantity class | Absolute allowance `a` | Relative allowance |
| --- | --- | --- |
| NLL/CE means | 0 | `r` |
| Difference of NLL/CE means | `4*gamma(T+2)*A` | `r` |
| Token-gap statistics and position medians | `e_token` | `r` |
| Candidate-gap and margin statistics | `e_candidate` | `r` |
| MC2 probability-mass means | `e_mass` | `r` |
| MC2 gap, signed-gap and absolute-gap statistics | `2*e_mass + 4*u` | `r` |

The comparator requires `abs(actual-expected) <= a + r*max(abs(actual),abs(expected))`.
It separately requires the exact discrete decisions above. The small approximately
`8.45e-10` MC2 mean gap therefore cannot be erased or change sign. There is no
blanket `1e-8` tolerance and no blanket significant-digit rounding rule.
Nine-significant-digit agreement is reported as an additional reconstruction
diagnostic, not used to define a tolerance. The machine-readable report records
maximum discrepancy and budget by quantity class.

`sequential-decisions.json` binds the individual maximizer sets (including ties),
pair signs, half-margin gates and held-out per-token signs. The independent
recomputation uses `math.fsum` for candidate totals and compares serialized
categories with exact types. This prevents compensating changes in aggregate
counts from hiding changed individual decisions.

The cancellation-sensitive MC2 means and signed medians also have an 80-digit
Decimal reference, using exact conversion of the retained binary64 inputs,
independent sums and Decimal exponentials. Differences from ideal high-precision
arithmetic and differences between two binary64 reconstructions are reported
separately. The high-precision comparison is an additional numerical check,
not evidence for a new training or inference result. Binary64 rederivation
retains the recorded nine-significant-digit summaries. Some cancellation-sensitive
diagnostics change their nine-digit rendering against the Decimal reference
within the declared budgets; binary64 agreement is not a claim that every
high-precision reference digit agrees.

Negative tests cover corrupted input hashes and categorical metadata, missing
keys, altered shape, changed counts, Boolean/integer confusion, unknown fields,
nonfinite values, changed signs or zero decisions, ties, and one representable
step beyond each class's comparison budget. Source provenance records the
inherited exact-serialization comparison and its replacement; archived evidence
bytes remain unchanged.
