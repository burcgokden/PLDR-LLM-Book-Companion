# From architectural foundations to training and inference dynamics

The book treats architectural identities, conditional dynamical theorems and finite measurements as different kinds of results. The links below connect the early architectural questions to their dynamical treatment. The written proofs are self-contained; [formal coverage](FORMAL.md) and the correspondence manifests identify separately checked clauses.

| Architectural question | Established contribution in the book | Condition or limit retained |
| --- | --- | --- |
| How does the shared row map change during training? | Theorem 8.1 gives finite work; Theorem 9.3 retains complete AdamW transport and forcing. | Exact path identities do not imply an autonomous scalar successor. |
| When is collapse invariant or attracting? | Proposition 9.2 characterizes the zero-gate condition; Theorem 9.7 and Corollary 9.8 give sufficient attraction and context-cover conditions. | The native training law is not proved to enforce those assumptions generally. |
| Does row collapse imply a reusable operator? | Proposition 4.15 separates row equality from context invariance; Equation (4.9) retains the common-row cache defect. | Small row energy alone does not control common-row or downstream context dependence. |
| How does contraction reach the emitted operator? | Proposition 25.5 controls row and context variance; Proposition 31.6 retains nonlinear covariance residuals. | Fixed-input Jacobians differ from complete training-state derivatives, and sampled bounds are not uniform certificates. |
| Does predictive reduction preserve training? | Propositions and measurements in Sections 32.5.10 and 32.5.11 retain actual loss sources and clipped shared gradients. | Small predictive KL can coexist with a large force error. |
| Is there an explicit RG? | Chapters 12 and 16 construct chronological affine and complete-law maps; Theorem 29.4 gives covariance blocking. | Token-space covariance and autonomous reduced-state closure are separate requirements. |
| Is criticality established? | Section 19.3 measures independent finite width/control families and clock-dependent crossings; Section 29.2.4 gives a conditional feedback criterion. | No universal exponent or native self-tuning law follows from a finite peak. |
| What positional operator sector is occupied? | Section B.1.5 measures the relative off-commutant Frobenius norm. | Its square is the energy fraction; predictive use requires a separate projection intervention. |
| When does caching work? | Section 23.4 gives conditional risk and state transport; Section 24.4 tests recalibration and disjoint contexts. | State-specific aggregate fidelity allows context exceptions; transporting the initial cache fails in the positive-control cells. |
| What computational benefit is measured? | Section 24.2 measures paired native/cache model-call savings with all weights resident. | The result concerns batches of eight 64-token proper prefixes, without KV caching; it is not a generated-history or SDPA speed comparison. |
| Do the transfer conjectures follow? | Section 28.2 measures native source adaptation; the cache experiments distinguish context reuse from state transfer. | Constrained cross-domain congruence and matched-resource whole-model transfer remain conjectures. |

The empirical laws retain their source, optimizer, schedule and initialization conditioning. Number of heads, head width, context length, training-token count and physical age count different objects. The RMS-normalized deductive fluctuation is bounded by two; source results using a mean-based normalization must not be assigned unchanged to that statistic.

All scientific programs and proof libraries needed for these contributions are included under `companions/foundations/` and `companions/dynamics/`. [The chapter guide](CHAPTERS.md) identifies their entry points; [execution scope](EXECUTION.md) distinguishes compact checks from routes requiring separately specified raw data or model assets. These finite checks do not establish a missing population hypothesis.
