# Guide by book content

| Book part | Included code | Evidence and interpretation |
| --- | --- | --- |
| I. Mathematical foundations and exact row dynamics | `companions/foundations/PldrLlmMathFoundations/`, `companions/foundations/audit/`, `companions/dynamics/vendor/row/` | Exact PLGA/SDPA identities, rotary geometry, LayerNorm, cache/causal contracts, row energy, finite work and complete optimizer state |
| II. Chronological renormalization and closure | `companions/dynamics/vendor/rg/` | Affine blocking and face dynamics, state-sufficiency interventions, conditional orbit criteria |
| III. Complete training laws and shared collectives | `companions/dynamics/vendor/model/src/model_rg/`, `companions/dynamics/vendor/model/scripts/` | Consuming-corpus laws, single-pass producers, shared/head variables and physical clocks |
| IV. Predictive reduction, caches and adaptation | Same model family; use named cache, memory, moment and fine-tuning programs | Native-law fidelity, external-target risk, calibration, conditional forecasts and state transfer |
| V. Conditional scaling and inference visibility | Model-family scaling, physical calibration and readout programs; included Lean `ModelRG/` | Conditional initial/sign limits and readout error budgets, with finite experimental boundaries |
| VI. Appendices | `docs/`, `provenance/`, unified `scripts/` | Statistical units, complete outcomes, execution levels, independent proof correspondence and notation |

The foundations collection contains the retained released-checkpoint arrays.
The dynamics evidence is in the public dataset, pinned by
`provenance/evidence-source.json`. Dataset display/claim numbering belongs to its
source-work mapping; the book preserves its scientific records and provides its
own correspondence. No data record is rewritten to accommodate book numbering.

Execution support is narrower than source inclusion: consult [EXECUTION.md](EXECUTION.md)
before treating an archival program as an acquisition workflow. The source layouts
and namespaces remain stable to preserve meaningful experiment and proof imports.

The empirical studies open with question, independent unit, intervention,
endpoint, outcome and limitation. Appendix C is the authoritative map of
reproduction levels and current release identities; the admission-specific
contract is in [ADMISSION.md](ADMISSION.md). Appendix D maps the LayerNorm
norm identity and strict dimension bound separately from its written
compact-containment and radial Lipschitz arguments.
