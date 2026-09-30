# PLDR-LLM Book Companion

Scientific code for **Power Law Graph Attention and PLDR-LLMs: Mathematical
Foundations, Training Dynamics, and Predictive Inference**, by Burc Gokden.

Repository: <https://github.com/burcgokden/PLDR-LLM-Book-Companion>

[The research integration guide](docs/RESEARCH_INTEGRATION.md) maps architectural questions to established dynamics, experiments and remaining conditions.

[The chapter guide](docs/CHAPTERS.md) connects the book's six parts to included
proof libraries, scientific programs and evidence. [Formal coverage](docs/FORMAL.md)
states the hypotheses and exclusions of the independent Lean checks.
[Single-pass admission](docs/ADMISSION.md) describes the complete CPU preflight
and qualification-evidence contract.

| Location | Book subjects |
| --- | --- |
| `companions/foundations/` | PLGA algebra, rotary geometry, normalization, inference contracts and released-checkpoint audits |
| `companions/dynamics/vendor/row/` | Row geometry, finite work, complete optimizer state and interventions |
| `companions/dynamics/vendor/rg/` | Chronological blocking, closure, face dynamics and orbit criteria |
| `companions/dynamics/vendor/model/` | Corpus laws, native training, predictive reductions, scaling and experiments |
| `companions/dynamics/PldrTrainingDynamics/` | Cross-part mathematical interfaces and initializer support |
| `companions/dynamics/vendor/native/` | Included pinned native implementation and tokenizer assets |
| `scripts/`, `tests/`, `provenance/` | Unified checks, book-specific regressions, source identities and correspondence |

The PLDR sources are included as ordinary independent files. No manuscript
checkout, predecessor source repository, submodule or source symlink is required.
Public third-party dependencies remain declared dependencies. Pretrained weights,
corpora and large raw acquisitions are external assets for the routes that need
them. There are no manuscript PDFs, LaTeX sources or document build commands here.

## Checks

```sh
python3 -m pip install -r requirements-checks.txt
python3 scripts/check.py integrity
python3 scripts/check.py scientific
python3 scripts/check.py resources
lake exe cache get
lake build
python3 scripts/check.py lean
```

Lean is pinned to `leanprover/lean4:v4.33.0-rc1`; Mathlib is pinned to
`9c0c555bde5a8277cd36dc4dc6dfe2a5a77a2b11`. The unified formal checker rebuilds
all owned modules and checks all owned declarations, including unused axioms.
`--dependency-cache PATH --lean LEAN-BINARY` may supply a matching read-only
third-party cache and compiler. Build objects and check reports stay in this
checkout under ignored `build/` and `validation/` directories.

The scientific suite includes the retained foundations audit, book initializer
checks and all four dynamics suites. Missing raw fixtures are explicit skips;
publication-only exclusions are recorded separately. The [supported audit release](docs/SUPPORTED_RELEASE.md) is pinned in the
book's Appendix C. The field-specific numerical
comparison and independent high-precision check are documented in [AUDIT.md](docs/AUDIT.md).

## Evidence and execution

The numerical evidence dataset is available at
<https://huggingface.co/datasets/fromthesky/pldr-llm-training-dynamics-data>.

```sh
python3 scripts/check.py evidence --data-repo DATA-REPO
python3 scripts/check.py standalone --data-repo DATA-REPO
python3 scripts/native_smoke.py --output validation/native-smoke.json
```

The last command needs a compatible CUDA stack and uses synthetic tokens, two
small widths and a bounded optimizer-state replay. It does not load trained
weights or reproduce a scientific training campaign. [EXECUTION.md](docs/EXECUTION.md)
distinguishes bounded interfaces, raw CPU reconstruction, synthetic smoke and
full acquisition, and specifies resource-contract limits.

`provenance/current-release.json` authenticates current source bytes independently
of Git HEAD. The immutable import manifest and separate book change record retain
source lineage. A new commit is not required to validate an uncommitted export.
The included source mappings keep their original scientific scope; the book-owned
correspondence supplies book labels and current numbering.

Apache-2.0 applies subject to the included license files and [NOTICE.md](NOTICE.md).
Third-party terms and attribution remain with their respective assets.
