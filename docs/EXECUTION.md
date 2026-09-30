# Execution levels

All paths below are in this standalone book companion. The included dynamics
inventory has 25 supported bounded interfaces and 948 archival scientific
utilities. The immutable import record also identifies two removed publication-only
utilities. The book's synthetic native smoke is a separate supported route.
Support is assigned to the stated check level, never inferred from a filename.

| Level | Inputs and command | What success establishes |
| --- | --- | --- |
| Bounded interface | `python3 scripts/check.py resources`; included CLI fixtures, dry runs, missing-input rejection and worker dispatch | Exercised parsing, admission, subprocess and dispatch behavior |
| Raw CPU reconstruction | Completed raw cache study, protocol, external identities and location manifest; commands below | Retained thirty-cell analysis and independent verification follow from admitted raw inputs |
| Synthetic GPU smoke | Included native source, compatible CUDA/PyTorch/Transformers stack; `python3 scripts/native_smoke.py --output validation/native-smoke.json` | Actual forward/backward, own-cache equality and generator bypass, registered schema agreement, serialized restoration, finite outputs, norm clipping, replay from all nonempty optimizer moments/counters, and equality of RNG endpoints and observations |
| Full admitted acquisition | Producer-specific states, tokenizer/token bytes, protocol, frozen identities, input roles, fresh output and resource admission | A completed producer, reducer and independent verifier supply a fresh scientific result |

Bounded dispatch and synthetic smoke do not establish checkpoint quality, trained
wide limits, throughput or completion of a scientific acquisition. The smoke uses
heads 2 and 4, five layers, eight metric units per layer, a two-example synthetic
batch, 64-token prefixes and one warm-up step followed by one replayed AdamW step.
`--device cpu` is available for a separately qualified CPU run; CUDA results must
not be described as CPU validation. Existing output paths are rejected.

## Single-pass admission

[The admission contract](ADMISSION.md) specifies mandatory source/input roles,
registered variants, CPU selection reconstruction, exact frozen jobs and
qualification evidence before parent dispatch or direct-worker initialization.
The native route pins the runtime and RNG formats in `docs/ADMISSION.md`.
Its CPU semantic tests have bounded-interface scope; native restoration is a
separate synthetic smoke. Unknown bindings reject before CUDA initialization. Structural success, rechecked
qualification evidence and completed acquisition remain separate stages.

## Raw reconstruction

From `companions/dynamics/`:

```sh
sh vendor/model/scripts/workspace-wrappers/analyze-cache-state-transfer.sh \
  RAW-STUDY FRESH-OUTPUT INPUT-LOCATIONS.json
python3 scripts/check_cache_relocation.py --source-study RAW-STUDY --workspace FRESH-WORKSPACE
```

| Role | Identifier and requirements |
| --- | --- |
| Scientific design | `rg-cache-state-transfer-v1` |
| Retained acquisition | `cache-state-transfer-v2`, with implementation `cache-state-contract-v2`; preserve the protocol and acquisition bytes |
| Current analysis and independent verification | `cache-state-contract-v3` |
| New acquisition | `cache-state-transfer-v3`, implementation `cache-state-contract-v3`, shared reservation receipt, bound hash and all admission metadata |
| Current output record schemas | `cache-state-transfer-analysis-v2` and `cache-state-transfer-independent-v2`, with contract field `cache-state-contract-v3` |

The pinned immutable v1 protocol and qualified v2 records have separate
reconstruction branches. The v2 branch still requires its specified identities
and complete inputs. Neither historical branch admits a new acquisition.

The input location manifest has schema `pldr-acquisition-locations-v1`, the
canonical absolute `logical_root` of the admitted acquisition, a `study` record
with exact `identity` and current `path`, and a `files` list. Each file supplies
its immutable acquisition `identity`, current regular-file `path` and `sha256`.
These caller locations are not distributed container paths. Supply every
external dependency bound by the protocol, as well as the completed study arrays.
Missing aliases, modified bytes, duplicate destinations and symlinks fail.

Replace the uppercase placeholders with admitted locations, quote paths
containing spaces, and supply a fresh output directory meeting the wrapper
contract. The backslash continues a single shell command.

The wrapper writes `analysis.json` and `verification.json`. The relocation check
copies the complete input graph, verifies source and relocated hashes, blocks
original-location reads, requires equality of all thirty scientific cell
dictionaries and runs the independent verifier. Allow about 3 GB temporary input
space and up to fifteen minutes per bounded CPU subprocess. This route performs
no native forward pass and does not regenerate trained states.

The compact Hugging Face dataset provides evidence records and complete outcome
grids. It does not provide the full raw graph, trained/optimizer states or corpus
bytes required for this route. Optional raw-fixture tests may skip explicitly;
publication-only exclusions are not counted as passes.

## Archival qualification helper

From the Book Companion root, the working directory and import path are both
part of this historical invocation:

```sh
cd companions/dynamics/vendor/row
env PYTHONPATH=experiments:experiments/confirm \
  python3 -B experiments/confirm/confirmation_qualification.py \
  --device cpu --output FRESH-OUTPUT.json
```

Choose an unused output path for `FRESH-OUTPUT.json` and install the declared
dependencies first. Direct execution without this import path can fail to
resolve `train_run`. This helper runs finite CPU qualification fixtures. It
remains one of the archival utilities, outside the maintained command inventory
and outside supported acquisition; its successful execution supplies neither
training data nor a completed scientific campaign. The 25 bounded interfaces
and the separate book native smoke retain their existing support levels.

## Qualified compatibility

The registered native stack is PyTorch 2.12.1+cu132 (CUDA 13.2) and Transformers
5.12.1, with the source, configuration and RNG contracts in [ADMISSION.md](ADMISSION.md).
TorchScript calls and the older RoPE configuration-validation interface emit
deprecation warnings on this stack. These warnings remain visible. Finite CPU
checks have their own narrower scope and do not qualify arbitrary later library
releases. The delivery validation record reports the executed Python and package
versions separately from the registered native runtime.

A TorchScript replacement or RoPE-configuration migration requires a new source
and runtime binding, review of rotation and cache semantics, and checks of logits,
gradients, optimizer state, serialization and replay. Historical acquisitions
retain their executed identities. No dependency modernization is part of the
entrywise-positivity comment correction; [its source record](../provenance/native-comment-adaptation.json)
binds the corrected bytes and preserves the original source with identical AST.

## Resources and support changes

See [the included resource contract](../companions/dynamics/docs/RESOURCE_EXECUTION.md),
[the support inventory](../companions/dynamics/provenance/supported-programs.json),
and [campaign accounting](../companions/dynamics/provenance/campaign-accounting.json).
Only block-normal, source-resolved and orbitwise shared interfaces carry the
durable cumulative-budget contract. Other launchers retain their declared
archival semantics. Generic supervision does not supply cumulative campaign
admission. Failed attempts, interruptions and retained journals remain charged
as specified. Sampled RSS, GPU-memory and output-size peaks are observations,
not continuous hard ceilings.

Promote an archival acquisition only after a producer-specific tiny end-to-end
fixture exercises admission, acquisition, reduction and independent verification,
including a retained failure case. No archival campaign is promoted by this book
release. Full campaign replication requires its own protocol and raw assets.

## Maintained command checks

From the book-companion root:

```sh
python3 -B scripts/check.py integrity
python3 -B scripts/check.py scientific
python3 -B scripts/check.py lean --dependency-cache CACHE --lean LEAN --jobs 4
```

`CACHE` is a compiled cache at the pinned Mathlib commit and `LEAN` is the
pinned Lean compiler. These are caller-supplied dependency locations. The
scientific dispatcher includes the book/foundations and all dynamics families;
use the row command in the component resource guide for that narrower suite.

`integrity` includes `scripts/check_documented_commands.py`. The maintained
[command inventory](../provenance/documented-commands.json) records each working
directory, shipped entry point, external interpreter and input placeholders.
The checker matches commands to their Markdown instructions, resolves local
files, runs declared safe help routes and checks shell syntax. It never runs
an acquisition example. This check detects documentation drift; it does not
validate a scientific protocol or supply external input assets.
