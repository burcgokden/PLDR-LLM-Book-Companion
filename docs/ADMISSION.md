# Prospective single-pass admission

The single-pass route uses `critical-onepass-admission-v2` in
`vendor/model/src/model_rg/onepass_admission.py`, relative to
`companions/dynamics/`. Admission runs on the CPU in the parent before worker
dispatch and again at direct worker entry before CUDA or model initialization.
The checkpoint and completed-study verifier retains its independent numerical
reconstruction. The native numerical update is unchanged.

## Registered preparation and observation contracts

The protocol explicitly selects `variant.preparation` (`base`, `refinement`,
`clock`) and `variant.observer` (`native`, `shared`). Shared recording requires
a refinement selection, including a clock-family refinement. The preparers
record these roles; adding a digest does not register another variant.

All nine numerical core sources, including training and the replay comparator,
the admission helper and native-state schema are mandatory. Refinement adds its preparer;
clock preparation also adds the clock-family preparer; shared recording adds
its recorder. Unknown source entries and missing core entries reject.

The nine base external roles are corpus tokens, records and manifest; probe
tokens, offsets, records and manifest; and native model/configuration source.
`input_roles` identifies full locations, and `input_sha256` authenticates exactly
those locations. Declared relocation uses the existing asset-path resolver;
basenames and substring matches do not establish roles. Refinement additionally
binds its selection decision and each completed search analysis. Clock members
also bind their copied family specification. The copied design must match the
protocol and its recorded digest.

Corpus/probe manifests bind their arrays and record files. Document identities
must be unique and the assessment documents disjoint from training. The native
configuration's literal vocabulary default is read without importing the native
model. Tokens and offsets must lie in range. Selection archives are opened with
`allow_pickle=False`, with exact keys, integer shapes, source-consistent probes,
reconstructed seeded populations and nonrepeating supervised blocks. The clock
population follows `nested-width-prefix-v1`, rather than the full-half base law.
Production dimensions are fixed in `PRODUCTION`; there is no CLI miniature mode.

Jobs cover the frozen Cartesian design exactly once with canonical identifiers,
steps, seeds and checkpoint policy. Base jobs save full optimizer state for the
first two declared seeds; refinement and clock jobs save all final optimizer
states. Qualification uses its separate widest, 128-step, control-2 job; it need
not belong to the scientific design grid.

## Qualification evidence

Scientific admission requires the native and replay `manifest.json`,
`final-state.pt` and `observations.npz`, with all six identities bound to the same
protocol. Completed manifests must record the specified job, steps, source
identities, full optimizer/RNG checkpoint and qualification-only accounting.
Each manifest binds `native_state_contract` and a single worker's `cuda:N`
device role. `native_state_schema.py` registers `pldr-native-training-state-v1`
against fixed source identities for the model, configuration, TrainingModel,
inference adapter and optimizer selector. Protocols bind the model-schema
digests for every declared width, runtime, RNG formats and observer contract.
Rehashing an unknown configuration does not register it. No native model import,
CPU/meta construction or CUDA call is used to discover the schema at admission.

### Model and optimizer semantics

The registered branch has five layers, head dimension 64, residual width 64N,
feed-forward width floor(8*64N/3), eight residual generator units (two GLUs each,
hidden width 170), and vocabulary 32000. Attention, GLU and output biases and
all affine normalization weights/biases are included. Learned operators,
reference RoPE and untied embeddings are required by the pinned configuration.
All model keys, tensor shapes and float32 dtypes must match. This branch has no
persistent buffers or parameter aliases; RoPE caches and learned past-G caches
are nonpersistent. The different predefined-operator/configuration branches
are unsupported and require a new registration, not guessed buffer obligations.

Two ordered groups are reconstructed using `generator_parameter`. Each saved
ID must have its expected position and `param_names` entry, with no missing,
duplicate or extra coordinates. The producer records these names as additional
serialization metadata; its numerical update is unchanged. Every native loss
parameter has a gradient tensor and therefore a populated Adam slot, even if
its value is zero. The native smoke checks this participation independently.
Both moments must match that parameter's shape and float32 dtype; second moments
are nonnegative. Each scalar float32 Adam clock equals the integer checkpoint
step and the attempted/completed endpoint (128 for qualification).

The rates are exactly `3e-4*g` and `3e-4*128/(64*N)`, using the producer's
operation order, with the qualification job's own N and g=2. Required settings
are betas (0.9, 0.95), epsilon 1e-8, decay 0.01, constant schedule, foreach=False,
amsgrad=False, maximize=False, capturable=False, differentiable=False,
fused=None and decoupled_weight_decay=True. Signed first moments and parameters
are valid. Clipping at one is a bound producer rule; the checkpoint does not
prove it was applied. The stored norm precedes clipping and may exceed one.

### Supported random-generator formats

The registered runtime is PyTorch 2.12.1+cu132, its source commit
`7269437d655783a26cba32aa88195b741ff496aa`, CUDA 13.2, Transformers 5.12.1,
little-endian 64-bit pointers, and default float32. Admission reads version
metadata on the CPU and rejects an unknown runtime. These narrow registration
requirements apply to this prospective native route, not to every finite test
in the repository. Changing a version string in a protocol cannot extend support.

A local CPU `torch.Generator` restores the byte state and checks reproducible
continuation without changing global RNG state. It requires the current state
format returned by the pinned generator; no universal CPU byte count is assumed.
The CUDA parser runs entirely on the CPU. The pinned producer emits 16 contiguous
bytes: uint64 seed followed by int64 offset in native byte order. Restoration
casts the latter to uint64 and requires a multiple of four. Every seed bit and
both offset signs are legal. The native setter also accepts an eight-byte legacy
seed-only state; this producer never emits it, so this route rejects it.

The derivation follows the pinned [CUDAGeneratorImpl source](https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp),
specifically `get_state`, `set_state`, and `set_philox_offset_per_thread`.
A separate CUDA smoke tests restoration and continued draws, including high seed
and signed-offset bit patterns, plus one-byte and unaligned-offset controls.
This smoke initializes CUDA; admission does not. A CPU state cannot qualify as
CUDA state merely by renaming its field. Device role is recorded in the manifest;
the RNG bytes themselves encode no device identity or historical provenance.

### Observation semantics

The authoritative observer is `scripts/run_critical_onepass.py:observe`.
Its four head fields are row fraction, normalized attention entropy, operator
RMS and uncentered row energy, in that order. Reductions are float64 on native
float32 quantities; saved logits are float32, other floating arrays float64,
and schedules/blocks int64. Exact shapes and source selections remain required.

| Quantity | Registered admissible values |
| --- | --- |
| Row fraction | [-1e-10, 1+1e-10] |
| Attention entropy, final row on 64 positions divided by log(64) | [-1e-12, 1+1e-6] |
| Operator RMS, row energy, loss, NLL, gradient norm | finite and >= 0; no upper bound |
| Logits | finite and signed |

For row fraction, the allowance exceeds the binary64 accumulation budget
`16*4096*u/(1-16*4096*u)`, u=2^-53, for the fixed 64-by-64 reductions and
centering. The denominator floor cannot increase the fraction. The entropy
allowance preserves the producer's fixed stopping rule for rounded float32
probabilities, not a universal error theorem for an arbitrary softmax kernel.
These constants are fixed before acceptance testing; boundary tests use adjacent
float64 values. Invalid values reject without clipping or repair. Overlapping
milestone NLL entries must equal the recorded first-64-context NLL path.

### Equality and execution evidence

Each side must independently pass the semantic checks before logical C-order
replay equality and finiteness checks. File hashes alone do not establish replay.
Equality of two semantically admissible synthetic objects still does not prove
historical execution, clipping or source consumption. Authentication, usability,
paired equality and the producer's recorded execution remain distinct claims.

Shared recording additionally requires the replay's `shared-parameters.npy`,
`shared-steps.npy`, `first-gradient.npz` and `shared-metadata.json`. The full
recorded parameter selection, shapes and steps are checked, along with the
endpoint's equality to the saved model. The qualification record must describe
the uninstrumented/instrumented comparison.

Admission returns separate `structural_preflight`, `qualification_evidence`,
`completed_scientific_acquisition` and `native_updates_executed` fields under
`_admission`. A successful structural check does not certify qualification.
Rechecking a retained qualification pair performs no new native updates and
never sets completed scientific acquisition to true. Underscore-prefixed
validation fields are transient results, not frozen protocol evidence.

## Regression and support scope

The scientific suite discovers `vendor/model/tests/test_onepass_admission.py`.
Miniature external CPU assets carry full native checkpoint schemas and
backend-valid RNG formats. Their tensor values are explicitly synthetic; they
are not 128-step training observations. Base, refinement, clock and shared
variants (including shared clock) have positive controls. Each negative control
introduces one identical defect into both copies and refreshes dependent hashes.
Model, optimizer, RNG, observation and runtime defects are separately exercised,
with early-action guards on construction, dispatch, CUDA and optimizer updates.
Production-adapter tests retain the fixed external dimensions and entry ordering.
An independent schema test constructs real CPU models at widths 2 and 4 and
meta models at widths 3, 5 and 32; all width formulas are checked. A separate
native smoke restores a serialized complete model, populated Adam moments and
CPU/CUDA generators and compares a finite continuation's full state and
observations. It performs six smoke optimizer updates, no qualification updates
and no scientific acquisition.
These are bounded contract tests. They neither execute native qualification nor
promote an archival campaign to a supported full-acquisition route. The compact
evidence dataset lacks the large raw input graph needed for acquisition.
Historical protocols and their recorded hashes remain unchanged; reconstruct
them with their preserved executed-source identities. Newly prepared protocols
bind the prospective source bytes. See [execution levels](EXECUTION.md) and the
book's single authoritative reproduction table in Appendix C.

## Current native source adaptation

The [native comment record](../provenance/native-comment-adaptation.json)
identifies both bundled variants, their original and current byte hashes, and
the preserved original source. The sole native-model edit changes the positivity
comment; the Python AST and operation order are identical. The model schema and
runtime are unchanged, while `NATIVE_SOURCES` registers the corrected exact bytes.
New protocols bind that identity through `native_state_contract`. Unknown sources,
the preserved original source, and caller-rehashed alternatives reject on this
current route; none is automatically promoted to a new acquisition.

Historical protocols must use their preserved executed source and corresponding
historical admission contract. Do not rewrite their hashes or substitute the
current registration. The immutable upstream asset inventory retains downloaded
source identities and does not attribute the adapted bytes to an upstream commit.
The separate bounded native smoke validates the current source registration
before model construction and records that binding with its replay results.
