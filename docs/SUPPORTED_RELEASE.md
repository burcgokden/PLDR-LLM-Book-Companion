# Supported book audit

The book's supported numerical audit is the implementation included at
`companions/foundations/audit/` in this Book Companion. Appendix C prints the
public commit and immutable payload SHA-256 from `provenance/current-release.json`. Check that
identity before running the audit, then run `python3 scripts/check.py integrity`.
The manifest binds the source files, toolchain pins, exact retained-input and
decision manifests, formal correspondence and maintained command inventory.
A Git commit identifies the checkout; the payload identity also checks exports.

[The numerical contract](AUDIT.md) specifies the six quantity classes, operation
budgets and supported binary64 domain. Individual decisions, types, hashes,
shapes, keys, counts, signs, zeros and ties remain exact. The separate 80-digit
Decimal checks and nine-significant-digit diagnostic do not replace that
contract. Use the Book Companion commands in that document for reconstruction.

## Relationship to the separate foundations checkout

The separate PLDR-LLM-Math-Foundations checkout at
`a87ccc1949e57d20640ec425ab24138401641aa6` used exact JSON serialization equality
in `TestShippedArtifacts.test_rederive_all_summaries`. It is not the supported
book audit. That checkout's reported MC2 roundoff mismatch was approximately
2.17e-24. The Book Companion retains the scientific inputs and archived
observations, compares derived fields under explicit operation budgets and
checks individual decisions independently. Its implementation lineage is
recorded in `provenance/audit-comparison-history.json`. The separate repository
remains an experimental-array source; the book's code instructions use this repo.

The [retained first-failure record](../provenance/public-audit-first-failure.json)
records 124 passing tests and one failing exact-JSON comparison on the separate
checkout. Its differences describe the first failing summary only; the stopped
assertion is not an exhaustive comparison of later fields. The original failed
log is bound by SHA-256, and the earlier full-field comparison history remains
unchanged. A subsequent separate public checkout at
`ca399d0b0f8435dafef3caac2dcd9b92629a15a2` has a retained 135-test pass,
summary reconstruction and Lean validation. The
[resolution record](../provenance/upstream-audit-resolution.json) binds those
executions to that commit. They resolve the numerical issue for that tested
checkout while preserving the earlier failed execution.

## Cleanup fixture and validation scope

The descendant fixture publishes, flushes, synchronizes and closes its PID file
before releasing the child's pipe gate. Its controlled boundary cases hold the
file empty while the child acknowledges that it is still attached. Attached
and detached paths both require complete terminal accounting, a recorded child
identity, complete cleanup and no late write. The detached cases additionally
require that PID in the supervisor's detached-descendant record. Handshake EOF
and deadline expiry fail the fixture. Production supervision is unchanged.

Run `python3 scripts/check.py scientific` and `python3 scripts/check.py resources`
for the maintained suites. These suites overlap; do not sum their counts.
Generated logs, first outcomes, named skips, exclusions and failures remain in
`validation/`. The book delivery's validation record binds the executed release,
software environment and corresponding outputs. The complete standalone check
also tests a fresh export with original Python source reads denied and verifies
its negative control; this is a Python-level portability check.

Lean remains pinned to `leanprover/lean4:v4.33.0-rc1` and Mathlib to
`9c0c555bde5a8277cd36dc4dc6dfe2a5a77a2b11`. [Formal coverage](FORMAL.md)
retains partial mappings and exclusions. Fresh owned-module compilation,
transitive axiom checks and declaration-type resolution check those mapped
clauses. Compact evidence verification, raw reconstruction, synthetic native
smoke and new acquisition have the distinct scopes in [EXECUTION.md](EXECUTION.md).

## Registered qualification state

[Admission](ADMISSION.md) specifies the exact source, model schema, named Adam
coordinates, runtime and RNG formats for newly prepared single-pass protocols.
Both qualification checkpoints pass those checks independently before replay
comparison. The [native smoke](EXECUTION.md) restores a serialized full state
and compares moments, RNG endpoints and observations after a finite continuation.
These tests do not authenticate the historical origin of synthetic values or
retroactively alter retained acquisition protocols.

## Source and environment identity

The [native adaptation record](../provenance/native-comment-adaptation.json)
binds the entrywise-positivity comment correction without changing executable
model content or historical acquisitions. The current schema registration and
payload manifest bind the corrected source. The [archival invocation and runtime
notes](EXECUTION.md#archival-qualification-helper) give the required working
directory and import path and retain TorchScript/RoPE deprecation warnings.
Those instructions do not promote archival helpers or newer dependency releases.

## Binding a checkout to an execution

Use the exact public commit printed in Appendix C, then verify
`provenance/current-release.json` before running the documented commands.
The book delivery's `validation/execution-release-binding.json` identifies
the tested payload, executed checks, input identities and observed package
versions. Its `validation/commits.json` binds that payload to the public and
internal commits. These records stay outside the payload they authenticate.
Neither a moving branch nor a later checkout inherits a passing result.

Declared package requirements describe dependencies; the observed runtime
describes one execution. TorchScript and RoPE deprecation warnings remain
recorded with their original native runs. No runtime API or dependency
modernization is included in this documentation maintenance. The scientific
programs and retained inputs keep their bytes. Lean comment changes receive
new source hashes while declaration types and proof bodies are checked for
preservation. The release validation records which checks were freshly
executed and which experiments retain their original evidence.

The separate upstream numerical issue is resolved for the tested commit in
the resolution record. Its passing results and the retained first-failure
log have distinct source/runtime bindings. Neither is reassigned to the
Book Companion or to a later prepared checkout.
