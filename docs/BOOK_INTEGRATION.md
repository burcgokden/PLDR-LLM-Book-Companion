# Included source lineage

The companion combines two independently copied source collections. Original
namespaces, licensing, native asset identities and scientific evidence are
preserved. The immutable `provenance/upstream-baseline.json` records imported
bytes, while `provenance/book-changes.json` identifies intentional book changes.
`provenance/current-release.json` authenticates the current standalone payload.

The changes add RoPE, row-softmax, LayerNorm and initializer checks, a portable
field-sensitive audit and its rejection controls, unified build/check interfaces,
execution-level documentation and a native synthetic smoke. Publication-only
utilities are removed as recorded in `provenance/scientific-only-disposition.json`.

The originals need not be checked out to build or test this repository. The clean
export check blocks original source reads, supplies a negative control and runs
scientific, resource and finite-interface checks. Its report remains separate
from full Lean compilation and native smoke evidence. No result claims that
long campaigns or pretrained checkpoint acquisition have been replicated.

## Publication references

The Book Companion is the available, standalone source for every Lean library
and scientific supporting program cited by the book. The book's code instructions
and formal correspondence use this repository. The two source works are
acknowledged in the preface; their results are developed and cross-referenced
within the book, and the written proofs are independent of the Lean checks.

Experimental records remain available in the
[training-dynamics dataset](https://huggingface.co/datasets/fromthesky/pldr-llm-training-dynamics-data).
The [Math Foundations repository](https://github.com/burcgokden/PLDR-LLM-Math-Foundations)
also supplies architectural experimental arrays. These data references do not
require a separate code checkout: the relevant programs and retained foundation
arrays are included here. The [chapter guide](CHAPTERS.md) locates them by book
subject, and [formal coverage](FORMAL.md) states which written clauses are checked.
