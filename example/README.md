# Learning examples

Start with [trace digits](trace_digits/README.md), then [stdin interaction](stdio_gym/README.md).
[Kernel pretraining](kernel_pretraining/README.md) and [kernel actions](kernel_gym/README.md)
show real full-system learning and paused-world actions.
[Capture overhead](benchmark-capture/README.md) measures a signal selection against
vanilla QEMU. [Tensor pipeline](benchmark-pipeline/README.md) measures full column
upload and bounded rich-model updates across supplied endpoints.
[Normalization notebook](normalization/README.md) teaches a new native tensor
event and compares learned positions across real executable relocations.

Python examples live in the installed `cpu2tensor.examples` package under `python/`;
native targets live under `native/examples/`. This directory is the entry point,
while both source roots remain natural IDE projects.
