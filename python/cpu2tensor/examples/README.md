# Learning examples

[Example index](../../../example/README.md) and [learning guide](../../../docs/learn-trace.md).

Run the installed modules with `python -m cpu2tensor.examples.learn_trace` or
`python -m cpu2tensor.examples.learn_stdio`.

`pretrain_kernel` is the next-block baseline; `build_initramfs` packages the benign
kernel guest. See [kernel instructions](../../../docs/kernel-examples.md).
`collect_layout` and `learn_layout` support the
[native tensor notebook](../../../docs/tutorials/normalization.ipynb).

`benchmark_capture` and `benchmark_pipeline` separate instrumentation cost from
[end-to-end tensor consumption](../../../docs/benchmark-pipeline.md).

- [Context-only blocks](../../../docs/context-only.md): compact paging state and vectorized block association.
