# Learning examples

[Example index](../../../example/README.md) and [learning guide](../../../docs/learn-trace.md).

Run the installed modules with `python -m cpu2tensor.examples.learn_trace` or
`python -m cpu2tensor.examples.learn_stdio`.

`pretrain_kernel` is the next-block baseline; `build_initramfs` packages the benign
kernel guest. See [kernel instructions](../../../docs/kernel-examples.md).
`custom_kernel_actions` drives the
[client-owned syscall-sequence guest](../../../docs/custom-kernel-actions.md).
`collect_layout` and `learn_layout` support the
[native tensor notebook](../../../docs/tutorials/normalization.ipynb).

`benchmark_capture` and `benchmark_pipeline` separate instrumentation cost from
[end-to-end tensor consumption](../../../docs/benchmark-pipeline.md).

`hardware_triage` is the first frozen, undecoded Intel PT anomaly baseline. Its
[experimental contract](../../../docs/raw-hardware-triage.md) records the sustained
capture and review-budget gates that are still open.
`hardware_triage_benchmark` runs the family-partitioned benign calibration gate
on a Linux Intel PT host without decoding trace packets. Its checkpoint reloads
with `load_frozen_raw_trace_pca` for ordinary frozen inference; scoring never
updates the fitted model.
[hardware_multimodal](hardware_multimodal.py) is the fixed-tensor masked PT,
PEBS, and PMU training and calibration fixture.
`hardware_multimodal_experiment` collects the gated
[kernel-only multimodal corpus](../../../docs/kernel-multimodal-experiment.md),
then trains and evaluates fused whole-modality and span-only frozen models. Its
collect-only/train-only split supports capture on Linux x86 and training on MPS.
[hardware_pretraining_health](hardware_pretraining_health.py) validates the
versioned JSONL health ledger and applies the pure bounded 24-hour campaign
policy from the [scale procedure](../../../docs/hardware-pretraining-scale.md); it
performs no provisioning or experiment execution.

- [Context-only blocks](../../../docs/context-only.md): compact paging state and vectorized block association.
