# cpu2tensor documentation

- [Learning examples](../example/README.md): trace classification, stdin actions and kernel work.
- [Trace learning](learn-trace.md): real CPU/MPS training and held-out controls.
- [Stdin learning](stdio-example.md): streamed reset/step, Gym and real policy results.
- [Next-block pretraining](kernel-pretraining.md): bounded multi-endpoint learner and its validation scope.
- [Kernel guest](kernel-examples.md): verified benign guest workloads and missing integration.
- [State correctness](state-correctness-results.md): current checkpoint and paging validation.
- [Capture benchmark](benchmark-capture.md): reproducible signal overhead and [x86 results](benchmark-capture-results.md).
- [Adding a signal notebook](tutorials/normalization.ipynb): executable layout, owned tensors and measured learning across relocation.
- [Instrumentation](instrumentation.md): current register/memory contract and limits.
- [Repeated action windows](action-windows.md): guest boundaries and fixed
  per-vCPU transition counts.
- [Instrumentation probe](instrumentation-probe.md): reviewed AlphaFlow and QEMU evidence.
- [Instrumentation results](instrumentation-results.md): current validation evidence.
- [Capture efficiency](capture-efficiency.md): hot-path audit, ring/collector reference,
  trace shape and proposed optimization boundary.
- [Quickstart](quickstart.md): build a worker and consume its trace as tensors.
- [Batches](batches.md): framing, source order, completion, and buffer ownership.
- [Validation](validation.md): portable, installed-package, and real worker checks.
- [QEMU probe](qemu-probe.md): checked dependencies, development build, and limits.
- [First-slice results](first-slice-results.md): named-host correctness evidence.
- [IDE evidence](ide-evidence.md): actual source-navigation checks and limitations.
- [Architecture](architecture.md): accepted requirements, proposed source layout,
  ownership boundaries, and reference map.
- [Development](development.md): small delivery slices, agent work assignment,
  review, and evidence.
- [Backlog](backlog.md): current task, sequenced milestones, risks, and completion
  criteria. This is the single work-status register.
- [Environments](environments.md): portable setup rules and machine roles.
- [Language and IDE setup](ide.md): standalone CMake, source roots, imports, and
  build/toolchain navigation requirements.

Open this directory as an Obsidian vault. Product requirements come from the
2026-09-07 design interview. Structure and delivery sequence are initial proposals
to revise as experiments produce evidence.

- [Capture performance](capture-performance.md): mixed frames, per-vCPU rings, bounded endpoint pools and measurement.
- [Kernel window acceptance](kernel-window-results.md): one-shot start/stop on the fixed getpid workload.
- [Tensor pipeline benchmark](benchmark-pipeline.md): rich uploads and bounded model updates across supplied endpoints.

- [Measured performance](pipeline-performance-results.md): matched MPS transfer/collation results and live rich workers; [native matrix](performance-matrix-results.md).

- [Worker deadlines](worker-deadlines.md): absolute observation budgets, incomplete traces and child cleanup.

- [Context-only blocks](context-only.md): paging attribution without register/memory traces and measured cost.
