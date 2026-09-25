# Hardware anomaly evidence

The triage output is an evidence bundle, not a scalar alert. A deep-reasoning
reviewer must be able to reproduce the exact input, inspect the model residual
that selected it, decode the retained hardware observations, and distinguish a
missing symbol from a resolved one. The bundle never treats an unexplained alert
as a bug or a benign false positive.

`cpu2tensor.examples.hardware_anomaly_bundle` validates the sealed corpus and
checkpoint, rescoring one retained execution without updating the model. It emits:

- the exact subject, event, source, binary, raw-shard, derived-shard, and model
  hashes;
- argv, stdin, stdout, repetition, partition, and loop count needed to replay the
  input;
- scalar score, frozen threshold, calibration count, and finite-sample empirical
  tail probability;
- separate PT, PEBS, and PMU scores;
- ranked PT segments mapped back to raw AUX byte offsets, byte-feature
  residuals, and the timing residual used by the score;
- ranked PEBS samples with timestamp bucket, CPU, IP, address, weight, period,
  exact-IP state, and a lossless semantic decode of `perf_mem_data_src`; and
- ranked PEBS and PMU feature residuals, plus PEBS- and PMU-token timing
  residuals.

Per-token reconstruction error is the actual contribution used by the scalar
score, not an unrelated attention visualization. Per-feature residuals come from
the same masking pass that supplied each retained token maximum. This is the
first faithful attribution surface. A later learned `[ANOMALY]` token is accepted
only if intervention tests show that its localization tracks causal corruptions
better than these residuals; attention weights alone are not evidence.

Example:

```bash
python -m cpu2tensor.examples.hardware_anomaly_bundle \
  /path/to/sealed-corpus suspicious-00042 \
  --checkpoint /path/to/sealed-corpus/fused.pt \
  --kallsyms /path/to/exact-boot-kallsyms \
  --output /path/to/review/suspicious-00042.json
```

The symbol table must contain real addresses from the same boot and KASLR epoch.
On the qualified `trail-x86` boot, unprivileged `/proc/kallsyms` contains only
zero addresses because `kernel.kptr_restrict=1`; `/boot/System.map-*` is also
root-readable. An unprivileged `perf record`/`perf script --itrace=b` smoke test
therefore produced only address `0` and `[unknown]`. Operational collection must
retain a root-authorized `kallsyms` snapshot or equivalent perf sideband once per
session. The loader rejects an all-zero table instead of emitting counterfeit
symbols.

The raw direct-perf AUX representation is not yet a complete `perf.data` file.
The bundle localizes raw PT byte windows but does not claim packet decoding until
the persistent collector retains the required AUXTRACE, mmap, build-ID, module,
and BPF sideband. A diagnostic replay may collect a normal `perf.data`, but it is
secondary evidence and cannot replace a non-reproducible original window.

The LLM handoff gate passes only when controlled canaries demonstrate all of the
following across three sessions:

1. exact replay input and custody hashes reproduce the result;
2. PT localization overlaps the injected flow interval above a preregistered
   baseline;
3. PEBS IPs resolve on the original optimized boot and decoded data-source fields
   match `perf script` on sampled fixtures;
4. removing the highest-ranked evidence window decreases the frozen score more
   than removing a matched low-ranked window;
5. confidence remains calibrated on an independent benign partition; and
6. an LLM receives bounded decoded evidence plus references to full raw custody,
   never megabytes of undifferentiated trace or hidden condition labels.
