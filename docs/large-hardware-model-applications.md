# Large Hardware Model applications and evaluation

The Large Hardware Model (LHM) is useful only if broad self-supervised hardware
pretraining produces representations that transfer to narrower applications.
Vulnerability research is the motivating task, not the only test of that claim.

## Candidate downstream tasks

- **Performance diagnosis:** localize unexpected stalls, cache behavior,
  contention, NUMA placement, and tail-latency regressions.
- **Compiler and JIT optimization:** rank code transformations using predicted
  runtime, energy, cache, and branch effects before expensive execution.
- **Database and runtime tuning:** propose query plans, indexes, memory layouts,
  garbage-collector policies, or worker configurations from execution effects.
- **Scheduling and resource control:** predict interference and choose CPU,
  memory, interrupt, and power placement under changing workloads.
- **Regression detection:** distinguish legitimate workload changes from
  compiler, firmware, kernel, library, or hardware regressions.
- **Cross-machine transfer:** predict how the same application changes across
  CPU families or ISAs and reduce the measurements needed to port or tune it.
- **Silicon and coherency validation:** surface rare errata, ordering failures,
  cache-coherency anomalies, and other behavior not explained by learned normal
  execution.
- **Energy optimization:** select DVFS and placement policies under latency and
  power constraints.
- **Concurrency testing:** steer schedules toward unusual but causal
  synchronization and memory-ordering effects.
- **Adaptive instrumentation:** decide which expensive trace modality or window
  is worth collecting next, and compress familiar traces without losing rare
  evidence.

## Evaluation by hierarchy

### Architecture pretraining

Hold out entire applications, workload families, boots, CPU models, and—when
available—ISAs. Measure masked and cross-modal prediction, next-window loss,
view identity, effective rank, retrieval across sessions, calibration tails,
and inference throughput. The decisive transfer metric is how quickly a frozen
or lightly adapted representation learns an unseen task compared with the same
architecture trained from scratch.

### Application fine-tuning

Hold out APIs, subsystems, action compositions, and intensity ranges within one
application. Measure forward action-to-effect prediction, inverse action
retrieval, counterfactual ranking, invariant residuals, and sample/compute
efficiency over a from-scratch application model. A pretrained model earns its
place only if it reaches the same quality with fewer interactions or reaches a
better frontier under equal resources.

### Harnessed inference

Use hidden effects, vulnerabilities, seeds, and lawful near-misses. Report
verified effects per physical execution, time to first terminal result,
recall/top-$k$, false positives at the review budget, input throughput, and the
fraction of anomalies that survive reproduction and minimization. Screen every
execution when measuring recall; learner rank and anomaly rank are separate.

## Flagship transfer evaluations

1. **Compiler/autotuning:** under a fixed trial budget, compare the LHM-guided
   policy with random search, Bayesian optimization, and profile-guided
   baselines on runtime/energy/code-size Pareto quality.
2. **Cross-architecture application transfer:** train on an application's x86
   executions and measure zero- and few-shot adaptation to Arm, and vice versa,
   against single-architecture pretraining and training from scratch.
3. **Performance-regression diagnosis:** hide application, compiler, kernel, or
   firmware changes and score detection, culprit localization, calibration,
   and false alarms on lawful workload shifts.
4. **Vulnerability-oriented harnessing:** fine-tune an application model on
   lawful interactions, then evaluate hidden known effects and vulnerabilities
   without using their labels during pretraining or model selection.

## Autoresearch cadence

Keep three coordinated loops rather than one mixed leaderboard:

- a pretraining loop optimizes reusable held-out representation quality;
- an application loop optimizes world-model transfer and interaction
  efficiency;
- a harness loop optimizes exploration yield, detection, throughput, and review
  cost.

Use multi-fidelity promotion: approximately five-minute scouts, 30--60 minute
confirmations, 2--4 hour scaling checks, and 24-hour runs only for finalists.
Every candidate keeps the same locked splits and baselines. Improvements move
down the hierarchy only after surviving several seeds and a larger rung; a
harness win must not retroactively select a foundation checkpoint on its hidden
evaluation set.
