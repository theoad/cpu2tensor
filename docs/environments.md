# Development environments

Operators provision machines, credentials, connectivity, dependencies, and guest
artifacts. cpu2tensor validates requirements; it does not install QEMU. The project
contains no cloud provisioning or credential-management runtime.

On 2026-09-07 the user authorized separate development QEMU builds for the ARM
VM and the x86 kernel integration. The upstream x86 build and exact API checks
are recorded in [kernel dependency evidence](kernel-qemu-build.md). This does not authorize bundling QEMU or adding
QEMU installation to the package. Keep that build outside the source checkout and
record its source, configuration, and headers.

| Role | Default local alias | Purpose |
| --- | --- | --- |
| macOS coordinator | Local | Source editing, Python client, MPS validation |
| AArch64 Linux worker | `trail-arm` | UTM Ubuntu 24 server; user-process capture |
| x86-64 Linux worker | `trail-x86` | Kernel capture, RL example, performance reference |
| AWS workers | Operator supplied | Multiworker observation/pretraining |

Configure the aliases in your SSH configuration using your endpoints and keys.
Use `ssh trail-arm` and `ssh trail-x86` in task briefs and commands. This checkout's
verified private inventory is in ignored `.local/environments.md`. Agents should
read it when working on these hosts. It is not needed by public clients.

Keep source in the shared folder. Give each agent/task a distinct worker-local
build and output directory under a cache path such as `~/.cache/cpu2tensor/`.
The ARM guest path for a shared checkout may differ from the macOS path; verify it
before running a command. Never measure shared-source filesystem throughput as
though it were the local capture pipeline.

Detailed QEMU plugin instrumentation uses TCG, including on a KVM-capable x86 host.
KVM availability does not accelerate the TCG-instrumented target. Record host ISA,
guest ISA, backend, dependency build, and signal configuration for evidence. Existing
AlphaFlow build commands/presets are references, not runnable cpu2tensor commands.

No performance claim without a named host. Use the x86 host as the performance
reference; report Mac/ARM development latency separately and do not infer a speedup
by comparing unlike host/guest configurations. Serialize benchmarks sharing a host.
