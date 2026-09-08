# Working on cpu2tensor

Read [the current task board](docs/backlog.md), then only the architecture and
environment sections needed for the task. Block, register and memory observation, simple stdin interaction, and small
CPU/MPS learning examples are implemented. Rich x86 kernel capture, multiworker pretraining, and KernelEnv/Gym now run; see the board for exact evidence.
Do not report later planned APIs or checks as implemented.

## Product constraints

- Optimize observation-only collection from supplied target inputs. It must not
  require a policy, action requests, or Gym machinery.
- Keep first-version Python clients simple and synchronous. Advanced async batch
  APIs, public opaque trajectory IDs, state caches, and migration are deferred.
- Support Linux user processes and full-system guests through one capture/data
  pipeline. Keep kernel constraints visible from the first iteration.
- Preserve per-vCPU concurrency and sequence; do not manufacture a total guest
  memory order. Lossless backpressure and explicit incomplete data are required.
- Operators supply QEMU, hosts, connections, and dependencies. The distributed
  package and its setup must not build, download, install, or bundle QEMU. The user
  authorized separate development QEMU builds on the ARM VM and x86 host on 2026-09-07. Neither is a shipped dependency.

## Source rules

Write simple, readable code that contributors can follow without project lore.
Use plain English names, errors, comments, and documentation. Prefer small functions
and direct control flow. Explain ownership and surprising constraints; do not add
frameworks, layers, or extension points before a real use case needs them.

All native code, bindings, native tests, and native example targets belong under
`native/`, with one `native/CMakeLists.txt`. All Python belongs under `python/`,
with one importable package, `cpu2tensor`, and absolute imports. No `sys.path`
modification or duplicate import identities. Create subdirectories when code needs
them; the architecture tree is a direction, not an instruction to add stubs.
For build/package changes, follow [the IDE contract](docs/ide.md): standalone
native CMake, target-scoped includes, one Python source root, and host-local
build/interpreter paths. Smooth IDE navigation is part of acceptance, not cleanup.

Keep substantive prose in `docs/`. Code directories carry short README links.
Agent entrypoints, license files, and ignored operator notes are operational
exceptions. Use `$...$` and `$$...$$` for math; `docs/` can be opened as an Obsidian vault.

Carry forward the lean native style: C++20, no exceptions/RTTI, STL containers,
iostream, or hot-path allocations. Use `Result<T>` for failures; normal outcomes
are values, without status-code out-parameters. Use final classes, leading-underscore
members, named constants, and ownership/invariant comments. Guest adapter headers
remain C17. Python-facing failures should become ordinary Python exceptions at the
binding boundary. Audit the chosen binding's runtime requirements before adopting it.

Implement from scratch; references are evidence, not templates to copy wholesale.
New project code is AGPL-3.0-only. Verify reference/dependency licensing before
incorporating code. AlphaFlow is an external reference and future client.

## Agent work

For a development slice, use [cpu2tensor-development](.agents/skills/cpu2tensor-development/SKILL.md)
and [the workflow](docs/development.md). Delegate only concrete independent work;
small tasks stay with one agent. One owner edits shared contracts and build files.
Do not introduce an orchestration service or an agent dependency into the runtime.

Use the short SSH aliases in [environments](docs/environments.md). Read
`.local/environments.md` when present for this checkout's operator inventory.
Keep worker builds/results off shared source mounts. Name the host for every
measurement; x86-64 is the performance reference. Never compare cross-ISA runs as
if they used the same execution environment. Follow existing user authorization;
do not add confirmation steps for routine reversible development work.
