# Agile development with agents

The user accepted this orchestration proposal on 2026-09-07. Refine it from actual
delivery evidence while keeping the simple-client and observation-only constraints.

## Work in demonstrable slices

Keep one integrated slice active. Start with a client example and its observable
exit conditions, then implement only the boundaries that example exercises. A
slice should end with a runnable demonstration or a bounded investigation that
answers a named question. If it cannot be demonstrated independently, split it.

Use [the backlog](backlog.md) as the only work-status register. The coordinator
updates its current task, owner, evidence, and next action. Do not maintain duplicate
boards in skill files, agent chat summaries, and architecture documents. Record
new durable constraints in architecture; add a short decision entry only when
evidence changes a previous design choice.

## Initial agent topology

Start with one coordinator and up to two builders. Use a separate reviewer for
cross-component, concurrency, lifetime, or semantic changes; small low-impact tasks
do not need a permanent review agent. A builder becomes a reviewer only after
finishing its owned work. The topology is a starting limit, not a quota to fill.

The coordinator owns the slice, common event/batch contract, build/package files,
integration tests, and final client experience. Builders own disjoint files and
explicit deliverables. Delegate only if the work can proceed independently while
the coordinator does useful work. Do not create standing architecture, management,
documentation, or research agents.

For the first vertical slice, settle the minimal batch contract first. Then a
capture builder can produce real native events while a consumer builder exercises
the same contract with controlled fixture data. The coordinator implements their
transport/binding seam and assembles the real path. Fixture success is not evidence
that QEMU capture or remote delivery works.

## Task brief

Give each agent a short brief using this format:

```text
Outcome: one concrete behavior or question to resolve.
Owned files: exact paths or a narrow subtree; shared files have one named owner.
Contract: inputs, outputs, ownership, errors, and relevant version/base revision.
References: only the sources needed for this task.
Validation: a command or observable check, named host, and required evidence.
Limits: what is deferred, resource assignment, and any investigation timebox.
Handoff: changes, checks actually run, limitations, and next integration dependency.
```

If a shared interface must change, send the coordinator the concrete mismatch
before independently inventing another representation. The coordinator resolves
routine choices inside the agreed scope; consult the user for product-level
tradeoffs, not every reversible implementation choice.

Use `codex/` for task branches. Use separate worktrees when concurrent edits would
otherwise collide and an initial commit exists. Otherwise, enforce disjoint
ownership in the shared checkout. The
coordinator alone integrates build/package/schema changes. Do not let agents make
separate unpublished copies of a shared protocol or silently overwrite each other's
work. Preserve existing uncommitted user changes.

## Host coordination and evidence

Before remote work, assign the agent a host, a unique local build/output directory,
and workload scope in its brief. Use `trail-arm` and `trail-x86`; inspect local
operator notes for paths. Never run competing performance measurements on the same
host. Correctness jobs may share a host only when outputs/resources are isolated.
Do not copy build products or large traces into the shared source mount.

For evidence, report the command, source revision or dirty-diff identity, dependency
build, host/guest ISA, signal configuration, outcome, and artifact path. Performance
reports also need workload/input identity, repetitions, timing scope, memory use,
and backpressure behavior. Use the x86-64 host for performance claims. Mac/ARM runs
are development and portability evidence, not cross-ISA speed comparisons.

Failures that reveal missing operator dependencies produce a precise compatibility
report. Follow existing explicit authorization for separate developer builds; never add
QEMU installation to package setup. Work on independent native
and consumer components while that dependency is unresolved.

## Review and completion

Ask the reviewer to assess the task contract and actual diff, especially event
attribution, memory validity, buffer ownership, shutdown, and hidden client work.
Require concrete findings with paths and a consequence; do not reward checklist
volume or a predetermined finding count. The coordinator reviews and resolves them.

A slice is done when its end-to-end exit checks pass, the client example matches
real behavior, and relevant docs describe known limits. Run focused tests matching
the changed invariants; do not infer full coverage from skipped backend tests.
Do not claim throughput gains from a synthetic microbenchmark alone.

After each slice, briefly record what surprised us and change only the necessary
contract, task ordering, or instruction. Add a new skill only when repeated work
shows a distinct workflow; maintain one initial development skill now. No custom
agent daemon, scheduler, task database, or runtime orchestration dependency.

## How to start the next slice

Read the current task and its prerequisites in the backlog. Resolve the first
unknown through the smallest real check; update the task evidence; then select the
next ready card. Do not wait for a calendar sprint boundary or write the full future
architecture before proceeding. Detailed work is planned one slice ahead.

The skill lives in `.agents/skills/` so agents working in this repository can
discover it. This uses the documented [Codex repository skill convention](https://learn.chatgpt.com/docs/build-skills).
