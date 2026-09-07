---
name: cpu2tensor-development
description: Plan, implement, integrate, or review a small cpu2tensor development slice using the repo contracts, task board, and host evidence. Use for cpu2tensor repository work, not general RL research or unrelated AlphaFlow changes.
---

# cpu2tensor development

Resolve the repository root from this skill directory (`../../..`). Read its
`AGENTS.md` and the current card in `docs/backlog.md`. Load only relevant sections
of `docs/architecture.md`; use `docs/environments.md` and `.local/environments.md`
when host access is needed. All paths below are relative to that root.

1. Start from the card's client-visible outcome and exit evidence. Select the
   smallest next check or implementation that advances it. Distinguish accepted
   requirements, proposed contracts, and demonstrated behavior.
2. Follow `docs/development.md` for work assignment. Keep small tasks local. When
   delegation is useful and available, assign disjoint files and the same existing
   contract; one coordinator owns shared types, build files, and integration.
   Report interface conflicts to that owner rather than implementing a second ABI.
3. Use references to discover invariants and pitfalls. Write new code and validate
   it on the target path; do not port AlphaFlow assumptions, tests, widths, or
   performance numbers without checking their applicability and provenance.
4. For native/transport changes, check event attribution, per-source progress,
   buffer lifetime, bounded backpressure, and shutdown as relevant. For client
   changes, check a real minimal example and avoid public async/state machinery.
   Observation-only collection must never need client actions.
5. Run the card's relevant checks and record actual host/build evidence. Missing
   external dependencies are a finding, not permission to install/build QEMU.
   Follow explicit operator authorization for separate developer provisioning;
   never add QEMU installation or bundling to the package.
   Do not silently skip a required backend check or turn fixture success into an
   end-to-end claim. Reserve host resources before benchmarks.
6. Hand back changed paths, checks/results, limitations, and integration needs.
   The coordinator updates the single task board and any changed durable contract.
   Stop the slice when its exit checks pass; select further work from the board
   within the user's scope instead of expanding the framework speculatively.

For review, inspect the actual diff and task contract. Return concrete actionable
findings; do not assume the implementation or a reference baseline is correct.
The reviewed change must keep the simple Python experience and the optimized
observation-only path intact.
