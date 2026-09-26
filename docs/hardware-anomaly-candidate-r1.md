# Hardware anomaly candidate R1: exact-package screening

Status: **NO-GO for a second physical bug trigger** (2026-09-26). This is
read-only source and package-chronology research, not a reproduction or an
assertion that the signed `trail-x86` binary has been audited. No exploit,
capture, or host experiment was run. The admission criteria remain those in
[hardware anomaly validation R1](hardware-anomaly-validation-r1.md); Dirty
Pipe remains a secondary logic-flaw challenge.

## Strongest different-effect lead: TCP page-fragment stream corruption

The upstream [fix `dacb5d8875cc`](https://github.com/torvalds/linux/commit/dacb5d8875cc6cd3a553363b4d6f06760fcbe70c)
describes wrong bytes in a TCP stream when `tcp_sendmsg_locked()` copies from a
CIFS-backed mapping, a page fault starts a nested SMB transaction in the same
process context, and both sends reuse the task page fragment. This is a
different mechanism and oracle from Dirty Pipe: a receiver could compare a
bounded stream against known source bytes without writing a protected file.
The [stable backport](https://cos.googlesource.com/third_party/kernel/+/c6f340a331fb72e5ac23a083de9c780e132ca3ae)
also names that upstream commit and describes the required recursion.

Canonical's [exact `5.13.0-30.33` entry](https://lists.ubuntu.com/archives/impish-changes/2022-February/008704.html)
records two block-layer reverts, while its later [Impish `5.13.0-36.41`
history](https://lists.ubuntu.com/archives/impish-changes/2022-March/009011.html)
lists `tcp: fix page frag corruption on page fault` in a subsequent stable
patchset. Read-only `git ls-remote` resolves Canonical's
`Ubuntu-5.13.0-30.33` tag to
`c99150bb29da41155f3423daaa7bd221ccae44f8`. This is strong chronology,
not the required exact signed-package source/binary diff: an earlier
out-of-band or differently named backport has not been ruled out by source
inspection. The package state is therefore **not admitted**.

More importantly, the upstream report's trigger depends on a CIFS mount and a
nested network transaction during a page fault. A disposable local file or
ordinary prefaulted CIFS mapping is a useful *lawful control*, but it also
removes the nested SMB work and does not match kernel path length or load. A
credible high-load control would have to preserve CIFS, socket traffic,
mapping size, payload, affinity, and fault timing while removing only the
page-fragment recursion. No such matched control or repeated untraced
wrong-byte oracle has been demonstrated on disposable infrastructure. The
receiver and SMB server can run outside the gated target; asynchronous socket
work cannot be silently counted as target-process PT/PEBS/PMU signal. An
apparent score difference could instead be a page-fault/network-work or
session/relocation artifact. Do not exercise this path on the physical laptop
for this sprint.

## Two narrower leads also fail admission

| Lead | Primary-source chronology | Why it is not the primary gate |
| --- | --- | --- |
| `select()` indefinite sleep after another thread writes to and closes a watched fd | [Upstream fix `68514dacf271`](https://github.com/torvalds/linux/commit/68514dacf2715d11b91ca50d88de047c086fea9c); Canonical lists it in later [Impish `5.13.0-40.45`](https://lists.ubuntu.com/archives/impish-changes/2022-April/009232.html) | A bounded watchdog could identify a non-return, but the effect is a sleeping task: little or no target-kernel PT/PEBS/PMU accrues during the pathological interval. The write/close race also lacks a demonstrated repeatable, lawful, same-load sibling. No host run or exact-source diff. |
| `fanotify` stale installed fd after failed event-info usercopy | [Upstream fix `ee12595147ac`](https://git.zx2c4.com/linux-dev/commit/fs/notify/fanotify/fanotify_user.c?id=ee12595147ac1fbfb5bcb23837e26dd58d94b15d); Canonical's later [5.13.0-41.46 HWE source history](https://lists.ubuntu.com/archives/focal-changes/2022-May/032965.html) inherits the Impish patchset containing it | The fix says the path requires `CAP_SYS_ADMIN`; the effect is a stale fd following a failed usercopy, not bounded wrong bytes or pathological kernel execution. An EFAULT-vs-success comparison would mainly expose a distinct error path. No host run or exact-source diff. |

The later Canonical entries are package chronology, not proof from an upstream
version range. None of these leads has yet passed an exact signed-package
source/binary comparison, independent untraced oracle, matched lawful and
high-load controls, and target-attribution check. Their ordering does not
authorize trying successive triggers on `trail-x86`.

## Decision and safe alternative

Keep the physical known-bug campaign **NO-GO**. Continue with already retained
benign kernel PT/PEBS/PMU executions and the bounded lawful intensity, phase,
parameter-boundary, and modality/time-swap controls in the validation plan.
These can test sensor invariance and nuisance rates, but must not be reported
as an oracle-positive known vulnerability. If a future candidate is considered,
first audit the exact signed Ubuntu tag/package files against its fixing patch,
then obtain a second-person harness safety review and demonstrate a repeatable
effect with matching controls on disposable infrastructure. Only a separately
authorized, health-limited physical pilot could follow.
