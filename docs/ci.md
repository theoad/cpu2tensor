# Continuous integration

GitHub Actions runs every unit test for pushes and pull requests. The unit job
enforces at least 95% line coverage independently for the Python package and the
native core. A 55-second process deadline leaves five seconds for runner cleanup,
so the reported unit suite remains below one minute.

System tests run when the tested commit has more than one parent on a `push`, or
when its title contains the exact marker `[TESTME]`. Pull requests use their head
commit rather than GitHub's synthetic merge ref, so the marker has the same
meaning before and after merge. The suite has a 590-second process deadline and
exercises real QEMU plugin loading rather than fixture frames:

- prescribed-input traces from x86-64 and AArch64 user processes;
- rich register and memory-value capture from three concurrent workers into one
  CPU learner step;
- two stdin action boundaries with the target paused between actions.

The jobs build `ci/Dockerfile` and run the repository inside that image. The
workflow actions and Ubuntu base image are immutable revisions. The image pins
the Python test stack and QEMU 11.0.3, verifies the QEMU source archive checksum,
and builds only the two Linux user-mode emulators needed by the tests. GitHub's
Docker layer cache avoids rebuilding those dependencies for ordinary commits.
The image is development infrastructure; the package still does not install or
ship QEMU.

Run the same checks locally from the repository root:

```sh
docker build -f ci/Dockerfile -t cpu2tensor-ci:local .
docker run --rm -v "$PWD:/workspace" cpu2tensor-ci:local ./ci/run-unit-tests.sh
docker run --rm -v "$PWD:/workspace" cpu2tensor-ci:local ./ci/run-system-tests.sh
```

Coverage excludes installed examples, which are executable tutorials rather than
the package API. Device-specific MPS and CUDA behavior remains validated on the
corresponding hardware; an unavailable device skip does not count as backend
evidence.
