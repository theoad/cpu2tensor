# Reproducible learner runtimes

The x86 learner runtime is a small offline wheelhouse for CPython 3.12 on Linux
x86-64. It contains the cpu2tensor wheel, CPU PyTorch 2.13.0 and the Python wheels
needed by PyTorch. It does not contain QEMU, a guest, credentials or target data.

The `x86 learner runtime` GitHub workflow builds the wheelhouse inside the same
digest-pinned Ubuntu 24.04 image used by CI. It installs only those local wheels
into a fresh virtual environment and checks a CPU tensor plus the cpu2tensor
native decoder before publishing the archive. `runtime-manifest.json` records the
full source commit, builder image identity, build-file hashes, Python identity and
every wheel hash. `SHA256SUMS` covers the wheel directory, while the adjacent
archive checksum covers the deterministic tarball.

After downloading both workflow artifacts or release assets:

```sh
sha256sum --check cpu2tensor-runtime-cp312-linux-x86_64.tar.gz.sha256
tar -xzf cpu2tensor-runtime-cp312-linux-x86_64.tar.gz
./cpu2tensor-runtime-cp312-linux-x86_64/install.sh /absolute/runtime/venv
/absolute/runtime/venv/bin/python -c \
  'import cpu2tensor, torch; print(torch.__version__)'
```

The installer performs no network access. The wheel tag and native extension are
specific to CPython 3.12 and Linux x86-64. Other Python or operating-system targets
need their own artifact. QEMU and worker binaries remain separate operator-owned
runtime components because their plugin headers and optional extensions must match
the exact QEMU build.
