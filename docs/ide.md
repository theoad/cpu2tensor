# Language, layout, and IDE navigation

Native implementation language: C++20, following the lean runtime rules in
`AGENTS.md`. Use C17 for guest-facing adapter headers and example guest programs.
QEMU/plugin entrypoints use the required C-compatible ABI; implementation stays
in C++ where appropriate. Python remains an ordinary typed package.

This is the build contract for C2T-02. The standalone build and importable package
now exist. Command-line and actual VS Code checks pass. See [IDE evidence](ide-evidence.md)
for verified navigation and the JetBrains/Pylance limitations; a working compiler
is not mistaken for working editor navigation.

## Native project

`native/CMakeLists.txt` must be a standalone CMake project. Opening `native/` in
CLion or selecting it as the CMake source directory in another IDE must not depend
on opening a parent CMake project. Packaging invokes this same native build.
All real targets and native policy live in that one file. A root entrypoint, if
added for convenience, delegates without duplicating targets.

Use target-scoped include directories, compiler features, definitions, and
dependencies. Public include paths match the directory below `native/include/`:

```cpp
#include <cpu2tensor/trace.hpp>
```

Keep private headers next to their component sources and private include paths on
that component's target. No `../../` traversal into sibling implementation files,
absolute developer-machine includes, global include directories, or editor-only
include-path fixes. Express generated-header directories and external dependency
headers in CMake so compilation and navigation use the same configuration.

Use a Ninja development preset with `CMAKE_EXPORT_COMPILE_COMMANDS=ON`. CMake's
compilation database contains actual compiler invocations; point clangd or the
editor at the database in the selected build directory. CLion can read CMake
directly. Keep local database paths in per-user IDE/clangd configuration rather
than committing a machine-specific `compile_commands.json` or sharing a symlink
between Mac and Linux builds. Sources:
[CMake compilation database](https://cmake.org/cmake/help/latest/variable/CMAKE_EXPORT_COMPILE_COMMANDS.html),
[clangd setup](https://clangd.llvm.org/installation#project-setup), and
[CLion project loading](https://www.jetbrains.com/help/clion/opening-reopening-and-closing-projects.html).

Provide an explicit portable-core/client build profile for macOS and Linux, and
a Linux worker/QEMU profile. Selecting the latter must fail clearly when required
dependencies are absent. Do not fake Linux headers/macros merely to make local Mac
indexing look successful; use the Linux toolchain for backend sources.

## Python project

`python/` is the single Python source root and contains `cpu2tensor/`, `tests/`, and
`examples/`. Open `python/` directly in PyCharm, or open the whole repository and
mark only `python/` as the source root. Do not mark `python/cpu2tensor/` as another
source root: it would create an incorrect top-level import identity.

Keep one canonical `pyproject.toml` at the repository root, explicitly mapping
package discovery to `python/cpu2tensor`. Do not duplicate project metadata in
`python/` to satisfy an IDE. Use the selected interpreter's normal editable install
of the repository root. The command from `python/` is:

```sh
python -m pip install -e ..
```

Use absolute imports such as `from cpu2tensor.batch import Batch`; the module name
must match the file location. No `sys.path` mutation, custom module loaders, or
required working-directory tricks. Tests/examples must resolve the same installed
package. Test a regular wheel installation too: an editable checkout must not hide
missing packaged modules. Editable installation still requires rebuilding compiled
extensions after native changes. See [pip local project installs](https://pip.pypa.io/en/stable/topics/local-project-installs/).

Expose typed Python wrappers around the native binding, with matching `.pyi`
declarations where needed and `py.typed` packaging metadata. Maintain one extension
import name under `cpu2tensor`; no import-time compilation. Python navigation should
reach wrapper code or native API declarations. Navigation directly from Python
into a C++ binding body depends on IDE support and is not universally guaranteed.

## Builds and interpreters belong to their host

Sources are shared, but compiler output, CMake cache, extension binaries, generated
headers, and Python environments are host-specific. Use worker-local directories
under a cache such as `~/.cache/cpu2tensor/`, distinguished by checkout and build
profile. Do not reuse the same build directory across worktrees or configurations.
Keep Mac/MPS and Linux worker interpreters separate; never share a virtualenv or
load a Linux extension into macOS. Exclude local IDE workspace/cache files from Git.

For remote IDE operation, configure the source mapping between the Mac checkout
and the actual Linux checkout. The ARM shared path is in the ignored operator
inventory. The x86 checkout path must be established separately. Compile commands,
generated headers, debugger paths, and the interpreter must match that toolchain.

## Acceptance checks for the first build slice

- Configure `native/` directly without the parent project; build the portable
  target on macOS and Linux. A required QEMU profile never silently becomes a
  core-only build when a dependency is missing.
- Generate and inspect compile commands; public/private headers resolve using
  target configuration and the correct toolchain, including generated headers.
- Open `native/` in a standard CMake IDE and verify navigation to a declaration
  and implementation. Record the actual IDE/toolchain used, not a presumed pass.
- Open `python/` with an editable-installed interpreter; resolve package imports
  and binding types, and run the same example/test from outside the repo directory.
- Verify a regular installed wheel resolves the package and native extension with
  the same module identity. Confirm build/venv outputs stay off the shared source
  mount and do not collide across hosts or worktrees.
