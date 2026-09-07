# IDE navigation evidence

Checked on macOS AArch64, 2026-09-07. These checks use actual editor navigation,
not just a successful compiler invocation. Screenshots and accessibility results
were captured in the development task's tool outputs; no screenshot files were
added to the repository.

## Native navigation: passed in VS Code

Opened `native/` directly in VS Code 1.133.0 with the already installed Microsoft
C/C++ extension (`ms-vscode.cpptools@1.34.3`). A local, ignored
`native/.vscode/settings.json` selects the compilation database:

```text
/Users/theoad/.cache/cpu2tensor/Users/theoad/Workspace/ubuntu-server-share/cpu2tensor/native/portable/compile_commands.json
```

It also sets `cmake.configureOnOpen` to false. Include paths come from CMake's
compiler commands; no extra include directory was added to make navigation pass.

Actual UI sequence and results:

1. Opened `tests/trace_test.cpp` and put the caret on `encode_header` at line 11,
   column 8.
2. Pressed F12 (Go to Definition). VS Code opened `core/trace.cpp`, line 20,
   column 6, at the `encode_header` implementation. The accessibility output
   reported one found symbol in that file.
3. Ran Go to Declaration. VS Code opened `include/cpu2tensor/trace.hpp`, line 32,
   column 6, at the matching public declaration.
4. The status bar reported IntelliSense Ready and Parsing Complete.

CLion 2025.1 was tried first. It opened `native/` and recognized the CMake presets.
Project-only settings disabled its default Debug profile and enabled `portable`,
with the same external cache directory. Its CMake initialization remained at
`Updating…`; the log reported `Waiting for startup futures to complete`, and
explicit reload did not produce a configure command for this project. Navigation
reported `Cannot find declaration to go to`, with no target context. This is a
failed CLion check, not a pass inferred from the successful command-line build.
Existing unrelated project settings were not edited.

## Python navigation: passed in VS Code

Opened `python/` directly in VS Code 1.133.0. Installed the Microsoft Python
extension (`ms-python.python@2026.4.0`) and Pylance
(`ms-python.vscode-pylance@2026.3.1`) for this development check. Their installer
also added `ms-python.debugpy@2026.6.0` and
`ms-python.vscode-python-envs@1.36.0` as dependencies.

The local, ignored `python/.vscode/settings.json` selects the existing
`/Users/theoad/.cache/cpu2tensor/dev-python/bin/python` interpreter, keeps the
system environment manager, and disables automatic terminal environment
activation. Python: Select Interpreter was also used to choose the configured
interpreter explicitly. The status bar then showed Python 3.10.12 and that exact
path, with Pylance as the diagnostics source. No environment was created by the
editor and no Python search paths were added.

Actual F12 (Go to Definition) results:

1. `examples/observe.py:8`, imported `Pool`, opened `cpu2tensor/pool.py:28`.
2. `cpu2tensor/pool.py:13`, imported `Batch`, opened `cpu2tensor/batch.py:10`.
3. `cpu2tensor/pool.py:130`, `_native.new_stream()`, opened
   `cpu2tensor/_native.pyi:3`.

Pylance still reports `reportMissingModuleSource` for `cpu2tensor._native` on
line 12: it sees the stub but cannot resolve Python implementation source for the
compiled extension. The warning was left visible. Navigation to the typed API
works, and the import check below confirms the binary is loadable. No direct
Python-to-C++ binding implementation jump is required or claimed.

PyCharm was tried first. Its Open File or Project dialog resolved `python/`, but
after clicking Open, CUA returned
`Computer Use server error -10005: timeoutReached` for state and screenshot calls.
Retrying with a new control session produced the same result. The existing
PyCharm process was not killed or restarted. This is a failed PyCharm check.

An independent import check ran from `/tmp` with
`/Users/theoad/.cache/cpu2tensor/dev-python/bin/python`. It resolved:

| Object | Actual location |
| --- | --- |
| `cpu2tensor` | This checkout's `python/cpu2tensor/__init__.py` |
| `Batch` | This checkout's `python/cpu2tensor/batch.py` |
| `Pool` | This checkout's `python/cpu2tensor/pool.py` |
| `_native` | Host-local environment's `site-packages/cpu2tensor/_native.cpython-310-darwin.so` |
| `_native.pyi` | Present beside this checkout's Python package |

This independently verifies editable import identity outside the source directory;
the actual editor navigation results above establish source and stub navigation.
