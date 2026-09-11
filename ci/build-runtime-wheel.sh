#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

readonly root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly output="${CPU2TENSOR_OUTPUT_DIR:-/output}"
readonly source_sha="${CPU2TENSOR_SOURCE_SHA:?Set CPU2TENSOR_SOURCE_SHA to the tested commit.}"
readonly builder_image_id="${CPU2TENSOR_BUILDER_IMAGE_ID:?Set CPU2TENSOR_BUILDER_IMAGE_ID.}"
readonly torch_version="2.13.0"
readonly torch_index="https://download.pytorch.org/whl/cpu"
readonly bundle_name="cpu2tensor-runtime-cp312-linux-x86_64"
readonly work="$(mktemp -d /tmp/cpu2tensor-runtime.XXXXXX)"
trap 'rm -rf "$work"' EXIT

if [[ ! "$source_sha" =~ ^[0-9a-f]{40}$ ]]; then
    echo "CPU2TENSOR_SOURCE_SHA must be a full lowercase Git commit." >&2
    exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "The published runtime must be built on Linux x86-64." >&2
    exit 2
fi
if [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != 3.12 ]]; then
    echo "The published runtime must be built with CPython 3.12." >&2
    exit 2
fi
if [[ "$(git -c safe.directory="$root" -C "$root" rev-parse HEAD)" != "$source_sha" ]] ||
   ! git -c safe.directory="$root" -C "$root" diff --quiet ||
   ! git -c safe.directory="$root" -C "$root" diff --cached --quiet; then
    echo "The source checkout does not match the requested clean commit." >&2
    exit 2
fi

readonly bundle="$work/$bundle_name"
readonly wheels="$bundle/wheels"
mkdir -p "$wheels" "$output"

python -m pip wheel --no-deps --wheel-dir "$wheels" "$root"
python -m pip download --only-binary=:all: --dest "$wheels" \
    --index-url "$torch_index" "torch==$torch_version"

cat > "$bundle/install.sh" <<'INSTALL'
#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 /absolute/venv/path" >&2
    exit 2
fi
readonly bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$1" != /* || -e "$1" ]]; then
    echo "The virtual-environment path must be absolute and unused." >&2
    exit 2
fi
cpu2tensor_wheels=("$bundle"/wheels/cpu2tensor-*.whl)
if [[ ${#cpu2tensor_wheels[@]} -ne 1 || ! -f "${cpu2tensor_wheels[0]}" ]]; then
    echo "The bundle must contain exactly one cpu2tensor wheel." >&2
    exit 2
fi
python3.12 -m venv "$1"
"$1/bin/python" -m pip install --no-index --find-links "$bundle/wheels" \
    "torch==2.13.0" "${cpu2tensor_wheels[0]}"
INSTALL
chmod 0755 "$bundle/install.sh"

readonly verify="$work/verify"
"$bundle/install.sh" "$verify"
(
    cd /tmp
    EXPECTED_VERSION="$(python -c 'import tomllib; print(tomllib.load(open("/workspace/pyproject.toml", "rb"))["project"]["version"])')" \
        "$verify/bin/python" - <<'PY'
import importlib.metadata
import os
import platform

import torch
from cpu2tensor import _native

assert platform.machine() == "x86_64"
assert importlib.metadata.version("cpu2tensor") == os.environ["EXPECTED_VERSION"]
assert torch.__version__.startswith("2.13.0")
assert torch.arange(4, dtype=torch.int64).sum().item() == 6
assert _native.new_stream() is not None
PY
)

(
    cd "$wheels"
    sha256sum -- * | LC_ALL=C sort > "$bundle/SHA256SUMS"
)

ROOT="$root" BUNDLE="$bundle" SOURCE_SHA="$source_sha" \
BUILDER_IMAGE_ID="$builder_image_id" TORCH_INDEX="$torch_index" \
TORCH_VERSION="$torch_version" python - <<'PY'
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
import tomllib

root = Path(os.environ["ROOT"])
bundle = Path(os.environ["BUNDLE"])
package_version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]

def record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "filename": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }

manifest = {
    "schema_version": 1,
    "purpose": "cpu2tensor-offline-x86-learner-runtime",
    "source": {
        "repository": "https://github.com/theoad/cpu2tensor",
        "commit": os.environ["SOURCE_SHA"],
        "package_version": package_version,
    },
    "target": {
        "operating_system": "Linux",
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "wheel_abi": "cp312",
    },
    "builder": {
        "image_id": os.environ["BUILDER_IMAGE_ID"],
        "dockerfile_sha256": hashlib.sha256((root / "ci/Dockerfile").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(
            (root / "ci/build-runtime-wheel.sh").read_bytes()
        ).hexdigest(),
    },
    "torch": {
        "requirement": f"torch=={os.environ['TORCH_VERSION']}",
        "index": os.environ["TORCH_INDEX"],
    },
    "files": [record(path) for path in sorted((bundle / "wheels").iterdir())],
    "install_script": record(bundle / "install.sh"),
    "validation": {
        "offline_fresh_venv_install": True,
        "cpu_tensor_smoke": True,
        "native_extension_smoke": True,
    },
}
(bundle / "runtime-manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
PY

readonly archive="$output/$bundle_name.tar.gz"
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
    -C "$work" -cf - "$bundle_name" | gzip -n > "$archive"
(
    cd "$output"
    sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256"
)
