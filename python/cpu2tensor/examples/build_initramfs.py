"""Package the static kernel example as a small, rootless initramfs."""

# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import stat
from typing import BinaryIO


def _entry(
    archive: BinaryIO,
    number: int,
    name: str,
    mode: int,
    data: bytes = b"",
    *,
    device_major: int = 0,
    device_minor: int = 0,
) -> None:
    """Write one newc entry; device numbers describe the guest, not the host."""
    encoded_name = name.encode("ascii") + b"\0"
    fields = (
        number,
        mode,
        0,  # root UID inside the guest
        0,  # root GID inside the guest
        2 if stat.S_ISDIR(mode) else 1,
        0,  # fixed timestamp makes the archive reproducible
        len(data),
        0,
        0,
        device_major,
        device_minor,
        len(encoded_name),
        0,
    )
    header = b"070701" + "".join(f"{value:08x}" for value in fields).encode("ascii")
    archive.write(header)
    archive.write(encoded_name)
    archive.write(b"\0" * (-(len(header) + len(encoded_name)) % 4))
    archive.write(data)
    archive.write(b"\0" * (-len(data) % 4))


def build_initramfs(init: Path, output: Path) -> None:
    """Include only the supplied ELF init, its serial devices, and proc."""
    data = init.read_bytes()
    if not data.startswith(b"\x7fELF"):
        raise ValueError("The guest init must be a statically linked Linux ELF executable")
    if init.resolve() == output.resolve():
        raise ValueError("The init executable and output archive must be different files")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        with gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as archive:
            _entry(archive, 1, "init", stat.S_IFREG | 0o755, data)
            _entry(archive, 2, "dev", stat.S_IFDIR | 0o755)
            _entry(
                archive,
                3,
                "dev/console",
                stat.S_IFCHR | 0o600,
                device_major=5,
                device_minor=1,
            )
            _entry(
                archive,
                4,
                "dev/ttyS1",
                stat.S_IFCHR | 0o600,
                device_major=4,
                device_minor=65,
            )
            _entry(archive, 5, "proc", stat.S_IFDIR | 0o555)
            _entry(archive, 6, "TRAILER!!!", 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True, help="Static Linux guest executable")
    parser.add_argument("--output", type=Path, required=True, help="Output .cpio.gz path")
    args = parser.parse_args()
    try:
        build_initramfs(args.init, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"build_initramfs: {error}\n")
    print(args.output)


if __name__ == "__main__":
    main()
