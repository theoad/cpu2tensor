from typing import Any, TypedDict

class RegisterColumns(TypedDict):
    pc: bytearray
    ids: bytearray
    widths: bytearray
    flags: bytearray
    values: bytearray
    value_width: int

class MemoryColumns(TypedDict):
    pc: bytearray
    addresses: bytearray
    sizes: bytearray
    flags: bytearray
    values: bytearray | None

def new_stream() -> Any: ...
def payload_size(header: bytes | bytearray | memoryview) -> int: ...
def decode(stream: Any, frame: bytes | bytearray | memoryview) -> tuple[int, int, int, int, int, bytearray | dict[int, str] | RegisterColumns | MemoryColumns]: ...
