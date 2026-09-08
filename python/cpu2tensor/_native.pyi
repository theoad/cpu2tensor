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
    physical_addresses: bytearray | None
    mapped_sizes: bytearray | None
    mapping_flags: bytearray | None
    context_sequences: bytearray | None

class ContextColumns(TypedDict):
    pc: bytearray
    cr0: bytearray
    cr3: bytearray
    cr4: bytearray
    efer: bytearray
    cs_base: bytearray
    mode: bytearray
    known: bytearray

class MixedRegisterColumns(RegisterColumns):
    sequences: bytearray

class MixedMemoryColumns(MemoryColumns):
    sequences: bytearray

class MixedContextColumns(ContextColumns):
    sequences: bytearray

class MixedColumns(TypedDict, total=False):
    blocks: bytearray
    block_sequences: bytearray
    registers: MixedRegisterColumns
    memory: MixedMemoryColumns
    context: MixedContextColumns

def new_stream() -> Any: ...
def payload_size(header: bytes | bytearray | memoryview) -> int: ...
def decode(stream: Any, frame: bytes | bytearray | memoryview) -> tuple[int, int, int, int, int, bytearray | dict[int, str] | RegisterColumns | MemoryColumns | ContextColumns | MixedColumns]: ...
