# SPDX-License-Identifier: AGPL-3.0-only
"""Turn CPU execution traces into owned tensor batches."""
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cpu2tensor.batch import AddressContext, Batch, ExecutableLayout, MemoryAccesses, RegisterChanges
    from cpu2tensor.pool import Pool
    from cpu2tensor.stdio import StdioEnv
    from cpu2tensor.kernel import KernelEnv

# Worker-side benchmarks use the native decoder without loading a learner runtime.
_EXPORTS = {
    'ExecutableLayout': 'batch',
    'AddressContext': 'batch', 'Batch': 'batch', 'MemoryAccesses': 'batch',
    'RegisterChanges': 'batch', 'Pool': 'pool', 'StdioEnv': 'stdio', 'KernelEnv': 'kernel',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(f'cpu2tensor.{module}'), name)
    globals()[name] = value
    return value
