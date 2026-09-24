# SPDX-License-Identifier: AGPL-3.0-only
"""Windows WPR memory captures. ETL decoding is a separate, post-capture step."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from cpu2tensor.hardware import HardwareCaptureError


@dataclass(frozen=True)
class WprConfig:
    """A system-wide ETW hardware capture, held in RAM until stop().

    PMU samples include user and kernel execution; use PID and IP context in
    the ETL to separate them. Processor trace can select either code mode.
    WPR's memory buffers are circular, so completeness is not guaranteed.
    """

    signal: Literal["cycles", "instructions", "processor_trace"] = "cycles"
    code_mode: Literal["User", "Kernel", "UserKernel"] = "UserKernel"
    buffer_kb: int = 1024
    buffers: int = 128
    period: int = 100_000

    def __post_init__(self) -> None:
        if self.signal not in ("cycles", "instructions", "processor_trace"):
            raise ValueError("Unknown WPR hardware signal")
        if self.code_mode not in ("User", "Kernel", "UserKernel"):
            raise ValueError("Unknown processor-trace code mode")
        if self.signal != "processor_trace" and self.code_mode != "UserKernel":
            raise ValueError("WPR PMU sampling cannot restrict code mode at capture")
        if self.buffer_kb <= 0 or self.buffers <= 0 or self.period <= 0:
            raise ValueError("WPR buffers and sampling period must be positive")


@dataclass(frozen=True)
class WprBatch:
    """A raw ETL container, not decoded instruction or sample tensors.

    ETW memory buffers can overwrite old events. ``complete`` is always false
    until a decoder verifies loss and the window fits the available buffers.
    """

    signal: str
    etl_bytes: torch.Tensor
    complete: bool = field(default=False, init=False)


def _profile(config: WprConfig) -> str:
    if config.signal == "processor_trace":
        counter = ("<ProcessorTrace><BufferSize Value=\"32\" />"
                   f"<CodeMode Value=\"{config.code_mode}\" />"
                   "<Events><Event Value=\"SampledProfile\" /></Events>"
                   "</ProcessorTrace>")
        keyword = "SampledProfile"
    else:
        name = "TotalCycles" if config.signal == "cycles" else "InstructionRetired"
        counter = ("<SampledCounters><SampledCounter "
                   f"Value=\"{name}\" Interval=\"{config.period}\" />"
                   "</SampledCounters>")
        keyword = "PmcProfile"
    return f'''<?xml version="1.0" encoding="utf-8"?>
<WindowsPerformanceRecorder Version="1.0" Author="cpu2tensor">
  <Profiles>
    <SystemCollector Id="Cpu2TensorCollector" Name="NT Kernel Logger">
      <BufferSize Value="{config.buffer_kb}" />
      <Buffers Value="{config.buffers}" />
    </SystemCollector>
    <SystemProvider Id="Cpu2TensorSystem">
      <Keywords>
        <Keyword Value="ProcessThread" />
        <Keyword Value="Loader" />
        <Keyword Value="{keyword}" />
      </Keywords>
    </SystemProvider>
    <HardwareCounter Id="Cpu2TensorCounter">{counter}</HardwareCounter>
    <Profile Id="Cpu2Tensor.Verbose.File" Name="Cpu2Tensor"
             Description="cpu2tensor hardware capture" DetailLevel="Verbose"
             LoggingMode="File">
      <Collectors><SystemCollectorId Value="Cpu2TensorCollector">
        <SystemProviderId Value="Cpu2TensorSystem" />
        <HardwareCounterId Value="Cpu2TensorCounter" />
      </SystemCollectorId></Collectors>
    </Profile>
    <Profile Id="Cpu2Tensor.Verbose.Memory" Name="Cpu2Tensor"
             Description="cpu2tensor hardware capture" DetailLevel="Verbose"
             LoggingMode="Memory" Base="Cpu2Tensor.Verbose.File" />
  </Profiles>
</WindowsPerformanceRecorder>
'''


def _run_wpr(*arguments: str) -> None:
    try:
        result = subprocess.run(("wpr", *arguments), capture_output=True, text=True,
                                check=False)
    except OSError as error:
        raise HardwareCaptureError("Windows Performance Recorder is unavailable") from error
    if result.returncode:
        message = (result.stderr or result.stdout).strip()
        raise HardwareCaptureError(f"WPR failed: {message}")


class WprCapture:
    """Start WPR in memory mode and export an ETL only after target execution.

    The caller must stop the measured target before calling stop(), because
    WPR writes the ETL during stop. Only one WPR session should run per host.
    """

    def __init__(self, config: WprConfig) -> None:
        if platform.system() != "Windows":
            raise HardwareCaptureError("WPR capture requires Windows")
        self.config = config
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._started = False
        self._stopped = False

    def __enter__(self) -> "WprCapture":
        if self._started or self._stopped:
            raise RuntimeError("Create a new WprCapture for each run")
        self._directory = tempfile.TemporaryDirectory(prefix="cpu2tensor-wpr-")
        profile = Path(self._directory.name) / "hardware.wprp"
        profile.write_text(_profile(self.config), encoding="utf-8")
        try:
            _run_wpr("-start", f"{profile}!Cpu2Tensor.Verbose")
        except BaseException:
            self.close()
            raise
        self._started = True
        return self

    def stop(self) -> WprBatch:
        if not self._started or self._stopped or self._directory is None:
            raise RuntimeError("Capture must be running and stopped only once")
        self._stopped = True
        output = Path(self._directory.name) / "capture.etl"
        _run_wpr("-stop", str(output))
        self._started = False
        contents = bytearray(output.read_bytes())
        if not contents:
            raise HardwareCaptureError("WPR produced an empty ETL")
        import torch

        return WprBatch(self.config.signal, torch.frombuffer(contents, dtype=torch.uint8))

    def close(self) -> None:
        if self._started:
            _run_wpr("-cancel")
            self._started = False
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None

    def __exit__(self, *_: object) -> None:
        self.close()
