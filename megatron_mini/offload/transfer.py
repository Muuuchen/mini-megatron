"""Host visibility != CUDA-stream ordering. CPU mode only tests semantics."""
from dataclasses import dataclass, field
from typing import Any
import torch
from megatron_mini.exercise import todo

@dataclass
class Transfer:
    tensor: torch.Tensor
    event: Any = None
    keepalive: tuple = field(default_factory=tuple)
    host_ready: bool = False

    def wait_host(self):
        """CPU may inspect/mutate buffers only after this completes; idempotent."""
        if not self.host_ready:
            if self.event is not None: self.event.synchronize()
            self.host_ready = True
        self.keepalive = ()
        return self.tensor

    def wait_stream(self, stream):
        """Queue a dependency, NOT a CPU wait. Keep sources until wait_host retires them."""
        if self.event is not None: stream.wait_event(self.event)
        if self.tensor.is_cuda: self.tensor.record_stream(stream)
        return self.tensor


def submit_copy(source, device, *, pin_memory=False, stream=None):
    """T65: return Transfer containing a NEW detached contiguous value snapshot.
    CPU->CPU: allocate+copy synchronously, no event, host_ready=True; label as emulation.
    CUDA<->CPU: allocate CPU destination pinned when requested; optional dedicated
    CUDA copy stream. Make copy stream wait on producer current stream, enqueue
    copy_, record completion event, keep source/destination alive until retired.
    For asynchronous H2D the CPU source must be pinned; do not mutate/reuse it early.
    A D2H consumer needs wait_host(), not merely wait_stream(). Synchronous fallback
    is valid but do not claim overlap. Reject CUDA->different CUDA device in this lab.
    Do not use cuda.synchronize() every transfer to disguise broken stream ordering.
    Caller must not mutate source before completion; keepalive alone prevents neither
    concurrent writes nor early buffer-pool reuse. T71 must enforce this ownership.
    """
    todo('T65')
