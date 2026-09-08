"""Saved-tensor offload is independent of parameter/optimizer offload.
This hook covers autograd saves, not arbitrary Python caches (notably SAC caches).
"""
from contextlib import contextmanager
from dataclasses import dataclass
import torch
from megatron_mini.exercise import todo
from megatron_mini.offload.transfer import Transfer

def storage_key(t):
    return (str(t.device),t.untyped_storage().data_ptr())

@dataclass
class SavedActivation:
    transfer: Transfer
    device: torch.device
    shape: tuple
    dtype: torch.dtype

class ActivationOffloader:
    def __init__(self, protected=(), *, pin_memory=False, emulate_cpu=False):
        self.protected = {storage_key(t) for t in protected}
        self.pin_memory, self.emulate_cpu = pin_memory, emulate_cpu
        self.records = []
        self.offloaded_bytes = 0

    def pack(self,tensor):
        """T66: offload floating eligible activations via T65 to an owned CPU snapshot.
        Exclude protected parameter STORAGE (including transposed views), nonfloating
        tensors and CPU tensors unless emulate_cpu=True. Return excluded tensor.detach().
        Eligible -> SavedActivation with device/shape/dtype, append records, count bytes.
        Do not hold a GPU tensor in the saved object except an in-flight Transfer lease.
        Preserve values for noncontiguous views; layout aliasing/in-place mutation is not
        supported by this lab. Do not return the original tensor as its own packed object.
        This test does not prove every activation is freed: other Python/graph references
        may keep it alive. Keep parameter and SAC cache ownership separate.
        """
        todo('T66')

    def unpack(self,packed):
        if isinstance(packed,torch.Tensor): return packed
        host = packed.transfer.wait_host()
        result = host.to(packed.device,dtype=packed.dtype)
        if tuple(result.shape) != packed.shape: raise RuntimeError('corrupt activation shape')
        return result

    def drain(self):
        for rec in self.records: rec.transfer.wait_host()

    @contextmanager
    def hooks(self):
        try:
            with torch.autograd.graph.saved_tensors_hooks(self.pack,self.unpack): yield self
        finally:
            # Simple bounded-lifetime baseline: finish D2H before returning from forward.
            # Backward can still use host snapshots. Async pool retirement is T71 work.
            self.drain()
