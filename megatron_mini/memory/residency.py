"""Synchronous residency exercise. No allocator claim is inferred from this ledger alone."""
from enum import Enum
from dataclasses import dataclass
import torch
from megatron_mini.exercise import todo

class Residency(Enum):
    SHARDED = 'sharded'
    FULL = 'full'

@dataclass
class ResidencyStats:
    gathers: int = 0
    releases: int = 0
    live_full_bytes: int = 0
    peak_full_bytes: int = 0

class ShardHandle:
    def __init__(self, local_shard, axis):
        if local_shard.ndim != 1: raise ValueError('flat local shard required')
        self.local_shard, self.axis = local_shard, axis
        self.state = Residency.SHARDED
        self.full = None
        self.stats = ResidencyStats()

    def materialize(self):
        """T61: SHARDED -> FULL; gather only within self.axis, keep one transient full.
        Repeated materialize before release must raise RuntimeError (no silent extra AG).
        Update gathers/live_full_bytes/peak_full_bytes, return self.full. Parameters and
        optimizer still own only local_shard. Use outside autograd inside T38/T69;
        this helper does not provide a collective backward. Do not gather the whole model.
        """
        todo('T61')

    def release(self):
        if self.state is not Residency.FULL: raise RuntimeError('no full buffer to release')
        self.full = None
        self.state = Residency.SHARDED
        self.stats.releases += 1
        self.stats.live_full_bytes = 0
        # Caller must also release views, autograd saves and aliases. Clearing this
        # reference is NOT proof that CUDA storage is free; verify memory separately.
