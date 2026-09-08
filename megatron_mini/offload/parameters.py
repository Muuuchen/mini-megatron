"""One canonical CPU shard; GPU staging/full buffers are transient."""
import torch
from megatron_mini.exercise import todo
from megatron_mini.memory.residency import ResidencyStats

class CPUParamStore:
    def __init__(self, cpu_shard, axis, device, *, pin_memory=False):
        if cpu_shard.device.type != 'cpu' or cpu_shard.ndim != 1:
            raise ValueError('canonical shard must be a flat CPU tensor/Parameter')
        self.cpu_shard, self.axis = cpu_shard, axis
        self.device = torch.device(device)
        self.pin_memory = pin_memory
        self.staging = self.full = self.transfer = None
        self.stats = ResidencyStats()

    def materialize(self):
        """T64: reject a second live materialization; H2D local CPU shard via T65,
        wait on the consuming CUDA stream (CPU path wait_host), then AG on self.axis.
        Keep temporary self.staging, self.transfer, self.full; update residency stats.
        Return full. NCCL must receive CUDA tensors, not CPU master weights.
        store.cpu_shard must be the SAME owner as CPUShardAdam.master (or the CPU
        local_shard Parameter updated by a CPU optimizer), not a stale copied shadow.
        For baseline retire copy source with wait_host before reusing pinned storage.
        BF16 compute casting is a separate T47 policy; initial test is FP32.
        """
        todo('T64')

    def release(self):
        if self.full is None: raise RuntimeError('nothing to release')
        if self.transfer is not None: self.transfer.wait_host()
        self.full = self.staging = self.transfer = None
        self.stats.live_full_bytes = 0
        self.stats.releases += 1
        # The caller must finish consumers or retain explicit leases until their CUDA
        # events complete. Do not return live buffers to a pool after merely launching work.
