"""CPU AdamW shard: optimizer compute and persistent master/moments all live on CPU."""
import copy
import torch
from torch import nn
from megatron_mini.exercise import todo

class CPUShardAdam:
    def __init__(self, local_shard, *, lr=1e-3, weight_decay=0.01, pin_memory=False):
        data = local_shard.detach().to(device='cpu',dtype=torch.float32).clone()
        if pin_memory:
            if not torch.cuda.is_available(): raise RuntimeError('pinned-memory path requires CUDA')
            data = data.pin_memory()
        self.master = nn.Parameter(data)
        self.optim = torch.optim.AdamW([self.master],lr=lr,weight_decay=weight_decay)
        self.pin_memory = pin_memory

    def step(self, reduced_grad, *, found_inf=False):
        """T63: input is the FINAL local grad shard after accumulation, RS/divisor,
        unscale and clipping; do not reduce or divide again. found_inf is already a
        globally agreed flag. If True: leave master, m/v and optimizer step unchanged.
        Else stage reduced_grad to CPU with T65, wait for HOST visibility, set an owned
        FP32 master.grad and step CPU AdamW once. Do not keep GPU grad/moments.
        The caller clears step accumulators and publishes an updated GPU compute shard
        when needed. Never reconstruct/update the full parameter on every CPU rank.
        """
        todo('T63')

    def state_dict(self):
        return {'master':self.master.detach().clone(),'optimizer':copy.deepcopy(self.optim.state_dict())}

    def load_state_dict(self,state):
        if state['master'].shape != self.master.shape: raise ValueError('wrong local shard shape')
        with torch.no_grad(): self.master.copy_(state['master'].to('cpu',dtype=torch.float32))
        self.optim.load_state_dict(copy.deepcopy(state['optimizer']))
