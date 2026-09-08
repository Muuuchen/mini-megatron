"""Bridge SAC to the explicit full-shard replay engine, not an outer checkpoint wrapper.
Same local-shard representation as FullShardUnit; first-order gradients only.
"""
import torch
from megatron_mini.exercise import todo
from megatron_mini.fsdp.unit import FullShardUnit
from megatron_mini.runtime.sac import SACStats,save_matmuls_policy

class _SelectiveUnit(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,local_shard,unit):
        """T69: fresh observed_contexts(unit.policy,unit.stats) for THIS invocation.
        Capture CPU/CUDA RNG + autocast state. AG parameters OUTSIDE SAC context;
        run only unit.call(x,full) in caching-forward context. Save x, local_shard,
        recompute context, unit and RNG on ctx; release full/view references.
        Never stash caches on unit where multiple microbatches overwrite each other.
        Do not call ordinary checkpoint inside this no_grad Function.forward: it will
        not establish the non-reentrant SAC lifecycle you need. Enter the paired SAC
        dispatch contexts directly (see observed_contexts), with the matching backward.
        A cached activation must not alias full parameter storage. Constrain unit to
        pure ops and outputs freshly produced from activations, not parameter views.
        """
        todo('T69')

    @staticmethod
    def backward(ctx,grad_output):
        """T70: AG again OUTSIDE cached-replay context. Rebuild local autograd graph
        under enable_grad + saved autocast/RNG; enter ctx's matching recompute context
        around unit.call only. Cached matmul OUTPUTS are reused; gradient matmuls are
        still computed normally. Then autograd.grad, optional grad_sum_axes SUM,
        RS/grad_divisor -> (dx,dlocal,None), with integer input dx=None.
        ctx replay storage must die after this backward; no cross-step stale outputs.
        Preserve subsequent RNG sequence. Reuse lifecycle helpers from T38/T39 instead
        of nesting a second full-unit recomputation or replaying communication twice.
        """
        todo('T70')

class SACFullShardUnit(FullShardUnit):
    def __init__(self,module,axis,grad_divisor=None,grad_sum_axes=(),policy=None):
        super().__init__(module,axis,grad_divisor,grad_sum_axes)
        self.policy = save_matmuls_policy if policy is None else policy
        self.stats = SACStats() # counters only; NEVER cache tensors here

    def forward(self,x):
        return _SelectiveUnit.apply(x,self.local_shard,self)
