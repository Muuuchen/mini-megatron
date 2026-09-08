"""Selective activation checkpointing. Whole-block AC and SAC are different exercises."""
from contextlib import contextmanager
from dataclasses import dataclass,field
from collections import Counter
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import create_selective_checkpoint_contexts
from megatron_mini.exercise import todo

@dataclass
class SACStats:
    forward: Counter = field(default_factory=Counter)
    recompute: Counter = field(default_factory=Counter)

class _ExecutedOps(TorchDispatchMode):
    def __init__(self,counter):
        super().__init__(); self.counter = counter
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        self.counter[str(func)] += 1
        return func(*args,**(kwargs or {}))

def observed_contexts(policy, stats):
    """Provided instrumentation: observer BELOW cache mode counts executed ops,
    not policy invocations/cache lookups. Fresh cache pair for EACH forward invocation.
    Uses a private TorchDispatchMode instrumentation API; validated against local 2.10.
    """
    cache_forward,cache_recompute = create_selective_checkpoint_contexts(policy)
    @contextmanager
    def enter(mode,counter):
        with _ExecutedOps(counter),mode: yield
    return enter(cache_forward,stats.forward),enter(cache_recompute,stats.recompute)

def save_matmuls_policy(ctx,op,*args,**kwargs):
    """T67: MUST_SAVE outputs of aten.mm.default / addmm.default / bmm.default;
    PREFER_RECOMPUTE other pure local ops. Return CheckpointPolicy, not bool.
    Use exact OpOverload values, not torch.mm or OpOverloadPacket.
    This is an MLP teaching policy, NOT a universally optimal attention policy:
    saving an SxS score matrix may defeat the memory goal. Inspect actual SDPA ops.
    No AG/RS/P2P/router state mutation/optimizer side effects inside this pure region.
    Do not set allow_cache_entry_mutation=True to suppress correctness failures.
    """
    todo('T67')

def selective_checkpoint(function,*args,policy=None,stats=None):
    """T68: non-reentrant checkpoint with fresh observed_contexts per invocation,
    preserve_rng_state=True. Default policy=save_matmuls_policy, stats=SACStats().
    Ordinary no_grad inference should execute function directly, not create dead caches.
    Do not keep one cache pair on a module and share it across microbatches/steps.
    Test values, grads, dropout/RNG, and executed matmul counts during recompute.
    SAC caches save selected op outputs, not arbitrary parameter views or full weights.
    """
    todo('T68')
