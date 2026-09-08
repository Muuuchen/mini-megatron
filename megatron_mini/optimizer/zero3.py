"""Explicit ZeRO-3 teaching frontend, sharing flat-shard primitives with mini FSDP.
Not a DeepSpeed fork or a separate parallel dimension. Native DeepSpeed and native
FSDP2 differ in parameter representation, hooks and scheduling.
"""
from torch import nn
from megatron_mini.exercise import todo

def build_zero3_sequential(module, axis, *, sac=False):
    """T62: return nn.Sequential with SAME immediate child names/order as module.
    One child becomes one FullShardUnit (or SACFullShardUnit when sac=True).
    Require nn.Sequential with parameter-bearing children; reject buffers and tied
    storage across children before wrapping (cross-unit tying is out of scope).
    Reuse T35/T38/T39 rather than copying another implementation. Initialize the
    optimizer AFTER wrapping; it must see only *.local_shard parameters.
    Do not retain original full module on wrapper. Test oracle retains one deliberately.
    For GPT/MoE use T40/T55 adapters rather than pretending every model is Sequential.
    ZeRO-3 step updates local shards; no whole-model updated-parameter AG at step end.
    Wire T61 residency accounting in the shared unit if validating live-byte bounds.
    """
    todo('T62')
