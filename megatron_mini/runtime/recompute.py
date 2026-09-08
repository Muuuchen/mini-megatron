from megatron_mini.exercise import todo

def recomputed(module,*args,enabled=True):
    """T33: enabled 时用 torch.utils.checkpoint.checkpoint(use_reentrant=False)。
    preserve_rng_state=True；disabled 直接 module(*args)。不 detach inputs。
    验收带 dropout 的 output/gradient/RNG一致性，至少一个输入不 requires_grad。
    block 与 attention 两种粒度由构造器选择 wrapper 位置；不是模型存盘。
    """
    todo('T33')
