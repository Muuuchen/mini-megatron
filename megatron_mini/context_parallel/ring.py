from megatron_mini.exercise import todo

def online_update(m,l,o,scores,value,valid):
    """T28: 一块 scores[...,Q,K] 的 online softmax 更新，返回(m,l,o)。
    m/l:[...,Q,1]；o 是未归一化分子[...,Q,D]；初态 -inf/0/0。
    valid 可广播到 scores，masked scores 视为 -inf。
    合并旧、新 max，指数重标定 l 和 o，禁止每个 KV block 各算 softmax 再加。
    全 masked 的行必须保持 0 分子/分母、不产生 NaN（-inf-(-inf)）。
    最后 output=o/l，仅在确有可见 key 时相除。
    """
    todo('T28')

def ring_attention(q,k,v,axis):
    """T29: 真正的 ring P2P + 自定义 backward。API 同 gather_attention。
    每轮只持有本地与接收 KV buffer，owner ID 随环转动，用全局位置 mask。
    forward 用 T28；backward 再循环 KV，dQ 本地累加、dK/dV 回到 owner。
    保存 q/local kv/output/logsumexp；不能保存全部 rank 的 KV。
    可按 block 重算 scores，不要求手写 FlashAttention CUDA kernel。
    验收: lab ring 对 gather baseline 的值/梯度；双 buffer 生命周期；CP2/4。
    高阶导数不在契约内。CPU只能验语义，真正内存/overlap必须测 CUDA。
    """
    todo('T29')
